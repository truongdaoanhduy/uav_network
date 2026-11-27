from .shared import *
from .configuration import EnvConfigV54, make_v57_config


class _ObservationMixin:

    def _fused_cell(self, receiver: int, cid: int, now_s: float | None = None) -> Tuple[float, float, float]:
        """Receiver-private evidence only. Never read the GCS map for a UAV actor."""
        net = self.net
        t_eval = float(self.t) if now_s is None else float(now_s)
        if self.cfg.belief_inference_mode == "bayesian_logodds":
            # Same independent-cell log-likelihood update as the Warp backend:
            # origin-indexed cumulative RAW counts, never a DS BBA.
            valid = net.ledger_gen_ts_s[receiver, :, cid] >= 0
            p = float(net.ledger_pos[receiver, valid, cid].astype(np.float64).sum())
            n = float(net.ledger_neg[receiver, valid, cid].astype(np.float64).sum())
            prior = self.cfg.bayes_prior_occupancy
            z = (math.log(prior / (1.0 - prior))
                 + p * math.log(self.cfg.pd / self.cfg.pf)
                 + n * math.log((1.0 - self.cfg.pd) / (1.0 - self.cfg.pf)))
            z = max(-40.0, min(40.0, z))
            posterior = 1.0 / (1.0 + math.exp(-z))
            # Coverage epistemic score (not a DS mass); matches Warp's 0.5**N.
            return posterior, 1.0 - posterior, math.exp(-math.log(2.0) * (p + n))
        if self.cfg.aoi_ewma_end_to_end:
            from ..belief.dempster_shafer import fuse_ewma_provenance_uq88
            return fuse_ewma_provenance_uq88(
                net.ledger_pos[receiver, :, cid], net.ledger_neg[receiver, :, cid],
                net.ledger_gen_ts_s[receiver, :, cid], t_eval,
                mf=self.evidence_mf, me=self.evidence_me,
                eta=self.cfg.ds_aoi_eta if self.cfg.ds_fusion_mode == "sum" else 0.0,
                mode=self.cfg.ds_fusion_mode,
            )
        return fuse_provenance_counts(
            net.ledger_pos[receiver,:,cid],net.ledger_neg[receiver,:,cid],
            net.ledger_gen_ts_s[receiver,:,cid],net.ledger_rx_age_s[receiver,:,cid],
            mf=self.evidence_mf,me=self.evidence_me,
            eta=self.cfg.ds_aoi_eta if self.cfg.ds_fusion_mode=='sum' else 0.0,
            mode=self.cfg.ds_fusion_mode,
        )


    def _local_belief_features(self, u: int) -> np.ndarray:
        """5x10x10 compressed map (20x20 source grid, 2x2 pooling).

        Channels: mass target, mass empty, mass unknown, normalized source AoI,
        own UAV location. All belief evidence is present in receiver u's ledger.
        Unknown cells have mass (0,0,1), AoI 1; no ground truth enters actor.
        """
        src_p=self.net.ledger_pos[u]
        src_n=self.net.ledger_neg[u]
        valid=(src_p.astype(np.int32)+src_n.astype(np.int32))>0
        any_seen=np.any(valid,axis=0)
        masses=np.zeros((3,self.num_cells),np.float32)
        masses[2].fill(1.0)
        if self.cfg.belief_inference_mode == "bayesian_logodds":
            # Warp maps retain the prior posterior on never-sensed cells.
            masses[0].fill(self.cfg.bayes_prior_occupancy)
            masses[1].fill(1.0 - self.cfg.bayes_prior_occupancy)
        for cid in np.flatnonzero(any_seen):
            masses[:,cid] = self._fused_cell(u,int(cid))
        latest=np.where(valid,self.net.ledger_gen_ts_s[u],-1).max(axis=0)
        age=np.ones(self.num_cells,dtype=np.float32)
        age[any_seen]=np.minimum(1.0,np.maximum(0.0,
                              (self.t-latest[any_seen])/self.cfg.t_ref_s))
        loc=np.zeros(self.num_cells,np.float32)
        x=int(np.clip(self.positions[u,0]//self.cell_size_m,0,self.grid_w-1))
        y=int(np.clip(self.positions[u,1]//self.cell_size_m,0,self.grid_w-1))
        loc[y*self.grid_w+x]=1.0
        channels=np.concatenate((masses,age[None,:],loc[None,:]),axis=0)
        side=self.cfg.obs_map_side
        factor=self.grid_w//side
        grid=channels.reshape(5,side,factor,side,factor)
        if self.cfg.obs_pool_mode == 'semantic':
            pooled=np.empty((5,side,side),dtype=np.float32)
            pooled[0]=grid[0].max(axis=(1,3))   # Target mass: preserve peak (no 4x dilution)
            pooled[1]=grid[1].min(axis=(1,3))   # Empty mass: conservative min
            pooled[2]=grid[2].mean(axis=(1,3))  # Unknown mass: mean unexplored density
            pooled[3]=grid[3].max(axis=(1,3))   # AoI: prioritize stalest cell
            pooled[4]=grid[4].max(axis=(1,3))   # Own location: exact one-hot
        else:
            pooled=grid.mean(axis=(2,4))
            # The pooled location is upweighted to preserve a one-hot 10x10 actor cue.
            pooled[4] *= float(factor * factor)
        return pooled.astype(np.float32).reshape(-1)


    def _build_obs_state_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Receiver-private 14 telemetry + 5 spatial maps. No target truth or global stats."""
        obs = np.zeros((self.actor_n, self.obs_dim), dtype=np.float32)
        mask = np.zeros((self.actor_n, self.disc_dim), dtype=np.uint8)
        mask[:, 0] = 1
        for u in range(self.actor_n):
            own_radio = not is_in_blackout(self.positions[u], self.blackout_centers, manhattan_r=self.blackout_r, grid_w=self.grid_w, cell_size_m=self.cell_size_m)
            own_alive = self.battery_j[u] > 0
            soc = max(0.0, self.battery_j[u] / self.batt_cfg.e_max_j)
            gcs_dx = (self.positions[self.gcs_id, 0] - self.positions[u, 0]) / self.cfg.map_size_m
            gcs_dy = (self.positions[self.gcs_id, 1] - self.positions[u, 1]) / self.cfg.map_size_m
            active_routes = self._active_local_routes(u)
            gcs_has_route = 1.0 if self.gcs_id in active_routes else 0.0
            dirty_gcs = min(1.0, self.net.count_all_dirty_entries_non_oracle(u, self.gcs_id) / float(self.num_cells))
            obs[u,:14] = [
                self.positions[u, 0] / self.cfg.map_size_m,
                self.positions[u, 1] / self.cfg.map_size_m,
                self.velocities[u, 0] / self.cfg.v_max_mps,
                self.velocities[u, 1] / self.cfg.v_max_mps,
                soc, float(own_radio and own_alive),
                len(self.net.radios[u].queue) / float(self.net.queue_cap),
                gcs_dx, gcs_dy, math.hypot(gcs_dx, gcs_dy),
                gcs_has_route, dirty_gcs,
                (float(np.count_nonzero(self.net.ack_wm_origin_ts[u] >= 0)) / float(self.radio_n * self.actor_n * self.num_cells)
                 if (self.net.enable_data_mule_forwarding and self.net.ack_wm_origin_ts is not None)
                 else float(np.count_nonzero((self.net.ack_wm_pos[u] > 0) | (self.net.ack_wm_neg[u] > 0))) / float(self.radio_n * self.num_cells)),
                float(len(active_routes)) / float(self.radio_n),
            ]
            obs[u,14:] = self._local_belief_features(u)
            if self.cfg.comm_action_mode == "compact_3":
                if own_radio and own_alive:
                    if self.gcs_id in active_routes and self.net.count_all_dirty_entries_non_oracle(u, self.gcs_id) > 0:
                        mask[u, 1] = 1
                    # Peer share legal if any eligible peer has dirty evidence
                    if self.cfg.enable_peer_dissemination:
                        cand_peers = (
                            [p_id for p_id in active_routes if 0 <= p_id < self.actor_n and p_id != u]
                            if self.cfg.compact3_decentralized_peer_select
                            else [p_id for p_id in range(self.actor_n) if p_id != u]
                        )
                        if any(
                            (self.cfg.compact3_decentralized_peer_select or self.battery_j[p_id] > 0.0)
                            and self.net.count_all_dirty_entries_non_oracle(u, p_id) > 0
                            for p_id in cand_peers
                        ):
                            mask[u, 2] = 1
            else:
                for a_idx in range(1, self.disc_dim):
                    dst = self._resolve_comm_dest(u, a_idx)
                    if dst >= 0 and own_radio and own_alive and self.net.count_all_dirty_entries_non_oracle(u, dst) > 0:
                        mask[u, a_idx] = 1
        state=obs.reshape(-1).copy()
        return obs,state,mask
