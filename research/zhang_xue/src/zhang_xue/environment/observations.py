from .shared import *
from ..communication.common import ROUTE_TTL_US
from .configuration import EnvConfigV54, make_v57_config


class _ObservationMixin:

    def _fused_cell(self, receiver: int, cid: int, now_s: float | None = None) -> Tuple[float, float, float]:
        """Receiver-private evidence only. Never read the GCS map for a UAV actor."""
        net = self.net
        t_eval = float(self.t) if now_s is None else float(now_s)
        if self.cfg.use_uq88_ewma:
            from ..belief.dempster_shafer import fuse_ewma_provenance_uq88
            return fuse_ewma_provenance_uq88(
                net.ledger_pos[receiver, :, cid], net.ledger_neg[receiver, :, cid],
                net.ledger_gen_ts_s[receiver, :, cid], t_eval,
                mf=self.evidence_mf, me=self.evidence_me,
                eta=self.cfg.ds_aoi_eta if self.cfg.ds_fusion_mode == "sum" else 0.0,
                mode=self.cfg.ds_fusion_mode,
            )
        if self.cfg.static_evidence_mode == "raw_u16_count":
            from ..belief.dempster_shafer import fuse_raw_provenance_counts
            return fuse_raw_provenance_counts(
                net.ledger_pos[receiver, :, cid], net.ledger_neg[receiver, :, cid],
                net.ledger_gen_ts_s[receiver, :, cid],
                mf=self.evidence_mf, me=self.evidence_me,
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

        Channels: target summary, empty summary, unknown summary, normalized source AoI,
        own UAV location. All belief evidence is present in receiver u's ledger.
        NOTE (S06): `obs_pool_mode='semantic'` (max m_T, min m_F, mean m_U) produces
        non-simplex actor features (sum in (0, 1.75]), NOT a pooled DS BBA; exact
        DS simplex (m_T+m_F+m_U=1) is preserved on the underlying 20x20 grid.
        """
        src_p=self.net.ledger_pos[u]
        src_n=self.net.ledger_neg[u]
        valid=(src_p.astype(np.int32)+src_n.astype(np.int32))>0
        any_seen=np.any(valid,axis=0)
        masses=np.zeros((3,self.num_cells),np.float32)
        masses[2].fill(1.0)
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
            map_end = 14 + 5 * self.cfg.obs_map_side**2
            obs[u, 14:map_end] = self._local_belief_features(u)
            if self.cfg.obs_schema_v2:
                obs[u, map_end:map_end + 35] = self._hello_peer_features(u)
            if self.cfg.enable_gimbal_azimuth_action:
                # CPU-only experiment; explicit 551-dimensional observation ABI.
                obs[u, -2:] = [math.sin(float(self.gimbal_azimuth_rad[u])),
                               math.cos(float(self.gimbal_azimuth_rad[u]))]
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

    def _hello_peer_features(self, u: int) -> np.ndarray:
        """35-dim non-oracle HELLO peer telemetry matching Warp observe_hello_peers_v89."""
        feat = np.zeros(35, dtype=np.float32)
        now = int(self.net.sim_time_us)
        x, y = float(self.positions[u, 0]), float(self.positions[u, 1])
        vx, vy = float(self.velocities[u, 0]), float(self.velocities[u, 1])
        cands = []
        for peer in range(self.actor_n):
            if peer == u:
                continue
            sent_at = int(self.net.hello_rx_at_us[u, peer])
            if sent_at < 0 or now - sent_at >= ROUTE_TTL_US:
                continue
            px, py, pvx, pvy = self.net.hello_rx_pose[u, peer]
            dx, dy = float(px) - x, float(py) - y
            cands.append((dx * dx + dy * dy, peer, dx, dy, float(pvx) - vx, float(pvy) - vy, max(0, now - sent_at)))
        cands.sort(key=lambda item: (item[0], item[1]))
        for rank in range(5):
            off = rank * 7
            feat[off + 5] = 1.0
            feat[off + 6] = 1.0
            if rank < len(cands):
                d2, _, dx, dy, dvx, dvy, age = cands[rank]
                feat[off + 0] = dx / self.cfg.map_size_m
                feat[off + 1] = dy / self.cfg.map_size_m
                feat[off + 2] = dvx / self.cfg.v_max_mps
                feat[off + 3] = dvy / self.cfg.v_max_mps
                feat[off + 4] = 1.0
                feat[off + 5] = min(1.0, max(0.0, float(age) / float(ROUTE_TTL_US)))
                feat[off + 6] = math.sqrt(d2) / self.cfg.map_size_m
        return feat
