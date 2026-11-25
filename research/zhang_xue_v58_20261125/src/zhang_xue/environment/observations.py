from .shared import *
from .configuration import EnvConfigV54, make_v57_config


class _ObservationMixin:

    def _fused_cell(self, receiver: int, cid: int) -> Tuple[float, float, float]:
        """Receiver-private evidence only. Never read the GCS map for a UAV actor."""
        net=self.net
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
        masses=np.zeros((3,NUM_CELLS),np.float32)
        masses[2].fill(1.0)
        for cid in np.flatnonzero(any_seen):
            masses[:,cid] = self._fused_cell(u,int(cid))
        latest=np.where(valid,self.net.ledger_gen_ts_s[u],-1).max(axis=0)
        age=np.ones(NUM_CELLS,dtype=np.float32)
        age[any_seen]=np.minimum(1.0,np.maximum(0.0,
                              (self.t-latest[any_seen])/self.cfg.t_ref_s))
        loc=np.zeros(NUM_CELLS,np.float32)
        x=int(np.clip(self.positions[u,0]//CELL_SIZE_M,0,GRID_W-1))
        y=int(np.clip(self.positions[u,1]//CELL_SIZE_M,0,GRID_W-1))
        loc[y*GRID_W+x]=1.0
        channels=np.concatenate((masses,age[None,:],loc[None,:]),axis=0)
        side=self.cfg.obs_map_side
        grid=channels.reshape(5,side,2,side,2)
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
            pooled[4]*=4.0
        return pooled.astype(np.float32).reshape(-1)


    def _build_obs_state_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Receiver-private 14 telemetry + 5 spatial maps. No target truth or global stats."""
        obs = np.zeros((ACTOR_N, self.obs_dim), dtype=np.float32)
        mask = np.zeros((ACTOR_N, DISC_DIM), dtype=np.uint8)
        mask[:, 0] = 1
        for u in range(ACTOR_N):
            own_radio = not is_in_blackout(self.positions[u], CANONICAL_BLACKOUT_CENTERS_2)
            own_alive = self.battery_j[u] > 0
            soc = max(0.0, self.battery_j[u] / self.batt_cfg.e_max_j)
            gcs_dx = (self.positions[GCS_ID, 0] - self.positions[u, 0]) / self.cfg.map_size_m
            gcs_dy = (self.positions[GCS_ID, 1] - self.positions[u, 1]) / self.cfg.map_size_m
            gcs_has_route = 1.0 if GCS_ID in self.net.routes[u] else 0.0
            dirty_gcs = self.net.count_all_dirty_entries_non_oracle(u, GCS_ID) / 400.0
            obs[u,:14] = [
                self.positions[u, 0] / self.cfg.map_size_m,
                self.positions[u, 1] / self.cfg.map_size_m,
                self.velocities[u, 0] / self.cfg.v_max_mps,
                self.velocities[u, 1] / self.cfg.v_max_mps,
                soc, float(own_radio and own_alive),
                len(self.net.radios[u].queue) / float(self.net.queue_cap),
                gcs_dx, gcs_dy, math.hypot(gcs_dx, gcs_dy),
                gcs_has_route, dirty_gcs,
                float(np.count_nonzero((self.net.ack_wm_pos[u] > 0) |
                       (self.net.ack_wm_neg[u] > 0))) / float(RADIO_N * NUM_CELLS),
                float(len(self.net.routes[u])) / float(RADIO_N),
            ]
            obs[u,14:] = self._local_belief_features(u)
            for a_idx in range(1, DISC_DIM):
                dst = action_to_dest_radio_id(u, a_idx)
                if own_radio and own_alive and len(self.net.get_dirty_entries_non_oracle(u, dst)) > 0:
                    mask[u, a_idx] = 1
        state=obs.reshape(-1).copy()
        return obs,state,mask
