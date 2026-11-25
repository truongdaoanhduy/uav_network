from .shared import *
from .configuration import EnvConfigV54, make_v57_config


class _TransitionMixin:

    def step(
        self, act_vel_cmd: np.ndarray, act_comm_disc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, Dict[str, float]]:
        c = self.cfg
        self.net.step_tx_airtime_us.fill(0)
        self.net.step_new_peer_cells = 0
        self.net.step_peer_info_gain = 0.0
        for u in range(ACTOR_N):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))

        act_vel_cmd = np.asarray(act_vel_cmd, dtype=np.float64)
        act_comm_disc = np.asarray(act_comm_disc)
        if act_vel_cmd.shape != (ACTOR_N, 2) or not np.isfinite(act_vel_cmd).all():
            raise ValueError("act_vel_cmd must be finite shape (6, 2)")
        if act_comm_disc.shape != (ACTOR_N,):
            raise ValueError("act_comm_disc must have shape (6,)")
        if not np.issubdtype(act_comm_disc.dtype, np.integer):
            raise ValueError("act_comm_disc must contain integers")
        if np.any((act_comm_disc < 0) | (act_comm_disc >= DISC_DIM)):
            raise ValueError("act_comm_disc must contain actions in [0, 6]")

        # 1. Enqueue prior-step dirty evidence per actor comm action (strictly non-oracle)
        for u in range(ACTOR_N):
            dst = action_to_dest_radio_id(u, int(act_comm_disc[u]))
            if dst >= 0 and self.battery_j[u]>0:
                entries = self.net.get_dirty_entries_non_oracle(u, dst)
                if entries:
                    self.net.enqueue_app_data(u, dst, entries)

        # 2. Advance REAL 7-radio CSMA/CA discrete-event network for dt_s (1,000,000 us)
        target_us = self.net.sim_time_us + int(round(c.dt_s * 1_000_000))
        if not self.net.run_until_us(target_us):
            raise RuntimeError("CSMA/CA event cap hit: refusing to advance RL state with stale network time")

        # 3. Compute GCS TP/FP latch & AoI rewards from ACTUAL packets received at GCS
        r_gcs_tp = r_gcs_aoi = r_gcs_fp = 0.0
        t_step_s = self.t + 1
        for cid in range(NUM_CELLS):
            if not np.any(self.net.ledger_pos[GCS_ID,:,cid]) and not np.any(self.net.ledger_neg[GCS_ID,:,cid]):
                continue
            m_t, _, m_u = self._fused_cell(GCS_ID, cid)
            if m_t >= c.confirm_th and m_u <= c.ds_uncertainty_ceiling:
                if bool(self.true_targets[cid]) and self.gcs_tp_latch[cid] == 0:
                    self.gcs_tp_latch[cid] = 1
                    rx_t_s = (self.net.gcs_last_rx_us[cid] - self.epoch_us) / 1e6
                    lat_s = max(0.0, rx_t_s - float(self.net.gcs_latest_u_s[cid]))
                    self.confirm_latencies_s.append(lat_s)
                    r_gcs_tp += c.r_tp
                    if c.aoi_reward_mode == 'exponential':
                        r_gcs_aoi += c.w_aoi * math.exp(-lat_s / c.t_ref_s)
                    else:
                        r_gcs_aoi += c.w_aoi * max(0.0, 1.0 - lat_s / c.t_ref_s)
                elif (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 0:
                    self.gcs_fp_latch[cid] = 1
                    r_gcs_fp -= c.r_fp
            elif c.enable_fp_unlatch and (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 1:
                self.gcs_fp_latch[cid] = 0

        # Update all 3 AoI accumulators at end of step
        for cid in range(NUM_CELLS):
            u_s = int(self.net.gcs_latest_u_s[cid])
            if u_s >= 0:
                age = max(0.0, float(t_step_s - u_s))
                self.delivered_age_sum_s += age
                self.delivered_age_samples += 1
                self.mapwide_age_sum_s += age
            else:
                self.mapwide_age_sum_s += float(t_step_s)
            self.mapwide_age_samples += 1

        # 4. Motion update: v5.7 Acceleration-constrained shield with post-step swept collision detection
        old_xy_all = self.positions[:ACTOR_N, :2].copy()
        old_vel_all = self.velocities.copy()
        alive_mask = self.battery_j > 0.0
        shield_intervention = np.zeros(ACTOR_N, dtype=np.float64)
        safety_violations = {"boundary": [], "building": [], "pairwise": []}
        safety_violation = False

        v_safe_all, shield_intervention = kinematic_safety_shield(
            old_xy_all, old_vel_all, act_vel_cmd, alive_mask,
            CANONICAL_BUILDINGS_5, c.map_size_m, c.dt_s,
            c.v_max_mps, c.a_max_mps2, c.uav_safe_distance_m,
        )
        rejected_motion_count = int(np.count_nonzero((shield_intervention > 1e-6) & alive_mask))
        safety_violations = trajectory_safety_violations(
            old_xy_all, v_safe_all, alive_mask, CANONICAL_BUILDINGS_5,
            map_size_m=c.map_size_m, dt_s=c.dt_s,
            uav_safe_distance_m=c.uav_safe_distance_m,
        )
        safety_violation = any(safety_violations.values())
        for u in range(ACTOR_N):
            if alive_mask[u]:
                self.velocities[u] = v_safe_all[u]
                self.positions[u, :2] = old_xy_all[u] + v_safe_all[u] * c.dt_s
            else:
                self.velocities[u] = np.zeros(2, dtype=np.float64)
            actual_speed = float(np.linalg.norm(self.velocities[u]))
            own_airtime_s = float(self.net.step_tx_airtime_us[u]) / 1e6
            self.battery_j[u] = max(0.0, self.battery_j[u] - uav_step_energy_j(
                actual_speed, own_airtime_s, c.dt_s, self.batt_cfg,
                v_prev_xy=old_vel_all[u] if c.enable_accel_energy else None,
                v_new_xy=self.velocities[u] if c.enable_accel_energy else None,
            ))
            if self.battery_j[u] <= 0.0:
                self.net.set_radio_alive(u, False)

        self.net.update_positions(self.positions)

        # 5. Local sensing & team-union exploration reward
        for u in range(ACTOR_N):
            if self.battery_j[u] <= 0.0:
                continue
            ux, uy = self.positions[u, 0], self.positions[u, 1]
            cmin_x = max(0, int((ux - c.fov_radius_m) // CELL_SIZE_M))
            cmax_x = min(GRID_W - 1, int((ux + c.fov_radius_m) // CELL_SIZE_M))
            cmin_y = max(0, int((uy - c.fov_radius_m) // CELL_SIZE_M))
            cmax_y = min(GRID_W - 1, int((uy + c.fov_radius_m) // CELL_SIZE_M))
            for cy in range(cmin_y, cmax_y + 1):
                for cx in range(cmin_x, cmax_x + 1):
                    cell_x = (cx + 0.5) * CELL_SIZE_M
                    cell_y = (cy + 0.5) * CELL_SIZE_M
                    if math.hypot(cell_x - ux, cell_y - uy) <= c.fov_radius_m:
                        if c.enable_3d_sensing_occlusion:
                            # C2 fix: skip cells inside building footprint or occluded by 70-80m buildings
                            occluded = False
                            for bx, by, br, bh in CANONICAL_BUILDINGS_5:
                                if math.hypot(cell_x - bx, cell_y - by) <= br:
                                    occluded = True
                                    break
                                if segment_hits_cylinder_3d_fp32(
                                    self.positions[u], np.array([cell_x, cell_y, 0.0]), bx, by, br, bh
                                ):
                                    occluded = True
                                    break
                            if occluded:
                                continue
                        cid = cy * GRID_W + cx
                        det = bool(self.rng.random() < (c.pd if self.true_targets[cid] else c.pf))
                        self.net.record_local_observation(u, cid, det, t_step_s)

        r_loc_team = 0.0
        for cid in range(NUM_CELLS):
            if c.ds_fusion_mode == 'zhang_max':
                # B10 fix: respect zhang_max fusion mode in team exploration reward
                n_p = int(max(self.net.ledger_pos[u, u, cid] for u in range(ACTOR_N)))
                n_n = int(max(self.net.ledger_neg[u, u, cid] for u in range(ACTOR_N)))
            else:
                n_p = int(sum(self.net.ledger_pos[u, u, cid] for u in range(ACTOR_N)))
                n_n = int(sum(self.net.ledger_neg[u, u, cid] for u in range(ACTOR_N)))
            if n_p > 0 or n_n > 0:
                _, _, u_now = counts_to_canonical_ds_stable(n_p, n_n, self.evidence_mf, self.evidence_me)
                # Signed potential difference avoids rewarding noise-induced oscillations.
                r_loc_team += c.w_loc_team * (self.team_u_prev[cid] - u_now)
                self.team_u_prev[cid] = u_now

        # Active airtime & FP hold penalties
        uav_airtime_total_s = float(self.net.step_tx_airtime_us[:ACTOR_N].sum()) / 1e6
        r_airtime = -c.lambda_b * uav_airtime_total_s
        # Reward *new, actual, receiver-integrated* peer information, not transmissions.
        if c.peer_reward_mode == 'info_gain':
            r_peer_information = c.w_p2p * min(float(c.reward_peer_cap_per_step), self.net.step_peer_info_gain)
        else:
            r_peer_information = c.w_p2p * min(c.reward_peer_cap_per_step, self.net.step_new_peer_cells)
        r_fp_hold = -c.fp_hold * float(self.gcs_fp_latch.sum())

        r_blocked_motion = -c.w_blocked_motion * float(shield_intervention.sum())
        r_safety_failure = -c.safety_failure_penalty if safety_violation else 0.0
        reward = (r_loc_team + r_gcs_tp + r_gcs_aoi + r_gcs_fp + r_fp_hold
                  + r_airtime + r_blocked_motion + r_peer_information + r_safety_failure)
        self.t += 1

        terminated = bool(
            np.all(self.gcs_tp_latch[self.true_targets] == 1) or np.any(self.battery_j <= 0.0)
            or (safety_violation and c.safety_failure_terminate)
        )
        truncated = bool((not terminated) and (self.t >= c.safety_horizon_s))
        obs, state, mask = self._build_obs_state_mask()

        info = {
            "step": self.t,
            "terminated": terminated,
            "truncated": truncated,
            "gcs_tp_confirmed": int(self.gcs_tp_latch[self.true_targets].sum()),
            "gcs_fp_latched": int(self.gcs_fp_latch.sum()),
            "r_loc_team": round(r_loc_team, 4),
            "r_gcs_tp": round(r_gcs_tp, 4),
            "r_gcs_aoi": round(r_gcs_aoi, 4),
            "r_airtime": round(r_airtime, 5),
            "r_peer_information": round(r_peer_information,5),
            "peer_new_cells_rx": int(self.net.step_new_peer_cells),
            "ds_fusion_mode":c.ds_fusion_mode,
            "ds_aoi_eta":c.ds_aoi_eta,
            "evidence_calibration":c.evidence_calibration,
            "r_blocked_motion": round(r_blocked_motion, 4),
            "r_safety_failure": float(r_safety_failure),
            "safety_violation": bool(safety_violation),
            "safety_violations": safety_violations,
            "rejected_motion_count": rejected_motion_count,
            "mean_first_tp_confirm_latency_s": (
                round(float(np.mean(self.confirm_latencies_s)), 4) if self.confirm_latencies_s else -1.0
            ),
            "mean_gcs_aoi_delivered_cells_s": (
                round(self.delivered_age_sum_s / self.delivered_age_samples, 3)
                if self.delivered_age_samples > 0
                else -1.0
            ),
            "delivered_cell_coverage_pct": round(float((self.net.gcs_latest_u_s >= 0).mean() * 100.0), 2),
            # For never-delivered cells we use elapsed mission time as an explicit
            # initialization convention; this is NOT ground-truth AoI.
            "mean_gcs_aoi_mapwide_censored_s": round(self.mapwide_age_sum_s / max(1, self.mapwide_age_samples), 3),
            "min_battery_pct": round(float(self.battery_j.min() / self.batt_cfg.e_max_j * 100.0), 2),
            "data_delivered_gcs": self.net.stats["data_delivered_gcs"],
            "mac_ack_ok": self.net.stats["mac_ack_ok"],
            "e2e_ack_rx": self.net.stats["e2e_ack_rx"],
        }
        return obs, state, mask, float(reward), terminated, truncated, info
