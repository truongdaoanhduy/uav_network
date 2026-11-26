from .shared import *
from .configuration import EnvConfigV54, make_v57_config
from .aoi_metrics import fixed_horizon_aoi_no_new_deliveries


class _TransitionMixin:

    def _active_local_routes(self, actor_id: int) -> Dict[int, Tuple[int, int, int, int]]:
        """Read usable HELLO routes without pruning the actor's local table."""
        now_us = self.net.sim_time_us
        return {dst: entry for dst, entry in self.net.routes[actor_id].items()
                if entry[1] > 0 and entry[3] > now_us}

    def _resolve_comm_dest(self, actor_id: int, action_idx: int) -> int:
        if not (0 <= action_idx < self.disc_dim) or not (0 <= actor_id < self.actor_n):
            raise ValueError(f"Invalid comm action {action_idx} for actor {actor_id}")
        if action_idx == 0:
            return -1
        if self.cfg.comm_action_mode == "compact_3":
            if action_idx == 1:
                return self.gcs_id
            if not self.cfg.enable_peer_dissemination:
                return -1
            # action_idx == 2 (SHARE_PEERS): decentralized HELLO-route peer or nearest alive peer
            if self.cfg.compact3_decentralized_peer_select:
                best_peer = -1
                best_hops = 10**9
                for p_id, r_entry in sorted(self._active_local_routes(actor_id).items()):
                    if not (0 <= p_id < self.actor_n) or p_id == actor_id:
                        continue
                    if self.net.count_all_dirty_entries_non_oracle(actor_id, p_id) <= 0:
                        continue
                    hops = int(r_entry[1])
                    if hops < best_hops:
                        best_hops = hops
                        best_peer = p_id
                return best_peer
            best_peer = -1
            best_dist = float("inf")
            for p_id in range(self.actor_n):
                if p_id == actor_id or self.battery_j[p_id] <= 0.0:
                    continue
                if self.net.count_all_dirty_entries_non_oracle(actor_id, p_id) <= 0:
                    continue
                d = float(np.linalg.norm(self.positions[p_id, :2] - self.positions[actor_id, :2]))
                if d < best_dist:
                    best_dist = d
                    best_peer = p_id
            return best_peer
        if action_idx == self.actor_n:
            return self.gcs_id
        if not self.cfg.enable_peer_dissemination:
            return -1
        peer = action_idx - 1
        return peer + 1 if peer >= actor_id else peer

    def step(
        self,
        act_vel_cmd: np.ndarray,
        act_comm_disc: np.ndarray,
        tx_power_w: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, Dict[str, float]]:
        c = self.cfg
        self.net.step_tx_airtime_us.fill(0)
        self.net.step_new_peer_cells = 0
        self.net.step_peer_info_gain = 0.0
        for u in range(self.actor_n):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))

        act_vel_cmd = np.asarray(act_vel_cmd, dtype=np.float64)
        act_comm_disc = np.asarray(act_comm_disc)
        if act_vel_cmd.shape != (self.actor_n, 2) or not np.isfinite(act_vel_cmd).all():
            raise ValueError(f"act_vel_cmd must be finite shape ({self.actor_n}, 2)")
        if act_comm_disc.shape != (self.actor_n,):
            raise ValueError(f"act_comm_disc must have shape ({self.actor_n},)")
        if not np.issubdtype(act_comm_disc.dtype, np.integer):
            raise ValueError("act_comm_disc must contain integers")
        if np.any((act_comm_disc < 0) | (act_comm_disc >= self.disc_dim)):
            raise ValueError(f"act_comm_disc must contain actions in [0, {self.disc_dim - 1}]")

        if (tx_power_w is None) == bool(c.enable_tx_power_control):
            raise ValueError("TX power action must match enable_tx_power_control")
        if tx_power_w is not None:
            tx_p = np.asarray(tx_power_w, dtype=np.float64)
            if tx_p.shape != (self.actor_n,) or not np.isfinite(tx_p).all():
                raise ValueError("TX power action must be finite shape (num_uavs,)")
            if np.any((tx_p < c.tx_power_min_w) | (tx_p > c.tx_power_max_w)):
                raise ValueError("TX power outside configured range")
            self.net.set_tx_power_per_uav(tx_p)
        else:
            tx_p = None
        # 1. Enqueue prior-step dirty evidence per actor comm action (strictly non-oracle)
        for u in range(self.actor_n):
            dst = self._resolve_comm_dest(u, int(act_comm_disc[u]))
            if dst >= 0 and self.battery_j[u]>0:
                entries = self.net.get_dirty_entries_non_oracle(u, dst)
                if entries:
                    self.net.enqueue_app_data(u, dst, entries)

        # 2. Advance REAL 7-radio CSMA/CA discrete-event network for dt_s (1,000,000 us)
        target_us = self.net.sim_time_us + int(round(c.dt_s * 1_000_000))
        if c.network_substeps == 1:
            if not self.net.run_until_us(target_us):
                raise RuntimeError("CSMA/CA event cap hit: refusing to advance RL state with stale network time")
        elif c.network_substeps == 2:
            # Use start and midpoint geometry in two equal radio sub-intervals.
            # Save/restore the physical state: the authoritative flight update,
            # safety and energy accounting still happen together below.
            initial_positions = self.positions.copy()
            half_time = self.net.sim_time_us + (target_us - self.net.sim_time_us) // 2
            if not self.net.run_until_us(half_time):
                raise RuntimeError("CSMA/CA first radio substep hit event cap")
            planned_vel, _ = kinematic_safety_shield(
                initial_positions[:self.actor_n, :2], self.velocities.copy(), act_vel_cmd,
                self.battery_j > 0.0, self.buildings, c.map_size_m,
                c.dt_s, c.v_max_mps, c.a_max_mps2, c.uav_safe_distance_m,
            )
            try:
                self.positions[:self.actor_n, :2] = (
                    initial_positions[:self.actor_n, :2] + 0.5 * c.dt_s * planned_vel
                )
                self.net.update_positions(self.positions)
                if not self.net.run_until_us(target_us):
                    raise RuntimeError("CSMA/CA midpoint radio substep hit event cap")
            finally:
                self.positions[:] = initial_positions
                self.net.update_positions(self.positions)
        else:
            raise ValueError("network_substeps must be 1 or 2")

        # 3. Compute GCS TP/FP latch & AoI rewards from ACTUAL packets received at GCS
        r_gcs_tp = r_gcs_aoi = r_gcs_fp = 0.0
        t_step_s = self.t + 1
        for cid in range(self.num_cells):
            if not np.any(self.net.ledger_pos[self.gcs_id,:,cid]) and not np.any(self.net.ledger_neg[self.gcs_id,:,cid]):
                continue
            m_t, _, m_u = self._fused_cell(self.gcs_id, cid, now_s=float(t_step_s))
            if m_t >= c.confirm_th and m_u <= c.ds_uncertainty_ceiling:
                if bool(self.true_targets[cid]) and self.gcs_tp_latch[cid] == 0:
                    self.gcs_tp_latch[cid] = 1
                    # End-step confirmation freshness at the latest novel receipt;
                    # this is not an exact intra-step threshold-crossing timestamp.
                    rx_t_s = (self.net.gcs_last_novel_rx_us[cid] - self.epoch_us) / 1e6
                    lat_s = max(0.0, rx_t_s - float(self.net.gcs_latest_u_s[cid]))
                    self.confirm_latencies_s.append(lat_s)
                    r_gcs_tp += c.r_tp
                    if c.aoi_reward_mode == 'exponential':
                        r_gcs_aoi += c.w_aoi * math.exp(-lat_s / c.t_ref_s)
                    else:
                        r_gcs_aoi += c.w_aoi * max(0.0, 1.0 - lat_s / c.t_ref_s)
                elif (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 0:
                    self.gcs_fp_latch[cid] = 1
                    self.total_fp_events += 1
                    if self.gcs_fp_ever_latched[cid] == 0:
                        self.gcs_fp_ever_latched[cid] = 1
                        r_gcs_fp -= c.r_fp
            elif c.enable_fp_unlatch and (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 1:
                self.gcs_fp_latch[cid] = 0

        # Update all 3 AoI accumulators at end of step
        current_mapwide_age_sum_s = 0.0
        for cid in range(self.num_cells):
            u_s = int(self.net.gcs_latest_u_s[cid])
            if u_s >= 0:
                age = max(0.0, float(t_step_s - u_s))
                self.delivered_age_sum_s += age
                self.delivered_age_samples += 1
            else:
                age = float(t_step_s)
            current_mapwide_age_sum_s += age
            self.mapwide_age_sum_s += age
            self.mapwide_age_samples += 1
        current_mapwide_aoi_s = current_mapwide_age_sum_s / self.num_cells

        # 4. Motion update: v5.7 Acceleration-constrained shield with post-step swept collision detection
        old_xy_all = self.positions[:self.actor_n, :2].copy()
        old_vel_all = self.velocities.copy()
        alive_mask = self.battery_j > 0.0
        shield_intervention = np.zeros(self.actor_n, dtype=np.float64)
        safety_violations = {"boundary": [], "building": [], "pairwise": []}
        safety_violation = False

        v_safe_all, shield_intervention = kinematic_safety_shield(
            old_xy_all, old_vel_all, act_vel_cmd, alive_mask,
            self.buildings, c.map_size_m, c.dt_s,
            c.v_max_mps, c.a_max_mps2, c.uav_safe_distance_m,
        )
        rejected_motion_count = int(np.count_nonzero((shield_intervention > 1e-6) & alive_mask))
        safety_violations = trajectory_safety_violations(
            old_xy_all, v_safe_all, alive_mask, self.buildings,
            map_size_m=c.map_size_m, dt_s=c.dt_s,
            uav_safe_distance_m=c.uav_safe_distance_m,
        )
        safety_violation = any(safety_violations.values())
        # One infeasible actor must not suppress sensing for every *other* UAV.
        # Pairwise violations affect both participants; GCS/team penalties remain global.
        unsafe_sensing_uavs = set(int(u) for u in safety_violations["boundary"])
        unsafe_sensing_uavs.update(int(u) for u, _ in safety_violations["building"])
        for i, j in safety_violations["pairwise"]:
            unsafe_sensing_uavs.update((int(i), int(j)))
        for u in range(self.actor_n):
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
                tx_power_w=float(tx_p[u]) if (tx_p is not None and c.enable_tx_power_control) else float(c.tx_power_w),
            ))
            if self.battery_j[u] <= 0.0:
                self.net.set_radio_alive(u, False)

        self.net.update_positions(self.positions)

        # 5. Local sensing & team-union exploration reward
        for u in range(self.actor_n):
            if self.battery_j[u] <= 0.0 or u in unsafe_sensing_uavs:
                continue
            ux, uy = self.positions[u, 0], self.positions[u, 1]
            cmin_x = max(0, int((ux - c.fov_radius_m) // self.cell_size_m))
            cmax_x = min(self.grid_w - 1, int((ux + c.fov_radius_m) // self.cell_size_m))
            cmin_y = max(0, int((uy - c.fov_radius_m) // self.cell_size_m))
            cmax_y = min(self.grid_w - 1, int((uy + c.fov_radius_m) // self.cell_size_m))
            for cy in range(cmin_y, cmax_y + 1):
                for cx in range(cmin_x, cmax_x + 1):
                    cell_x = (cx + 0.5) * self.cell_size_m
                    cell_y = (cy + 0.5) * self.cell_size_m
                    if math.hypot(cell_x - ux, cell_y - uy) <= c.fov_radius_m:
                        if c.enable_3d_sensing_occlusion:
                            # C2 fix: skip cells inside building footprint or occluded by 70-80m buildings
                            occluded = False
                            for bx, by, br, bh in self.buildings:
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
                        cid = cy * self.grid_w + cx
                        det = bool(self.rng.random() < (c.pd if self.true_targets[cid] else c.pf))
                        self.net.record_local_observation(u, cid, det, t_step_s)

        r_loc_team = 0.0
        if c.aoi_ewma_end_to_end:
            from ..belief.dempster_shafer import fuse_ewma_provenance_uq88
        for cid in range(self.num_cells):
            if c.aoi_ewma_end_to_end:
                own_p = self.net.ledger_pos[self._actor_diag, self._actor_diag, cid]
                own_n = self.net.ledger_neg[self._actor_diag, self._actor_diag, cid]
                if np.any(own_p) or np.any(own_n):
                    m_t_team, _, u_now = fuse_ewma_provenance_uq88(
                        own_p, own_n,
                        self.net.ledger_gen_ts_s[self._actor_diag, self._actor_diag, cid],
                        float(t_step_s), mf=self.evidence_mf, me=self.evidence_me,
                        eta=c.ds_aoi_eta if c.ds_fusion_mode == 'sum' else 0.0,
                        mode=c.ds_fusion_mode,
                    )
                    # Monotone minimum-uncertainty watermark prevents oscillatory
                    # reward farming when EWMA aging raises u_now between revisits.
                    if u_now < self.team_u_prev[cid]:
                        r_loc_team += c.w_loc_team * (self.team_u_prev[cid] - u_now)
                        self.team_u_prev[cid] = u_now
                    if (bool(self.true_targets[cid]) and self.local_tp_latch[cid] == 0
                            and m_t_team >= c.confirm_th and u_now <= c.ds_uncertainty_ceiling):
                        self.local_tp_latch[cid] = 1
            else:
                if c.ds_fusion_mode == 'zhang_max':
                    n_p = int(max(self.net.ledger_pos[u, u, cid] for u in range(self.actor_n)))
                    n_n = int(max(self.net.ledger_neg[u, u, cid] for u in range(self.actor_n)))
                else:
                    n_p = int(sum(self.net.ledger_pos[u, u, cid] for u in range(self.actor_n)))
                    n_n = int(sum(self.net.ledger_neg[u, u, cid] for u in range(self.actor_n)))
                if n_p > 0 or n_n > 0:
                    m_t_team, _, u_now = counts_to_canonical_ds_stable(n_p, n_n, self.evidence_mf, self.evidence_me)
                    r_loc_team += c.w_loc_team * (self.team_u_prev[cid] - u_now)
                    self.team_u_prev[cid] = u_now
                    if (bool(self.true_targets[cid]) and self.local_tp_latch[cid] == 0
                            and m_t_team >= c.confirm_th and u_now <= c.ds_uncertainty_ceiling):
                        self.local_tp_latch[cid] = 1

        # Active airtime & FP hold penalties
        if tx_p is not None and c.enable_tx_power_control:
            w_p = 0.5 + 0.5 * (tx_p / max(1e-6, c.tx_power_max_w))
            r_airtime = -c.lambda_b * float(np.sum((self.net.step_tx_airtime_us[:self.actor_n] / 1e6) * w_p))
        else:
            uav_airtime_total_s = float(self.net.step_tx_airtime_us[:self.actor_n].sum()) / 1e6
            r_airtime = -c.lambda_b * uav_airtime_total_s
        # Reward *new, actual, receiver-integrated* peer information, not transmissions.
        if c.peer_reward_mode == 'info_gain':
            r_peer_information = c.w_p2p * min(float(c.reward_peer_cap_per_step), self.net.step_peer_info_gain)
        else:
            r_peer_information = c.w_p2p * min(c.reward_peer_cap_per_step, self.net.step_new_peer_cells)
        r_fp_hold = -c.fp_hold * float(self.gcs_fp_latch.sum())

        if c.reward_scale_mode == 'per_target_norm':
            t_scale = 15.0 / float(max(1, c.num_targets))
            c_scale = 400.0 / float(max(1, self.num_cells))
            r_gcs_tp *= t_scale
            r_gcs_aoi *= t_scale
            r_loc_team *= c_scale
        r_cont_aoi = -c.continuous_aoi_penalty_w * (current_mapwide_aoi_s / float(c.t_ref_s))
        r_blocked_motion = -c.w_blocked_motion * float(shield_intervention.sum())
        r_safety_failure = -c.safety_failure_penalty if safety_violation else 0.0
        reward = (r_loc_team + r_gcs_tp + r_gcs_aoi + r_gcs_fp + r_fp_hold + r_cont_aoi
                  + r_airtime + r_blocked_motion + r_peer_information + r_safety_failure)
        self.t += 1

        battery_mission_done = (bool(np.any(self.battery_j <= 0.0))
                                if c.battery_termination_mode == 'any_depleted'
                                else bool(np.all(self.battery_j <= 0.0)))
        terminated = bool(
            np.all(self.gcs_tp_latch[self.true_targets] == 1) or battery_mission_done
            or (safety_violation and c.safety_failure_terminate)
        )
        truncated = bool((not terminated) and (self.t >= c.safety_horizon_s))
        obs, state, mask = self._build_obs_state_mask()

        gcs_tp_now = int(self.gcs_tp_latch[self.true_targets].sum())
        local_tp_now = int(self.local_tp_latch[self.true_targets].sum())
        self.discovery_auc_sum += gcs_tp_now / max(1, c.num_targets)
        self.queue_len_sum += sum(len(self.net.radios[u].queue) for u in range(self.actor_n)) / float(self.actor_n)
        obs_pct = float(np.any(self.net.ledger_gen_ts_s[self._actor_diag, self._actor_diag, :] >= 0, axis=0).mean() * 100.0)
        enq_gcs = self.net.stats["data_enqueued_gcs"]
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
            "unsafe_sensing_uavs": sorted(unsafe_sensing_uavs),
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
            "current_gcs_aoi_mapwide_censored_s": round(current_mapwide_aoi_s, 3),
            # Counterfactual terminal padding: no additional GCS deliveries
            # after early termination. Never confuse with measured AoI.
            "mean_gcs_aoi_fixed_horizon_no_future_delivery_s": (
                round(fixed_horizon_aoi_no_new_deliveries(
                    observed_sum_cell_seconds=self.mapwide_age_sum_s,
                    current_mean_cell_age_s=current_mapwide_aoi_s,
                    t=self.t, horizon=c.safety_horizon_s, num_cells=self.num_cells,
                ), 3) if terminated or truncated else None
            ),
            "min_battery_pct": round(float(self.battery_j.min() / self.batt_cfg.e_max_j * 100.0), 2),
            "battery_termination_mode": c.battery_termination_mode,
            "active_uavs": int(np.count_nonzero(self.battery_j > 0.0)),
            "data_delivered_gcs": self.net.stats["data_delivered_gcs"],
            "mac_ack_ok": self.net.stats["mac_ack_ok"],
            "e2e_ack_rx": self.net.stats["e2e_ack_rx"],
            "local_tp_discovered": local_tp_now,
            "true_area_observed_pct": round(obs_pct, 2),
            "pdr_gcs": round(self.net.stats["data_delivered_gcs"] / enq_gcs, 4) if enq_gcs > 0 else -1.0,
            "mean_queue_len": round(self.queue_len_sum / max(1, self.t), 4),
            "discovery_auc": round(self.discovery_auc_sum / max(1, self.t), 4),
            "r_continuous_aoi": round(r_cont_aoi, 4),
            "false_confirm_events": int(self.total_fp_events),
            "precision": round(gcs_tp_now / max(1, gcs_tp_now + int(self.gcs_fp_latch.sum())), 4) if (gcs_tp_now + int(self.gcs_fp_latch.sum())) > 0 else 1.0,
            "recall": round(gcs_tp_now / max(1, c.num_targets), 4),
            "f1": round((2.0 * gcs_tp_now) / max(1, 2 * gcs_tp_now + int(self.gcs_fp_latch.sum()) + (c.num_targets - gcs_tp_now)), 4),
            "false_confirm_rate": round(int(self.gcs_fp_ever_latched.sum()) / max(1, self.num_cells - c.num_targets), 4),
        }
        return obs, state, mask, float(reward), terminated, truncated, info
