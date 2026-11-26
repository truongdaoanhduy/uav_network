from .shared import *
from .configuration import EnvConfigV54, make_v57_config
from .observations import _ObservationMixin
from .transition import _TransitionMixin

class ZhangXue7NodePacketEnv(_ObservationMixin, _TransitionMixin):

    def __init__(self, cfg: EnvConfigV54 | None = None, seed: int = 20261125) -> None:
        self.cfg = cfg or EnvConfigV54()
        self.actor_n = int(self.cfg.num_uavs)
        self.gcs_id = self.actor_n
        self.radio_n = self.actor_n + 1
        self.grid_w = int(self.cfg.grid_w)
        self.num_cells = self.grid_w * self.grid_w
        self.cell_size_m = float(self.cfg.map_size_m) / float(self.grid_w)
        self.disc_dim = 3 if self.cfg.comm_action_mode == "compact_3" else (self.actor_n + 1)
        self.obs_dim = 14 + 5 * self.cfg.obs_map_side**2
        self.state_dim = self.actor_n * self.obs_dim
        if self.cfg.evidence_calibration == 'sensor_likelihood_ratio':
            self.evidence_mf, self.evidence_me = sensor_likelihood_to_bba(
                self.cfg.pd, self.cfg.pf, max_mass=self.cfg.bba_reliability_cap
            )
        else:
            self.evidence_mf, self.evidence_me = self.cfg.ds_mf, self.cfg.ds_me
        self.batt_cfg = ZengBatteryConfig(e_max_j=self.cfg.e_max_j)
        self.buildings, self.blackout_centers, self.blackout_r = scale_canonical_geometry(
            self.cfg.map_size_m, self.grid_w, scale_obstacles=self.cfg.scale_obstacles_with_map,
        )
        self.rng = np.random.default_rng(seed)
        self.net = SevenRadioCsmaCaNetwork(
            tx_power_w=self.cfg.tx_power_w, seed=seed,
            noise_power_w=self.cfg.mac_noise_power_w,
            snr_threshold_db=self.cfg.mac_snr_threshold_db,
            cca_threshold_dbm=self.cfg.mac_cca_threshold_dbm,
            queue_cap=self.cfg.mac_queue_capacity,
            protocol=CsmaProtocolConfig(
                difs_us=self.cfg.mac_difs_us, sifs_us=self.cfg.mac_sifs_us,
                slot_us=self.cfg.mac_slot_us, cw_min=self.cfg.mac_cw_min,
                max_retries=self.cfg.mac_max_retries,
                ack_timeout_us=self.cfg.mac_ack_timeout_us,
            ),
            aoi_ewma_end_to_end=self.cfg.aoi_ewma_end_to_end,
            aoi_eta=self.cfg.ds_aoi_eta,
            ds_mf=self.evidence_mf,
            ds_me=self.evidence_me,
            buildings=self.buildings,
            blackout_centers=self.blackout_centers,
            blackout_r=self.blackout_r,
            rf_obstacle_model=self.cfg.rf_obstacle_model,
            rf_per_building_loss_db=self.cfg.rf_per_building_loss_db,
            enable_data_mule_forwarding=self.cfg.enable_data_mule_forwarding,
            coalesce_duplicate_queue=self.cfg.coalesce_duplicate_queue,
            num_uavs=self.actor_n,
            grid_w=self.grid_w,
            cell_size_m=self.cell_size_m,
            ds_fusion_mode=self.cfg.ds_fusion_mode,
        )

        self.positions = np.zeros((self.radio_n, 3), dtype=np.float64)
        self.velocities = np.zeros((self.actor_n, 2), dtype=np.float64)
        self.battery_j = np.full(self.actor_n, self.batt_cfg.e_max_j, dtype=np.float64)
        self.true_targets = np.zeros(self.num_cells, dtype=bool)
        self.team_u_prev = np.ones(self.num_cells, dtype=np.float64)
        self.gcs_tp_latch = np.zeros(self.num_cells, dtype=np.int8)
        self.gcs_fp_latch = np.zeros(self.num_cells, dtype=np.int8)
        self.gcs_fp_ever_latched = np.zeros(self.num_cells, dtype=np.int8)
        self.total_fp_events: int = 0
        self.local_tp_latch = np.zeros(self.num_cells, dtype=np.int8)
        self.discovery_auc_sum: float = 0.0
        self.queue_len_sum: float = 0.0
        self._actor_diag = np.arange(self.actor_n)
        self.t = 0
        self.episode_id = 0
        self.epoch_us = 0

        # AoI accumulators (all 3 definitions)
        self.confirm_latencies_s: List[float] = []
        self.delivered_age_sum_s: float = 0.0
        self.delivered_age_samples: int = 0
        self.mapwide_age_sum_s: float = 0.0
        self.mapwide_age_samples: int = 0


    def _ensure_safe_spawn(self, spawn_scale: float) -> None:
        """Keep the canonical layout when valid; otherwise use physical spacing.

        Scaling the map must not scale the configured UAV clearance. The bounded
        fallback also avoids building footprints in fleets larger than six.
        """
        c = self.cfg
        zeros = np.zeros((self.actor_n, 2), dtype=np.float64)
        alive = np.ones(self.actor_n, dtype=bool)

        def violations():
            return trajectory_safety_violations(
                self.positions[:self.actor_n, :2], zeros, alive, self.buildings,
                map_size_m=c.map_size_m, dt_s=c.dt_s,
                uav_safe_distance_m=c.uav_safe_distance_m,
            )

        if not any(violations().values()):
            return
        spacing = max(self.cell_size_m, c.uav_safe_distance_m + 1e-3)
        axis = np.arange(10.0, c.map_size_m - 10.0 + 1e-6, spacing)
        anchor = (90.0 * spawn_scale, 110.0 * spawn_scale)
        candidates = sorted(
            ((float(x), float(y)) for y in axis for x in axis),
            key=lambda p: ((p[0] - anchor[0])**2 + (p[1] - anchor[1])**2, p[1], p[0]),
        )
        selected = []
        for x, y in candidates:
            if any(math.hypot(x - bx, y - by) <= br + 2.0
                   for bx, by, br, _ in self.buildings):
                continue
            selected.append((x, y))
            if len(selected) == self.actor_n:
                break
        if len(selected) != self.actor_n:
            raise ValueError("Cannot place UAV fleet safely on the deterministic spawn lattice; "
                             "reduce num_uavs or safe distance, or enlarge the map")
        self.positions[:self.actor_n, :2] = np.asarray(selected)
        if any(violations().values()):
            raise ValueError("Spawn geometry violates physical clearance")


    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.net.rng = np.random.default_rng(seed + 101)
        self.episode_id += 1
        self.t = 0
        self.positions.fill(0.0)
        spawn_scale = (self.cfg.map_size_m / 1000.0) if self.cfg.scale_obstacles_with_map else 1.0
        cols = max(1, min(self.actor_n, int((self.cfg.map_size_m - 180.0 * spawn_scale) // (65.0 * spawn_scale))))
        for u in range(self.actor_n):
            col = u % cols
            row = u // cols
            self.positions[u] = [
                spawn_scale * (90.0 + 65.0 * col),
                spawn_scale * (110.0 + 30.0 * (col % 2) + 75.0 * row),
                self.cfg.uav_alt_m,
            ]
        self._ensure_safe_spawn(spawn_scale)
        # Scale the fixed GCS *and* spawn geometry together with buildings.
        # Otherwise 2/3 km maps silently relocate the GCS in normalized space.
        self.positions[self.gcs_id] = np.array(
            [GCS_POS_M[0] * spawn_scale, GCS_POS_M[1] * spawn_scale, GCS_POS_M[2]],
            dtype=np.float64,
        )
        self.velocities.fill(0.0)
        self.battery_j.fill(self.batt_cfg.e_max_j)

        self.true_targets.fill(False)
        # Bounded placement: reject impossible configurations instead of hanging
        # forever in a rejection loop (num_targets can be <=400 yet infeasible).
        gcs_x, gcs_y = self.positions[self.gcs_id, :2]
        legal_cells = []
        for cid in range(self.num_cells):
            cx = (cid % self.grid_w + 0.5) * self.cell_size_m
            cy = (cid // self.grid_w + 0.5) * self.cell_size_m
            if any(math.hypot(cx - bx, cy - by) <= br for bx, by, br, _ in self.buildings):
                continue
            if math.hypot(cx - gcs_x, cy - gcs_y) < self.cfg.gcs_target_exclusion_m:
                continue
            legal_cells.append(cid)
        if len(legal_cells) < self.cfg.num_targets:
            raise ValueError(
                f"Not enough legal target cells: {len(legal_cells)} < {self.cfg.num_targets}"
            )
        if self.cfg.target_min_separation_m == 0.0:
            chosen = self.rng.choice(legal_cells, size=self.cfg.num_targets, replace=False)
            self.true_targets[chosen] = True
        else:
            selected_xy = []
            chosen = []
            for cid in self.rng.permutation(legal_cells):
                xy = np.array([((cid % self.grid_w) + 0.5) * self.cell_size_m,
                               ((cid // self.grid_w) + 0.5) * self.cell_size_m])
                if all(float(np.linalg.norm(xy - other)) >= self.cfg.target_min_separation_m
                       for other in selected_xy):
                    selected_xy.append(xy)
                    chosen.append(int(cid))
                    if len(chosen) == self.cfg.num_targets:
                        break
            if len(chosen) < self.cfg.num_targets:
                raise ValueError(
                    f"Only {len(chosen)} targets can be placed under min separation "
                    f"{self.cfg.target_min_separation_m} m with this randomized placement; "
                    "reduce count/separation or use a larger map"
                )
            self.true_targets[chosen] = True

        self.team_u_prev.fill(1.0)
        self.gcs_tp_latch.fill(0)
        self.gcs_fp_latch.fill(0)
        self.gcs_fp_ever_latched.fill(0)
        self.total_fp_events = 0
        self.local_tp_latch.fill(0)
        self.discovery_auc_sum = 0.0
        self.queue_len_sum = 0.0
        self.confirm_latencies_s.clear()
        self.delivered_age_sum_s = 0.0
        self.delivered_age_samples = 0
        self.mapwide_age_sum_s = 0.0
        self.mapwide_age_samples = 0

        self.net.reset(self.positions, episode_id=self.episode_id)
        # Warmup 0.6 s of HELLO/DV discovery before step 0 so initial 1-hop routes exist
        if not self.net.run_until_us(600_000):
            raise RuntimeError("Network HELLO warmup exceeded event budget")
        # Keep a strictly monotonic network clock; epoch is used only for mission AoI.
        self.epoch_us = self.net.sim_time_us
        self.net.mission_epoch_us = self.epoch_us
        self.net.step_tx_airtime_us.fill(0)
        self.net.step_new_peer_cells = 0
        self.net.step_peer_info_gain = 0.0
        for u in range(self.actor_n):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))
        return self._build_obs_state_mask()
