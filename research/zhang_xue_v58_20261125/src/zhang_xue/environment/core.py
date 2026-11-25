from .shared import *
from .configuration import EnvConfigV54, make_v57_config
from .observations import _ObservationMixin
from .transition import _TransitionMixin

class ZhangXue7NodePacketEnv(_ObservationMixin, _TransitionMixin):

    def __init__(self, cfg: EnvConfigV54 | None = None, seed: int = 20261125) -> None:
        self.cfg = cfg or EnvConfigV54()
        self.obs_dim = 14 + 5 * self.cfg.obs_map_side**2
        self.state_dim = ACTOR_N * self.obs_dim
        if self.cfg.evidence_calibration == 'sensor_likelihood_ratio':
            self.evidence_mf, self.evidence_me = sensor_likelihood_to_bba(
                self.cfg.pd, self.cfg.pf, max_mass=self.cfg.bba_reliability_cap
            )
        else:
            self.evidence_mf, self.evidence_me = self.cfg.ds_mf, self.cfg.ds_me
        self.batt_cfg = ZengBatteryConfig(e_max_j=self.cfg.e_max_j)
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
        )

        self.positions = np.zeros((RADIO_N, 3), dtype=np.float64)
        self.velocities = np.zeros((ACTOR_N, 2), dtype=np.float64)
        self.battery_j = np.full(ACTOR_N, self.batt_cfg.e_max_j, dtype=np.float64)
        self.true_targets = np.zeros(NUM_CELLS, dtype=bool)
        self.team_u_prev = np.ones(NUM_CELLS, dtype=np.float64)
        self.gcs_tp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
        self.gcs_fp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
        self.t = 0
        self.episode_id = 0
        self.epoch_us = 0

        # AoI accumulators (all 3 definitions)
        self.confirm_latencies_s: List[float] = []
        self.delivered_age_sum_s: float = 0.0
        self.delivered_age_samples: int = 0
        self.mapwide_age_sum_s: float = 0.0
        self.mapwide_age_samples: int = 0


    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.net.rng = np.random.default_rng(seed + 101)
        self.episode_id += 1
        self.t = 0
        self.positions.fill(0.0)
        for u in range(ACTOR_N):
            self.positions[u] = [90.0 + 65.0 * u, 110.0 + 30.0 * (u % 2), self.cfg.uav_alt_m]
        self.positions[GCS_ID] = np.array(GCS_POS_M, dtype=np.float64)
        self.velocities.fill(0.0)
        self.battery_j.fill(self.batt_cfg.e_max_j)

        self.true_targets.fill(False)
        # Bounded placement: reject impossible configurations instead of hanging
        # forever in a rejection loop (num_targets can be <=400 yet infeasible).
        gcs_x, gcs_y = self.positions[GCS_ID, :2]
        legal_cells = []
        for cid in range(NUM_CELLS):
            cx = (cid % GRID_W + 0.5) * CELL_SIZE_M
            cy = (cid // GRID_W + 0.5) * CELL_SIZE_M
            if any(math.hypot(cx - bx, cy - by) <= br for bx, by, br, _ in CANONICAL_BUILDINGS_5):
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
                xy = np.array([((cid % GRID_W) + 0.5) * CELL_SIZE_M,
                               ((cid // GRID_W) + 0.5) * CELL_SIZE_M])
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
        for u in range(ACTOR_N):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))
        return self._build_obs_state_mask()
