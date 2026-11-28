from .common import *


class _NetworkLifecycleMixin:

    def __init__(
        self,
        tx_power_w: float = 0.5,
        noise_power_w: float = 4.0e-11,
        snr_threshold_db: float = 6.0,
        cca_threshold_dbm: float = -82.0,
        queue_cap: int = QUEUE_CAP_PKTS,
        buildings: Sequence[Tuple[float, float, float, float]] = CANONICAL_BUILDINGS_5,
        blackout_centers: Sequence[Tuple[int, int]] = CANONICAL_BLACKOUT_CENTERS_2,
        seed: int = 20261125,
        protocol: CsmaProtocolConfig | None = None,
        aoi_ewma_end_to_end: bool = False,
        aoi_eta: float = 0.02,
        ds_mf: float = 0.80,
        ds_me: float = 0.80,
        num_uavs: int = ACTOR_N,
        grid_w: int = GRID_W,
        cell_size_m: float = CELL_SIZE_M,
        blackout_r: int = CANONICAL_BLACKOUT_R,
        enable_data_mule_forwarding: bool = False,
        ds_fusion_mode: str = "sum",
        coalesce_duplicate_queue: bool = False,
        rf_obstacle_model: str = "binary_nlos",
        rf_per_building_loss_db: float = 12.0,
        static_evidence_mode: str = "ewma_uq88",
        obs_schema_v2: bool = False,
    ) -> None:
        if not (1 <= int(num_uavs) <= 30):
            raise ValueError(f"num_uavs must be in [1, 30], got {num_uavs}")
        self.actor_n = int(num_uavs)
        self.gcs_id = self.actor_n
        self.radio_n = self.actor_n + 1
        self.grid_w = int(grid_w)
        self.cell_size_m = float(cell_size_m)
        self.num_cells = self.grid_w * self.grid_w
        self.node_mask_all = (1 << self.radio_n) - 1
        self.scalable_wire = (self.actor_n > 8 or self.num_cells > 512)
        self.tx_power_w = float(tx_power_w)
        self.tx_power_per_radio = np.full(self.radio_n, float(tx_power_w), dtype=np.float64)
        self.aoi_ewma_end_to_end = bool(aoi_ewma_end_to_end)
        self.static_evidence_mode = str(static_evidence_mode)
        self.use_uq88_ewma = bool(self.aoi_ewma_end_to_end and self.static_evidence_mode == "ewma_uq88")
        self.obs_schema_v2 = bool(obs_schema_v2)
        self.aoi_eta = float(aoi_eta)
        self.ds_mf = float(ds_mf)
        self.ds_me = float(ds_me)
        self.ds_fusion_mode = str(ds_fusion_mode)
        self.coalesce_duplicate_queue = bool(coalesce_duplicate_queue)
        self._origin_cursor: Dict[Tuple[int, int], int] = {}
        self.noise_power_w = float(noise_power_w)
        self.snr_th_lin = 10.0 ** (snr_threshold_db / 10.0)
        self.cca_th_w = 10.0 ** ((cca_threshold_dbm - 30.0) / 10.0)
        self.queue_cap = int(queue_cap)
        self.protocol = protocol if protocol is not None else CsmaProtocolConfig()
        if not isinstance(self.protocol, CsmaProtocolConfig):
            raise TypeError("protocol must be CsmaProtocolConfig")
        self.buildings = list(buildings)
        self.blackout_centers = list(blackout_centers)
        self.blackout_r = int(blackout_r)
        if rf_obstacle_model not in ("binary_nlos", "multi_obstacle", "rf_opaque"):
            raise ValueError("rf_obstacle_model must be binary_nlos, multi_obstacle, or rf_opaque")
        if not math.isfinite(rf_per_building_loss_db) or rf_per_building_loss_db < 0:
            raise ValueError("rf_per_building_loss_db must be finite and non-negative")
        self.rf_obstacle_model = rf_obstacle_model
        self.rf_per_building_loss_db = float(rf_per_building_loss_db)
        self.enable_data_mule_forwarding = bool(enable_data_mule_forwarding)
        self.rng = np.random.default_rng(seed)

        self.episode_id: int = 0
        self.sim_time_us: int = 0
        self._next_pkt_id: int = 1
        self._next_app_seq: int = 1

        self.positions = np.zeros((self.radio_n, 3), dtype=np.float64)
        self.positions[self.gcs_id] = np.array(GCS_POS_M, dtype=np.float64)
        self.velocities = np.zeros((self.actor_n, 2), dtype=np.float64)
        self.hello_rx_pose = np.zeros((self.actor_n, self.actor_n, 4), dtype=np.float64)
        self.hello_rx_at_us = np.full((self.actor_n, self.actor_n), -1, dtype=np.int64)
        self.chan_gain = np.zeros((self.radio_n, self.radio_n), dtype=np.float64)
        self.radio_alive = np.ones(self.radio_n, dtype=bool)

        self.radios: List[RadioState] = [RadioState() for _ in range(self.radio_n)]
        # Distributed path-vector routing table: routes[r][dst] = (next_hop, hop_dist, path_mask, expire_us)
        self.routes: List[Dict[int, Tuple[int, int, int, int]]] = [{} for _ in range(self.radio_n)]
        self.hello_next_us: List[int] = [int(self.rng.integers(0, 50_000)) for _ in range(self.radio_n)]

        # Provenance ledgers: [self.radio_n, self.actor_n, self.num_cells, 2] and gen_ts_s [self.radio_n, self.actor_n, self.num_cells]
        self.ledger_pos = np.zeros((self.radio_n, self.actor_n, self.num_cells), dtype=np.uint16)
        self.ledger_neg = np.zeros((self.radio_n, self.actor_n, self.num_cells), dtype=np.uint16)
        self.ledger_gen_ts_s = np.full((self.radio_n, self.actor_n, self.num_cells), -1, dtype=np.int32)
        # Frozen delivery-age of the latest cumulative snapshot for Xue-style DS discount.
        # Locally generated evidence has zero reception delay; relayed evidence is delayed.
        self.ledger_rx_age_s = np.zeros((self.radio_n, self.actor_n, self.num_cells), dtype=np.float32)
        self.mission_epoch_us = 0
        # Actual novel information delivered to a peer this step, never raw packet count.
        self.step_new_peer_cells = 0
        self.step_peer_info_gain = 0.0
        self.peer_h_min = np.full((self.actor_n, self.num_cells), math.log2(3.0), dtype=np.float64)

        # Non-oracle sender E2E ACK watermarks: [self.actor_n, self.radio_n, self.num_cells]
        self.ack_wm_pos = np.zeros((self.actor_n, self.radio_n, self.num_cells), dtype=np.uint16)
        self.ack_wm_neg = np.zeros((self.actor_n, self.radio_n, self.num_cells), dtype=np.uint16)
        self.ack_wm_ts = np.full((self.actor_n, self.radio_n, self.num_cells), -1, dtype=np.int32)
        # v6.5 True Data-Mule multi-origin E2E ACK watermarks: [src, dst, origin, cell]
        if self.enable_data_mule_forwarding:
            self.ack_wm_origin_pos = np.zeros((self.actor_n, self.radio_n, self.actor_n, self.num_cells), dtype=np.uint16)
            self.ack_wm_origin_neg = np.zeros((self.actor_n, self.radio_n, self.actor_n, self.num_cells), dtype=np.uint16)
            self.ack_wm_origin_ts = np.full((self.actor_n, self.radio_n, self.actor_n, self.num_cells), -1, dtype=np.int32)
        else:
            self.ack_wm_origin_pos = None
            self.ack_wm_origin_neg = None
            self.ack_wm_origin_ts = None
        # Sender pending E2E table: (src, app_seq) -> (dst, packed_entries)
        self.e2e_pending: Dict[Tuple[int, int], Tuple[int, Tuple[Tuple[int, int], ...]]] = {}
        # Independent time-bounded metadata: pending E2E records must not grow without limit.
        self.e2e_pending_expires_us: Dict[Tuple[int, int], int] = {}
        # Application-level dedup survives MAC forwarding with a NEW hop-local pkt_id.
        self.delivered_app_expires_us: Dict[Tuple[int, int, int], int] = {}

        # GCS receiver-side AoI timestamps (microsecond event receipt & generation time)
        self.gcs_latest_u_s = np.full(self.num_cells, -1, dtype=np.int32)
        self.gcs_last_rx_us = np.full(self.num_cells, -1, dtype=np.int64)
        # Latest novel origin snapshot, even when its generation predates the freshest origin.
        self.gcs_last_novel_rx_us = np.full(self.num_cells, -1, dtype=np.int64)

        # Per-radio TX airtime (us) — resets each env step by caller
        self.step_tx_airtime_us = np.zeros(self.radio_n, dtype=np.int64)

        # Telemetry & drop counters
        self.stats: Dict[str, int] = {
            "data_enqueued": 0,
            "data_enqueued_gcs": 0,
            "data_delivered_dst": 0,
            "data_delivered_gcs": 0,
            "mac_ack_ok": 0,
            "mac_ack_timeout": 0,
            "e2e_ack_rx": 0,
            "drop_qfull": 0,
            "drop_ttl": 0,
            "drop_retx": 0,
            "drop_noroute": 0,
            "drop_loop": 0,
            "dedup_hits": 0,
            "half_duplex_drops": 0,
            "sinr_drops": 0,
            "events_processed": 0,
        }
        # Test hook: force-drop the next MAC ACK from a given node (for DATA-rx + MAC-ACK-lost test)
        self.force_drop_next_mac_ack_from: Optional[int] = None


    def reset(self, positions_7x3: np.ndarray, episode_id: int = 0) -> None:
        self.episode_id = int(episode_id)
        self.sim_time_us = 0
        self._next_pkt_id = 1
        self._next_app_seq = 1
        self.positions = np.asarray(positions_7x3, dtype=np.float64).copy()
        self.velocities.fill(0.0)
        self.hello_rx_pose.fill(0.0)
        self.hello_rx_at_us.fill(-1)
        self.radios = [RadioState() for _ in range(self.radio_n)]
        self.routes = [{} for _ in range(self.radio_n)]
        self.hello_next_us = [int(self.rng.integers(0, 20_000)) for _ in range(self.radio_n)]
        self.ledger_pos.fill(0)
        self.ledger_neg.fill(0)
        self.ledger_gen_ts_s.fill(-1)
        self.ledger_rx_age_s.fill(0.0)
        self.mission_epoch_us = 0
        self.step_new_peer_cells = 0
        self.step_peer_info_gain = 0.0
        self.peer_h_min.fill(math.log2(3.0))
        self.tx_power_per_radio.fill(self.tx_power_w)
        self.radio_alive.fill(True)
        self.ack_wm_pos.fill(0)
        self.ack_wm_neg.fill(0)
        self.ack_wm_ts.fill(-1)
        if self.enable_data_mule_forwarding and self.ack_wm_origin_pos is not None:
            self.ack_wm_origin_pos.fill(0)
            self.ack_wm_origin_neg.fill(0)
            self.ack_wm_origin_ts.fill(-1)
        self.e2e_pending.clear()
        self.e2e_pending_expires_us.clear()
        self.delivered_app_expires_us.clear()
        self._origin_cursor.clear()
        self.gcs_latest_u_s.fill(-1)
        self.gcs_last_rx_us.fill(-1)
        self.gcs_last_novel_rx_us.fill(-1)
        self.step_tx_airtime_us.fill(0)
        for k in self.stats:
            self.stats[k] = 0
        self.update_channel_matrix()


    def set_tx_power_per_uav(self, powers_6: Sequence[float]) -> None:
        arr = np.asarray(powers_6, dtype=np.float64)
        if arr.shape != (self.actor_n,) or not np.isfinite(arr).all() or np.any(arr <= 0.0):
            raise ValueError("powers_6 must be finite positive shape (6,)")
        self.tx_power_per_radio[:self.actor_n] = arr

    def update_positions(self, positions_7x3: np.ndarray, velocities: np.ndarray | None = None) -> None:
        self.positions = np.asarray(positions_7x3, dtype=np.float64).copy()
        if velocities is not None:
            self.velocities = np.asarray(velocities, dtype=np.float64).copy()
        self.update_channel_matrix()


    def set_radio_alive(self, radio_id: int, alive: bool) -> None:
        if not (0 <= radio_id < self.radio_n):
            raise ValueError(f"Invalid radio_id {radio_id}")
        alive_b = bool(alive)
        if self.radio_alive[radio_id] != alive_b:
            self.radio_alive[radio_id] = alive_b
            if not alive_b:
                r = self.radios[radio_id]
                r.mac_state = ST_IDLE
                r.state_end_us = INF_TIME_US
                r.active_pkt = None
                r.active_next_hop = -1
                r.pending_hello = None
                r.ack_pend_dst = -1
                r.rx_locked_tx = -1
                r.rx_locked_pkt = None
                r.queue.clear()
                self.routes[radio_id].clear()
            self.update_channel_matrix()


    def update_channel_matrix(self) -> None:
        self.chan_gain.fill(0.0)
        for i in range(self.radio_n):
            if not self.radio_alive[i]:
                continue
            for j in range(i + 1, self.radio_n):
                if not self.radio_alive[j]:
                    continue
                g = link_path_gain(
                    self.positions[i], self.positions[j], self.buildings, self.blackout_centers,
                    grid_w=self.grid_w, cell_size_m=self.cell_size_m, blackout_r=self.blackout_r,
                    obstacle_model=self.rf_obstacle_model,
                    per_building_loss_db=self.rf_per_building_loss_db,
                )
                self.chan_gain[i, j] = g
                self.chan_gain[j, i] = g
