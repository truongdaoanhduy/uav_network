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
    ) -> None:
        self.tx_power_w = float(tx_power_w)
        self.noise_power_w = float(noise_power_w)
        self.snr_th_lin = 10.0 ** (snr_threshold_db / 10.0)
        self.cca_th_w = 10.0 ** ((cca_threshold_dbm - 30.0) / 10.0)
        self.queue_cap = int(queue_cap)
        self.protocol = protocol if protocol is not None else CsmaProtocolConfig()
        if not isinstance(self.protocol, CsmaProtocolConfig):
            raise TypeError("protocol must be CsmaProtocolConfig")
        self.buildings = list(buildings)
        self.blackout_centers = list(blackout_centers)
        self.rng = np.random.default_rng(seed)

        self.episode_id: int = 0
        self.sim_time_us: int = 0
        self._next_pkt_id: int = 1
        self._next_app_seq: int = 1

        self.positions = np.zeros((RADIO_N, 3), dtype=np.float64)
        self.positions[GCS_ID] = np.array(GCS_POS_M, dtype=np.float64)
        self.chan_gain = np.zeros((RADIO_N, RADIO_N), dtype=np.float64)
        self.radio_alive = np.ones(RADIO_N, dtype=bool)

        self.radios: List[RadioState] = [RadioState() for _ in range(RADIO_N)]
        # Distributed path-vector routing table: routes[r][dst] = (next_hop, hop_dist, path_mask, expire_us)
        self.routes: List[Dict[int, Tuple[int, int, int, int]]] = [{} for _ in range(RADIO_N)]
        self.hello_next_us: List[int] = [int(self.rng.integers(0, 50_000)) for _ in range(RADIO_N)]

        # Provenance ledgers: [RADIO_N, ACTOR_N, NUM_CELLS, 2] and gen_ts_s [RADIO_N, ACTOR_N, NUM_CELLS]
        self.ledger_pos = np.zeros((RADIO_N, ACTOR_N, NUM_CELLS), dtype=np.uint16)
        self.ledger_neg = np.zeros((RADIO_N, ACTOR_N, NUM_CELLS), dtype=np.uint16)
        self.ledger_gen_ts_s = np.full((RADIO_N, ACTOR_N, NUM_CELLS), -1, dtype=np.int32)
        # Frozen delivery-age of the latest cumulative snapshot for Xue-style DS discount.
        # Locally generated evidence has zero reception delay; relayed evidence is delayed.
        self.ledger_rx_age_s = np.zeros((RADIO_N, ACTOR_N, NUM_CELLS), dtype=np.float32)
        self.mission_epoch_us = 0
        # Actual novel information delivered to a peer this step, never raw packet count.
        self.step_new_peer_cells = 0
        self.step_peer_info_gain = 0.0

        # Non-oracle sender E2E ACK watermarks: [ACTOR_N, RADIO_N, NUM_CELLS]
        self.ack_wm_pos = np.zeros((ACTOR_N, RADIO_N, NUM_CELLS), dtype=np.uint16)
        self.ack_wm_neg = np.zeros((ACTOR_N, RADIO_N, NUM_CELLS), dtype=np.uint16)
        self.ack_wm_ts = np.full((ACTOR_N, RADIO_N, NUM_CELLS), -1, dtype=np.int32)
        # Sender pending E2E table: (src, app_seq) -> (dst, packed_entries)
        self.e2e_pending: Dict[Tuple[int, int], Tuple[int, Tuple[Tuple[int, int], ...]]] = {}
        # Independent time-bounded metadata: pending E2E records must not grow without limit.
        self.e2e_pending_expires_us: Dict[Tuple[int, int], int] = {}
        # Application-level dedup survives MAC forwarding with a NEW hop-local pkt_id.
        self.delivered_app_expires_us: Dict[Tuple[int, int, int], int] = {}

        # GCS receiver-side AoI timestamps (microsecond event receipt & generation time)
        self.gcs_latest_u_s = np.full(NUM_CELLS, -1, dtype=np.int32)
        self.gcs_last_rx_us = np.full(NUM_CELLS, -1, dtype=np.int64)

        # Per-radio TX airtime (us) — resets each env step by caller
        self.step_tx_airtime_us = np.zeros(RADIO_N, dtype=np.int64)

        # Telemetry & drop counters
        self.stats: Dict[str, int] = {
            "data_enqueued": 0,
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
        self.radios = [RadioState() for _ in range(RADIO_N)]
        self.routes = [{} for _ in range(RADIO_N)]
        self.hello_next_us = [int(self.rng.integers(0, 20_000)) for _ in range(RADIO_N)]
        self.ledger_pos.fill(0)
        self.ledger_neg.fill(0)
        self.ledger_gen_ts_s.fill(-1)
        self.ledger_rx_age_s.fill(0.0)
        self.mission_epoch_us = 0
        self.step_new_peer_cells = 0
        self.step_peer_info_gain = 0.0
        self.radio_alive.fill(True)
        self.ack_wm_pos.fill(0)
        self.ack_wm_neg.fill(0)
        self.ack_wm_ts.fill(-1)
        self.e2e_pending.clear()
        self.e2e_pending_expires_us.clear()
        self.delivered_app_expires_us.clear()
        self.gcs_latest_u_s.fill(-1)
        self.gcs_last_rx_us.fill(-1)
        self.step_tx_airtime_us.fill(0)
        for k in self.stats:
            self.stats[k] = 0
        self.update_channel_matrix()


    def update_positions(self, positions_7x3: np.ndarray) -> None:
        self.positions = np.asarray(positions_7x3, dtype=np.float64).copy()
        self.update_channel_matrix()


    def set_radio_alive(self, radio_id: int, alive: bool) -> None:
        if not (0 <= radio_id < RADIO_N):
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
        for i in range(RADIO_N):
            if not self.radio_alive[i]:
                continue
            for j in range(i + 1, RADIO_N):
                if not self.radio_alive[j]:
                    continue
                g = link_path_gain(
                    self.positions[i], self.positions[j], self.buildings, self.blackout_centers
                )
                self.chan_gain[i, j] = g
                self.chan_gain[j, i] = g
