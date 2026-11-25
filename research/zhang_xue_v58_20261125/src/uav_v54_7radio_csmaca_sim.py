"""v5.4 Real 7-Radio/GCS Packet-Level Discrete-Event IEEE 802.11 CSMA/CA Simulator.

Implements EVERY protocol boundary required by Gate 2-3:
  - Strict role separation: ACTOR_N=6 (UAVs 0..5), RADIO_N=7 (GCS_ID=6 passive app sink, active 802.11 ACK/E2E ACK)
  - 7-state CSMA/CA FSM per radio: IDLE -> DIFS (50us) -> BACKOFF (slot=20us, CW_min=31, freeze on busy)
    -> TX_DATA (L_frame=70+8k B @ 2Mbps) -> WAIT_ACK (180us) / SIFS_ACK (10us) -> TX_MAC_ACK (30B=120us)
  - Half-duplex radio constraint + mid-packet overlapping SINR interference (SNR_th = 6 dB)
  - 3D FP32-safe segment-cylinder urban NLoS (UavNetSim a2a.py alpha=3, excess=12dB) + 2 blackout zones
  - Persistent per-radio FIFO queue (Q=64), TTL expiry (10s), max_retx=5, packet deduplication
  - Path-vector 7-bit loop-free multi-hop routing (0x7F, max 6 hops), periodic HELLO/DV neighbor discovery
  - Non-oracle sender dirty selection via E2E ACK watermarks (senders NEVER read receiver ledgers)
  - 2 x uint32 (8 B/entry) payload packing carrying exact 20-bit generation timestamps (gen_ts_s)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Topology constants
ACTOR_N: int = 6
RADIO_N: int = 7
GCS_ID: int = 6
NODE_MASK_7BIT: int = 0x7F
MAX_HOPS: int = 6
NUM_CELLS: int = 400
GRID_W: int = 20
CELL_SIZE_M: float = 50.0

# UavNetSim wire & MAC timing constants (microseconds)
LOWER_HEADERS_B: int = 58  # 20B IP + 14B MAC + 24B PHY (UavNetSim v1/v2 convention)
APP_HEADER_B: int = 12
ENTRY_WIRE_B: int = 8
MAX_ENTRIES_PER_PKT: int = 62
MAC_ACK_BYTES: int = 30
E2E_ACK_FRAME_BYTES: int = LOWER_HEADERS_B + 32  # 90 B
HELLO_FRAME_BYTES: int = LOWER_HEADERS_B + 32    # 90 B
US_PER_BYTE: int = 4  # 2 Mbps -> 4 us/byte

DIFS_US: int = 50
SIFS_US: int = 10
SLOT_US: int = 20
TX_MAC_ACK_US: int = MAC_ACK_BYTES * US_PER_BYTE  # 120 us
ACK_TIMEOUT_US: int = SIFS_US + TX_MAC_ACK_US + 50  # 180 us
CW_MIN: int = 31
MAX_RETX: int = 5
PACKET_TTL_US: int = 10_000_000  # 10 s
QUEUE_CAP_PKTS: int = 64
HELLO_INTERVAL_US: int = 500_000  # 0.5 s
ROUTE_TTL_US: int = 2_000_000     # 2.0 s
INF_TIME_US: int = 2_000_000_000

# Packet types
PKT_DATA: int = 0
PKT_E2E_ACK: int = 1
PKT_HELLO_DV: int = 2

# MAC states
ST_IDLE: int = 0
ST_DIFS: int = 1
ST_BACKOFF: int = 2
ST_TX_DATA: int = 3
ST_WAIT_ACK: int = 4
ST_SIFS_ACK: int = 5
ST_TX_MAC_ACK: int = 6

SPEED_OF_LIGHT: float = 299_792_458.0

CANONICAL_BUILDINGS_5: List[Tuple[float, float, float, float]] = [
    (280.0, 280.0, 45.0, 75.0),
    (520.0, 320.0, 45.0, 80.0),
    (480.0, 560.0, 45.0, 72.0),
    (740.0, 520.0, 45.0, 78.0),
    (320.0, 740.0, 45.0, 70.0),
]
CANONICAL_BLACKOUT_CENTERS_2: List[Tuple[int, int]] = [(15, 5), (14, 15)]
CANONICAL_BLACKOUT_R: int = 3
GCS_POS_M: Tuple[float, float, float] = (50.0, 50.0, 15.0)


def pack_entry_2x_uint32(
    origin_id: int, cell_id: int, gen_ts_s: int, n_pos: int, n_neg: int
) -> Tuple[int, int]:
    if not (0 <= origin_id <= 5):
        raise ValueError(f"Sensor origin must be 0..5, got {origin_id}")
    if not (0 <= cell_id < NUM_CELLS):
        raise ValueError(f"cell_id out of range: {cell_id}")
    if not (0 <= gen_ts_s <= 0xFFFFF):
        raise ValueError(f"gen_ts_s out of 20-bit range: {gen_ts_s}")
    if not (0 <= n_pos <= 0xFFFF and 0 <= n_neg <= 0xFFFF):
        raise ValueError(f"counts out of uint16 range: ({n_pos}, {n_neg})")
    w0 = (origin_id & 0x7) | ((cell_id & 0x1FF) << 3) | ((gen_ts_s & 0xFFFFF) << 12)
    w1 = (n_pos & 0xFFFF) | ((n_neg & 0xFFFF) << 16)
    return w0, w1


def unpack_entry_2x_uint32(w0: int, w1: int) -> Tuple[int, int, int, int, int]:
    return (
        w0 & 0x7,
        (w0 >> 3) & 0x1FF,
        (w0 >> 12) & 0xFFFFF,
        w1 & 0xFFFF,
        (w1 >> 16) & 0xFFFF,
    )


def segment_hits_cylinder_3d_fp32(
    p0: Sequence[float],
    p1: Sequence[float],
    cx: float,
    cy: float,
    radius: float,
    height: float,
    eps: float = 1e-6,
) -> bool:
    x0, y0, z0 = np.float32(p0[0]), np.float32(p0[1]), np.float32(p0[2])
    x1, y1, z1 = np.float32(p1[0]), np.float32(p1[1]), np.float32(p1[2])
    cx32, cy32 = np.float32(cx), np.float32(cy)
    r32, h32 = np.float32(radius), np.float32(height)
    if (z0 > h32 and z1 > h32) or (z0 < np.float32(0.0) and z1 < np.float32(0.0)):
        return False
    dx, dy, dz = np.float32(x1 - x0), np.float32(y1 - y0), np.float32(z1 - z0)
    l_xy = np.float32(math.hypot(float(dx), float(dy)))
    fx, fy = np.float32(cx32 - x0), np.float32(cy32 - y0)
    if l_xy <= np.float32(eps):
        if float(fx * fx + fy * fy) <= float(r32 * r32):
            return min(float(z0), float(z1)) <= float(h32) and max(float(z0), float(z1)) >= 0.0
        return False
    ux, uy = np.float32(dx / l_xy), np.float32(dy / l_xy)
    d_perp = np.float32(fx * uy - fy * ux)
    sigma = np.float32(r32 * r32 - d_perp * d_perp)
    if sigma < np.float32(0.0):
        return False
    s_c = np.float32(fx * ux + fy * uy)
    half_chord = np.float32(math.sqrt(max(0.0, float(sigma))))
    t0 = max(0.0, float((s_c - half_chord) / l_xy))
    t1 = min(1.0, float((s_c + half_chord) / l_xy))
    if t0 > t1:
        return False
    za = float(z0) + t0 * float(dz)
    zb = float(z0) + t1 * float(dz)
    return min(za, zb) <= float(h32) and max(za, zb) >= 0.0


def is_in_blackout(
    pos: Sequence[float],
    centers: Sequence[Tuple[int, int]] = CANONICAL_BLACKOUT_CENTERS_2,
    manhattan_r: int = CANONICAL_BLACKOUT_R,
) -> bool:
    cx = max(0, min(GRID_W - 1, int(pos[0] // CELL_SIZE_M)))
    cy = max(0, min(GRID_W - 1, int(pos[1] // CELL_SIZE_M)))
    return any(abs(cx - bx) + abs(cy - by) <= manhattan_r for bx, by in centers)


def link_path_gain(
    p0: Sequence[float],
    p1: Sequence[float],
    buildings: Sequence[Tuple[float, float, float, float]],
    blackout_centers: Sequence[Tuple[int, int]],
    fc_hz: float = 2.4e9,
    nlos_alpha: float = 3.0,
    nlos_excess_db: float = 12.0,
) -> float:
    """Compute linear channel gain with blackout gating & 3D urban building NLoS."""
    if is_in_blackout(p0, blackout_centers) or is_in_blackout(p1, blackout_centers):
        return 0.0
    d = max(1.0, math.dist(p0, p1))
    g_ref = (SPEED_OF_LIGHT / (4.0 * math.pi * fc_hz)) ** 2
    blocked = any(
        segment_hits_cylinder_3d_fp32(p0, p1, bx, by, br, bh)
        for bx, by, br, bh in buildings
    )
    if blocked:
        return g_ref * (d ** (-nlos_alpha)) * (10.0 ** (-nlos_excess_db / 10.0))
    return g_ref / (d * d)


@dataclass
class Packet:
    pkt_id: int
    episode_id: int
    pkt_type: int          # PKT_DATA, PKT_E2E_ACK, PKT_HELLO_DV
    app_src: int           # 0..5 for DATA, 0..6 for E2E_ACK/HELLO
    app_dst: int           # 0..6 (-1 for broadcast HELLO)
    app_seq: int
    create_us: int
    expire_us: int
    hop_cnt: int = 0
    visited_mask: int = 0
    packed_entries: Tuple[Tuple[int, int], ...] = ()  # tuple of (w0, w1) 2x uint32
    dv_table: Optional[Dict[int, Tuple[int, int]]] = None  # dst -> (hop_dist, path_mask)

    @property
    def frame_bytes(self) -> int:
        if self.pkt_type == PKT_DATA:
            return LOWER_HEADERS_B + APP_HEADER_B + ENTRY_WIRE_B * len(self.packed_entries)
        if self.pkt_type == PKT_E2E_ACK:
            return E2E_ACK_FRAME_BYTES
        return HELLO_FRAME_BYTES

    @property
    def tx_duration_us(self) -> int:
        return self.frame_bytes * US_PER_BYTE


@dataclass
class RadioState:
    mac_state: int = ST_IDLE
    state_end_us: int = INF_TIME_US
    backoff_rem_us: int = 0
    backoff_frozen: bool = False
    retx_cnt: int = 0
    active_pkt: Optional[Packet] = None
    active_next_hop: int = -1
    tx_is_ack: bool = False
    tx_pkt_id: int = -1
    tx_ack_dst: int = -1
    # Pending MAC ACK (bypasses FIFO queue per UavNetSim)
    ack_pend_dst: int = -1
    ack_pend_pkt_id: int = -1
    # RX lock & worst-case overlapping interference tracker
    rx_locked_tx: int = -1
    rx_locked_pkt: Optional[Packet] = None
    rx_locked_is_ack: bool = False
    rx_locked_pkt_id: int = -1
    rx_corrupted: bool = False
    rx_sig_w: float = 0.0
    rx_max_interf_w: float = 0.0
    # Persistent FIFO queue (Q=64)
    queue: Deque[Packet] = field(default_factory=deque)
    # HELLO is one coalesced control-plane slot, not blocked by DATA head-of-line.
    pending_hello: Optional[Packet] = None
    # Deduplication ring for DATA (prevents double E2E ACK on lost MAC ACK)
    dedup_ids: Deque[int] = field(default_factory=lambda: deque(maxlen=32))


class SevenRadioCsmaCaNetwork:
    """Real discrete-event 7-radio (6 UAV + 1 passive GCS) IEEE 802.11 CSMA/CA network."""

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
    ) -> None:
        self.tx_power_w = float(tx_power_w)
        self.noise_power_w = float(noise_power_w)
        self.snr_th_lin = 10.0 ** (snr_threshold_db / 10.0)
        self.cca_th_w = 10.0 ** ((cca_threshold_dbm - 30.0) / 10.0)
        self.queue_cap = int(queue_cap)
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

    def record_local_observation(self, actor_id: int, cell_id: int, detected: bool, t_s: int) -> None:
        if not (0 <= actor_id < ACTOR_N):
            raise ValueError("GCS cannot generate sensor observations")
        if detected:
            self.ledger_pos[actor_id, actor_id, cell_id] = min(
                65535, int(self.ledger_pos[actor_id, actor_id, cell_id]) + 1
            )
        else:
            self.ledger_neg[actor_id, actor_id, cell_id] = min(
                65535, int(self.ledger_neg[actor_id, actor_id, cell_id]) + 1
            )
        self.ledger_gen_ts_s[actor_id, actor_id, cell_id] = int(t_s)

    def count_all_dirty_entries_non_oracle(self, src: int, dst: int) -> int:
        """Total dirty cells before MAX_ENTRIES_PER_PKT (62) packetization truncation (B11 fix)."""
        if not (0 <= src < ACTOR_N) or not (0 <= dst < RADIO_N) or src == dst:
            return 0
        own_p = self.ledger_pos[src, src]
        own_n = self.ledger_neg[src, src]
        own_t = self.ledger_gen_ts_s[src, src]
        wm_p = self.ack_wm_pos[src, dst]
        wm_n = self.ack_wm_neg[src, dst]
        wm_t = self.ack_wm_ts[src, dst]
        return int(np.count_nonzero((own_p > wm_p) | (own_n > wm_n) | ((own_t > wm_t) & (own_t >= 0))))

    def get_dirty_entries_non_oracle(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """Strictly local dirty selection using sender's own ledger vs. sender's E2E ACK watermark."""
        if not (0 <= src < ACTOR_N) or not (0 <= dst < RADIO_N) or src == dst:
            return []
        own_p = self.ledger_pos[src, src]
        own_n = self.ledger_neg[src, src]
        own_t = self.ledger_gen_ts_s[src, src]
        wm_p = self.ack_wm_pos[src, dst]
        wm_n = self.ack_wm_neg[src, dst]
        wm_t = self.ack_wm_ts[src, dst]
        dirty_cells = np.flatnonzero((own_p > wm_p) | (own_n > wm_n) | ((own_t > wm_t) & (own_t >= 0)))
        entries: List[Tuple[int, int]] = []
        for cid in dirty_cells[:MAX_ENTRIES_PER_PKT]:
            entries.append(
                pack_entry_2x_uint32(
                    src, int(cid), max(0, int(own_t[cid])), int(own_p[cid]), int(own_n[cid])
                )
            )
        return entries

    def enqueue_app_data(self, src: int, dst: int, entries: Sequence[Tuple[int, int]]) -> Optional[int]:
        if not (0 <= src < ACTOR_N):
            raise ValueError("Only actors 0..5 can originate DATA packets")
        if not entries:
            return None
        r = self.radios[src]
        if len(r.queue) >= self.queue_cap:
            self.stats["drop_qfull"] += 1
            return None
        pid = self._next_pkt_id
        self._next_pkt_id += 1
        seq = self._next_app_seq
        self._next_app_seq += 1
        packed = tuple(entries[:MAX_ENTRIES_PER_PKT])
        pkt = Packet(
            pkt_id=pid,
            episode_id=self.episode_id,
            pkt_type=PKT_DATA,
            app_src=src,
            app_dst=dst,
            app_seq=seq,
            create_us=self.sim_time_us,
            expire_us=min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US),
            hop_cnt=0,
            visited_mask=(1 << src),
            packed_entries=packed,
        )
        r.queue.append(pkt)
        self.e2e_pending[(src, seq)] = (dst, packed)
        self.e2e_pending_expires_us[(src, seq)] = pkt.expire_us
        self.stats["data_enqueued"] += 1
        return pid

    def _prune_app_tracking(self) -> None:
        """Free delivered/pending app records after their bounded lifetime."""
        now = self.sim_time_us
        for key in [k for k, exp in self.e2e_pending_expires_us.items() if exp <= now]:
            self.e2e_pending_expires_us.pop(key, None)
            self.e2e_pending.pop(key, None)
        for key in [k for k, exp in self.delivered_app_expires_us.items() if exp <= now]:
            self.delivered_app_expires_us.pop(key, None)

    def _enqueue_packet_internal(self, node_id: int, pkt: Packet) -> bool:
        r = self.radios[node_id]
        if len(r.queue) >= self.queue_cap:
            self.stats["drop_qfull"] += 1
            return False
        r.queue.append(pkt)
        return True

    def _prune_expired_routes(self, node_id: int) -> None:
        now = self.sim_time_us
        rt = self.routes[node_id]
        dead = [d for d, (_, _, _, exp) in rt.items() if exp <= now]
        for d in dead:
            del rt[d]

    def _lookup_next_hop(self, node_id: int, dst: int) -> int:
        self._prune_expired_routes(node_id)
        if dst == -1:
            return -1  # Broadcast HELLO
        entry = self.routes[node_id].get(dst)
        return entry[0] if entry is not None else -1

    def _is_channel_busy_at(self, node_id: int) -> bool:
        for j in range(RADIO_N):
            if j == node_id:
                continue
            if self.radios[j].mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                p_rx = self.tx_power_w * self.chan_gain[j, node_id]
                if p_rx >= self.cca_th_w:
                    return True
        return False

    def _compute_interference_at(self, rx: int, exclude_tx: int) -> float:
        interf = 0.0
        for m in range(RADIO_N):
            if m == rx or m == exclude_tx:
                continue
            if self.radios[m].mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                interf += self.tx_power_w * self.chan_gain[m, rx]
        return interf

    def _start_tx(self, tx: int, is_ack: bool) -> None:
        r = self.radios[tx]
        # Half-duplex: if tx was receiving something, corrupt its RX immediately
        if r.rx_locked_tx >= 0:
            r.rx_corrupted = True
            self.stats["half_duplex_drops"] += 1

        if is_ack:
            dur = TX_MAC_ACK_US
            r.mac_state = ST_TX_MAC_ACK
            r.state_end_us = self.sim_time_us + dur
            r.tx_is_ack = True
            r.tx_pkt_id = r.ack_pend_pkt_id
            r.tx_ack_dst = r.ack_pend_dst
            pkt_ref = None
            pkt_id = r.ack_pend_pkt_id
        else:
            assert r.active_pkt is not None
            dur = r.active_pkt.tx_duration_us
            r.mac_state = ST_TX_DATA
            r.state_end_us = self.sim_time_us + dur
            r.tx_is_ack = False
            r.tx_pkt_id = r.active_pkt.pkt_id
            r.tx_ack_dst = -1
            pkt_ref = r.active_pkt
            pkt_id = r.active_pkt.pkt_id

        # Notify all other radios: freeze backoff / update RX lock / update mid-packet interference
        for rx in range(RADIO_N):
            if rx == tx:
                continue
            rr = self.radios[rx]
            p_rx = self.tx_power_w * self.chan_gain[tx, rx]

            # Freeze ongoing DIFS or BACKOFF if CCA busy
            if p_rx >= self.cca_th_w:
                if rr.mac_state == ST_DIFS:
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US
                elif rr.mac_state == ST_BACKOFF:
                    rr.backoff_rem_us = max(0, rr.state_end_us - self.sim_time_us)
                    rr.backoff_frozen = True
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US

            # Half-duplex check: transmitting radio cannot lock onto incoming signal
            if rr.mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                continue

            if not self.radio_alive[rx]:
                continue
            if rr.rx_locked_tx < 0:
                # B1 fix: require p_rx >= cca_th_w so signals 26,000x below noise floor cannot lock RX
                if p_rx >= self.cca_th_w:
                    rr.rx_locked_tx = tx
                    rr.rx_locked_pkt = pkt_ref
                    rr.rx_locked_is_ack = is_ack
                    rr.rx_locked_pkt_id = pkt_id
                    rr.rx_corrupted = False
                    rr.rx_sig_w = p_rx
                    rr.rx_max_interf_w = self._compute_interference_at(rx, exclude_tx=tx)
            else:
                # Mid-packet interferer! Update worst-case interference on existing RX lock
                interf = self._compute_interference_at(rx, exclude_tx=rr.rx_locked_tx)
                if interf > rr.rx_max_interf_w:
                    rr.rx_max_interf_w = interf

    def _finish_tx(self, tx: int) -> None:
        r = self.radios[tx]
        was_ack = r.tx_is_ack
        pkt = r.active_pkt
        ack_dst = r.tx_ack_dst
        ack_pkt_id = r.tx_pkt_id

        # Charge TX airtime to actual transmitting radio (GCS airtime tracked separately, never billed to UAVs)
        dur = TX_MAC_ACK_US if was_ack else (pkt.tx_duration_us if pkt is not None else 0)
        self.step_tx_airtime_us[tx] += dur

        # Deliver to any radio locked onto tx whose SINR >= threshold
        for rx in range(RADIO_N):
            if rx == tx:
                continue
            rr = self.radios[rx]
            if rr.rx_locked_tx != tx:
                continue
            # Evaluate SINR with worst-case overlapping interference
            sinr = rr.rx_sig_w / (self.noise_power_w + rr.rx_max_interf_w)
            ok = (not rr.rx_corrupted) and (sinr >= self.snr_th_lin) and (self.chan_gain[tx, rx] > 0.0)
            if was_ack and self.force_drop_next_mac_ack_from == tx:
                ok = False
                self.force_drop_next_mac_ack_from = None
            # Clear RX lock
            rr.rx_locked_tx = -1
            rr.rx_locked_pkt = None
            rr.rx_corrupted = False

            if not ok:
                self.stats["sinr_drops"] += 1
                continue

            if was_ack:
                if rx == ack_dst and rr.mac_state == ST_WAIT_ACK and rr.tx_pkt_id == ack_pkt_id:
                    # Successful MAC ACK receipt!
                    self.stats["mac_ack_ok"] += 1
                    rr.active_pkt = None
                    rr.active_next_hop = -1
                    rr.retx_cnt = 0
                    rr.backoff_frozen = False
                    rr.backoff_rem_us = 0
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US
            else:
                assert pkt is not None
                self._handle_rx_frame(rx=rx, prev_hop=tx, pkt=pkt)

        # Transition transmitter state
        if was_ack:
            r.ack_pend_dst = -1
            r.ack_pend_pkt_id = -1
            r.tx_is_ack = False
            r.mac_state = ST_IDLE
            r.state_end_us = INF_TIME_US
        else:
            assert pkt is not None
            if pkt.pkt_type == PKT_HELLO_DV:
                # Broadcast HELLO needs no MAC ACK
                r.active_pkt = None
                r.active_next_hop = -1
                r.retx_cnt = 0
                r.backoff_frozen = False
                r.mac_state = ST_IDLE
                r.state_end_us = INF_TIME_US
            else:
                r.mac_state = ST_WAIT_ACK
                r.state_end_us = self.sim_time_us + ACK_TIMEOUT_US

    def _handle_rx_frame(self, rx: int, prev_hop: int, pkt: Packet) -> None:
        if pkt.episode_id != self.episode_id:
            return  # Reject stale packet from prior episode

        if pkt.pkt_type == PKT_HELLO_DV:
            # Update 1-hop neighbor & path-vector DV routes (reject loops via 7-bit mask)
            exp = min(INF_TIME_US, self.sim_time_us + ROUTE_TTL_US)
            self.routes[rx][prev_hop] = (prev_hop, 1, (1 << rx) | (1 << prev_hop), exp)
            if pkt.dv_table:
                for dst, (hops, pmask) in pkt.dv_table.items():
                    if dst == rx or (pmask & (1 << rx)) != 0:
                        continue  # Path-vector loop prevention!
                    cand_hops = hops + 1
                    if cand_hops > MAX_HOPS:
                        continue
                    cur = self.routes[rx].get(dst)
                    if cur is None or cur[3] <= self.sim_time_us or cand_hops < cur[1] or cur[0] == prev_hop:
                        self.routes[rx][dst] = (prev_hop, cand_hops, (pmask | (1 << rx)) & NODE_MASK_7BIT, exp)
            return

        # Unicast DATA or E2E_ACK: only intended next hop processes it
        if self.radios[prev_hop].active_next_hop != rx:
            return

        # Schedule immediate SIFS -> TX_MAC_ACK (bypasses FIFO queue)
        rr = self.radios[rx]
        if rr.mac_state in (ST_IDLE, ST_DIFS, ST_BACKOFF):
            if rr.mac_state == ST_BACKOFF:
                rr.backoff_rem_us = max(0, rr.state_end_us - self.sim_time_us)
                rr.backoff_frozen = True
            rr.ack_pend_dst = prev_hop
            rr.ack_pend_pkt_id = pkt.pkt_id
            rr.mac_state = ST_SIFS_ACK
            rr.state_end_us = self.sim_time_us + SIFS_US

        # Deduplicate packet ID (handles DATA received + MAC ACK lost -> retry)
        if pkt.pkt_id in rr.dedup_ids:
            self.stats["dedup_hits"] += 1
            return
        rr.dedup_ids.append(pkt.pkt_id)

        # Check if rx is final application destination
        if pkt.app_dst == rx:
            if pkt.pkt_type == PKT_DATA:
                app_key = (rx, pkt.app_src, pkt.app_seq)
                already_delivered = app_key in self.delivered_app_expires_us
                # Retry at a relay gets a different hop-local pkt_id: dedup using app identity.
                if not already_delivered:
                    self.delivered_app_expires_us[app_key] = min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US)
                    self.stats["data_delivered_dst"] += 1
                    if rx == GCS_ID:
                        self.stats["data_delivered_gcs"] += 1
                    # Merge packed 2x uint32 entries into receiver's provenance ledger.
                    for w0, w1 in pkt.packed_entries:
                        origin, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1)
                        new_info = (n_p > self.ledger_pos[rx, origin, cid]
                                    or n_n > self.ledger_neg[rx, origin, cid])
                        if n_p > self.ledger_pos[rx, origin, cid]:
                            self.ledger_pos[rx, origin, cid] = n_p
                        if n_n > self.ledger_neg[rx, origin, cid]:
                            self.ledger_neg[rx, origin, cid] = n_n
                        if gen_ts > self.ledger_gen_ts_s[rx, origin, cid]:
                            self.ledger_gen_ts_s[rx, origin, cid] = gen_ts
                            # Delivery-time-only discount, no artificial refresh on old retries.
                            self.ledger_rx_age_s[rx, origin, cid] = max(
                                0., (self.sim_time_us - self.mission_epoch_us) / 1e6 - gen_ts)
                        if rx != GCS_ID and new_info:
                            self.step_new_peer_cells += 1
                            # Approximate marginal uncertainty reduction weight: diminishes rapidly with existing counts
                            prev_total = int(self.ledger_pos[rx, :, cid].sum() + self.ledger_neg[rx, :, cid].sum())
                            self.step_peer_info_gain += 1.0 / float(max(1, prev_total))
                        if rx == GCS_ID and (
                            gen_ts > self.gcs_latest_u_s[cid]
                            or (gen_ts == self.gcs_latest_u_s[cid] and new_info)
                        ):
                            # Stale retries cannot refresh receiver AoI; same-second novel evidence updates rx time (B12).
                            self.gcs_latest_u_s[cid] = gen_ts
                            self.gcs_last_rx_us[cid] = self.sim_time_us
                else:
                    self.stats["dedup_hits"] += 1
                # E2E ACK re-sent even on duplicates: original ACK may have been lost.
                ack_pid = self._next_pkt_id
                self._next_pkt_id += 1
                e2e_ack = Packet(
                    pkt_id=ack_pid,
                    episode_id=self.episode_id,
                    pkt_type=PKT_E2E_ACK,
                    app_src=rx,
                    app_dst=pkt.app_src,
                    app_seq=pkt.app_seq,
                    create_us=self.sim_time_us,
                    expire_us=min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US),
                    hop_cnt=0,
                    visited_mask=(1 << rx),
                )
                self._enqueue_packet_internal(rx, e2e_ack)
            elif pkt.pkt_type == PKT_E2E_ACK:
                # Advance sender's non-oracle E2E ACK watermark!
                self.stats["e2e_ack_rx"] += 1
                pend = self.e2e_pending.pop((rx, pkt.app_seq), None)
                self.e2e_pending_expires_us.pop((rx, pkt.app_seq), None)
                if pend is not None:
                    dst_acked, entries = pend
                    for w0, w1 in entries:
                        _, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1)
                        if n_p > self.ack_wm_pos[rx, dst_acked, cid]:
                            self.ack_wm_pos[rx, dst_acked, cid] = n_p
                        if n_n > self.ack_wm_neg[rx, dst_acked, cid]:
                            self.ack_wm_neg[rx, dst_acked, cid] = n_n
                        if gen_ts > self.ack_wm_ts[rx, dst_acked, cid]:
                            self.ack_wm_ts[rx, dst_acked, cid] = gen_ts
            return

        # Intermediate relay: DO NOT merge into relay's own belief; check loop & hop budget
        if pkt.hop_cnt + 1 >= MAX_HOPS or (pkt.visited_mask & (1 << rx)) != 0:
            self.stats["drop_loop"] += 1
            return
        if self.sim_time_us >= pkt.expire_us:
            self.stats["drop_ttl"] += 1
            return
        fwd = Packet(
            pkt_id=self._next_pkt_id,
            episode_id=pkt.episode_id,
            pkt_type=pkt.pkt_type,
            app_src=pkt.app_src,
            app_dst=pkt.app_dst,
            app_seq=pkt.app_seq,
            create_us=pkt.create_us,
            expire_us=pkt.expire_us,
            hop_cnt=pkt.hop_cnt + 1,
            visited_mask=(pkt.visited_mask | (1 << rx)) & NODE_MASK_7BIT,
            packed_entries=pkt.packed_entries,
        )
        self._next_pkt_id += 1
        self._enqueue_packet_internal(rx, fwd)

    def _try_arm_idle_radios(self) -> None:
        for r_id in range(RADIO_N):
            if not self.radio_alive[r_id]:
                continue
            r = self.radios[r_id]
            if r.mac_state != ST_IDLE:
                continue
            # Pending MAC ACK has strict priority
            if r.ack_pend_dst >= 0:
                r.mac_state = ST_SIFS_ACK
                r.state_end_us = self.sim_time_us + SIFS_US
                continue
            # Control plane must still be able to discover routes when DATA has no route.
            if r.active_pkt is None and r.pending_hello is not None:
                candidate = r.pending_hello
                r.pending_hello = None
                if self.sim_time_us < candidate.expire_us:
                    r.active_pkt = candidate
                    r.active_next_hop = -1
                    r.retx_cnt = 0
                    r.backoff_frozen = False
                    r.backoff_rem_us = 0
            # B3 fix: Drop expired packets and avoid HOL blocking of E2E_ACK or routable DATA
            if r.active_pkt is None and r.queue:
                # Prune expired packets across queue
                kept = deque()
                while r.queue:
                    p = r.queue.popleft()
                    if self.sim_time_us >= p.expire_us:
                        self.stats["drop_ttl"] += 1
                    else:
                        kept.append(p)
                r.queue = kept
                # Select first routable packet, prioritizing E2E_ACK over DATA
                chosen_idx = -1
                chosen_nh = -1
                for want_ack in (True, False):
                    for idx, cand in enumerate(r.queue):
                        is_e2e = (cand.pkt_type == PKT_E2E_ACK)
                        if want_ack and not is_e2e:
                            continue
                        if (not want_ack) and is_e2e:
                            continue
                        if cand.pkt_type == PKT_HELLO_DV:
                            chosen_idx = idx
                            chosen_nh = -1
                            break
                        nh = self._lookup_next_hop(r_id, cand.app_dst)
                        if nh >= 0:
                            chosen_idx = idx
                            chosen_nh = nh
                            break
                    if chosen_idx >= 0:
                        break
                if chosen_idx >= 0:
                    del_list = list(r.queue)
                    r.active_pkt = del_list.pop(chosen_idx)
                    r.queue = deque(del_list)
                    r.active_next_hop = chosen_nh
                    r.retx_cnt = 0
                    r.backoff_frozen = False
                    r.backoff_rem_us = 0
            if r.active_pkt is not None:
                if self.sim_time_us >= r.active_pkt.expire_us:
                    r.active_pkt = None
                    r.active_next_hop = -1
                    self.stats["drop_ttl"] += 1
                    continue
                if not self._is_channel_busy_at(r_id):
                    r.mac_state = ST_DIFS
                    r.state_end_us = self.sim_time_us + DIFS_US

    def _enqueue_hello_dv(self, r_id: int) -> None:
        self._prune_expired_routes(r_id)
        dv = {dst: (hops, pmask) for dst, (_, hops, pmask, _) in self.routes[r_id].items()}
        pid = self._next_pkt_id
        self._next_pkt_id += 1
        pkt = Packet(
            pkt_id=pid,
            episode_id=self.episode_id,
            pkt_type=PKT_HELLO_DV,
            app_src=r_id,
            app_dst=-1,
            app_seq=0,
            create_us=self.sim_time_us,
            expire_us=min(INF_TIME_US, self.sim_time_us + HELLO_INTERVAL_US),
            dv_table=dv,
        )
        # Periodic HELLO/DV replaces any older unsent HELLO rather than consuming Q.
        self.radios[r_id].pending_hello = pkt

    def run_until_us(self, target_us: int, max_events: int = 2000) -> bool:
        """Advance discrete-event CSMA/CA network to target_us. Returns True if completed cleanly."""
        if target_us < self.sim_time_us:
            raise ValueError(f"Cannot run network clock backward: {target_us} < {self.sim_time_us}")
        self._prune_app_tracking()
        for _ in range(max_events):
            self._try_arm_idle_radios()
            t_next = target_us
            for r_id in range(RADIO_N):
                if not self.radio_alive[r_id]:
                    continue
                if self.radios[r_id].state_end_us < t_next:
                    t_next = self.radios[r_id].state_end_us
                if self.hello_next_us[r_id] < t_next:
                    t_next = self.hello_next_us[r_id]
            if t_next >= target_us:
                self.sim_time_us = target_us
                self._prune_app_tracking()
                return True

            self.sim_time_us = t_next
            self.stats["events_processed"] += 1

            # 1. Process periodic HELLO/DV timers (alive radios only)
            for r_id in range(RADIO_N):
                if self.radio_alive[r_id] and self.hello_next_us[r_id] == t_next:
                    self._enqueue_hello_dv(r_id)
                    self.hello_next_us[r_id] = min(INF_TIME_US, t_next + HELLO_INTERVAL_US)

            # 2a. B13 fix: Finish all due TX/TX_ACK first (matches Warp two-phase ordering)
            for r_id in range(RADIO_N):
                if not self.radio_alive[r_id]:
                    continue
                r = self.radios[r_id]
                if r.state_end_us == t_next and r.mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                    self._finish_tx(r_id)

            # 2b. Process remaining due radio state expirations in deterministic ID order
            for r_id in range(RADIO_N):
                if not self.radio_alive[r_id]:
                    continue
                r = self.radios[r_id]
                if r.state_end_us != t_next:
                    continue
                st = r.mac_state
                if st == ST_DIFS:
                    if r.backoff_frozen:
                        rem = r.backoff_rem_us
                        r.backoff_frozen = False
                    else:
                        cw = (CW_MIN + 1) * (1 << r.retx_cnt) - 1
                        rem = int(self.rng.integers(0, cw + 1)) * SLOT_US
                    if rem == 0:
                        self._start_tx(r_id, is_ack=False)
                    else:
                        r.mac_state = ST_BACKOFF
                        r.state_end_us = self.sim_time_us + rem
                elif st == ST_BACKOFF:
                    r.backoff_rem_us = 0
                    self._start_tx(r_id, is_ack=False)
                elif st == ST_SIFS_ACK:
                    self._start_tx(r_id, is_ack=True)
                elif st == ST_WAIT_ACK:
                    self.stats["mac_ack_timeout"] += 1
                    r.retx_cnt += 1
                    if r.retx_cnt >= MAX_RETX or (r.active_pkt is not None and self.sim_time_us >= r.active_pkt.expire_us):
                        self.stats["drop_retx"] += 1
                        r.active_pkt = None
                        r.active_next_hop = -1
                        r.retx_cnt = 0
                        r.backoff_frozen = False
                        r.backoff_rem_us = 0
                        r.mac_state = ST_IDLE
                        r.state_end_us = INF_TIME_US
                    else:
                        # B9 fix: re-lookup next hop on MAC retry in case neighbor moved out of range
                        if r.active_pkt is not None and r.active_pkt.pkt_type != PKT_HELLO_DV:
                            nh = self._lookup_next_hop(r_id, r.active_pkt.app_dst)
                            if nh >= 0:
                                r.active_next_hop = nh
                        r.backoff_frozen = False
                        r.backoff_rem_us = 0
                        r.mac_state = ST_IDLE
                        r.state_end_us = INF_TIME_US
        return False
