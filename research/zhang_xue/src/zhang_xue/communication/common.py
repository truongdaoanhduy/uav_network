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


@dataclass(frozen=True)
class CsmaProtocolConfig:
    """CPU-reference event timing; Warp kernels still use fixed protocol constants."""
    difs_us: int = DIFS_US
    sifs_us: int = SIFS_US
    slot_us: int = SLOT_US
    cw_min: int = CW_MIN
    max_retries: int = MAX_RETX
    ack_timeout_us: int = ACK_TIMEOUT_US

    def __post_init__(self):
        numeric = (self.difs_us, self.sifs_us, self.slot_us, self.cw_min,
                   self.max_retries, self.ack_timeout_us)
        if any(type(v) is not int or v < 1 for v in numeric):
            raise ValueError("CSMA timing/counts must be positive integers")
        if self.cw_min > 1023 or self.max_retries > 12 or self.slot_us > 10_000:
            raise ValueError("CSMA contention configuration out of supported range")
        if self.ack_timeout_us <= self.sifs_us + TX_MAC_ACK_US:
            raise ValueError("ACK timeout must exceed SIFS plus MAC-ACK airtime")
