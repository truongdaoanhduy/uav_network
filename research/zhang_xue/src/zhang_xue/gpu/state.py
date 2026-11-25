"""v5.7 Executable Seven-Radio Warp CSMA/CA Simulator (6 UAVs + 1 GCS).

Upgrades over v5.6:
  - Removed dead duplicate kernels (B14) so only modular reset/inject/event kernels compile.
  - Fixed weak-signal RX lock: requires received power >= cca_th_w (B1).
  - Fixed multi-interferer SINR: computes TRUE SUM of all concurrent transmitters on lock and mid-packet (B2).
  - Fixed immutable HELLO DV snapshot (`hello_dv_*` captured at HELLO enqueue, read at RX) for CPU-Warp parity.
  - Fixed duplicate DATA handling: re-sends E2E_ACK on duplicate DATA, separate dedup for DATA vs E2E_ACK (B5).
  - Fixed `stats_peer` to count novel peer cells (not packets) and added `ledger_rx_age_s`, `gcs_last_rx_us`,
    `mission_epoch_us`, and `radio_alive` (B4, B6).
  - Fixed `stats_drop` race via `wp.atomic_add` (B7), fixed LCG low-bit bias via upper-bit shift,
    fixed `rx_corrupt`/`frozen`/`backoff_rem` resets, added early `break` on `k >= entries`, and clamped timestamps.
"""
from __future__ import annotations

import math
import numpy as np

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

N_RADIO = 7
N_UAV = 6
GCS = 6
Q = 64
K = 62
CELLS = 400
DATA = 0
E2E_ACK = 1
HELLO = 2
IDLE = 0
DIFS = 1
BACKOFF = 2
TX = 3
WAIT_ACK = 4
SIFS = 5
TX_ACK = 6
NEVER = 2_000_000_000

if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

    @wp.struct
    class NetworkBuffers:
        now: wp.array(dtype=wp.int32)
        mission_epoch_us: wp.array(dtype=wp.int32)
        episode: wp.array(dtype=wp.int32)
        overflow: wp.array(dtype=wp.int32)
        event_count: wp.array(dtype=wp.int32)
        next_pid: wp.array(dtype=wp.int32)
        next_seq: wp.array(dtype=wp.int32)
        rng: wp.array2d(dtype=wp.uint32)
        radio_alive: wp.array2d(dtype=wp.uint8)
        gain: wp.array3d(dtype=wp.float32)
        state: wp.array2d(dtype=wp.int32)
        deadline: wp.array2d(dtype=wp.int32)
        hello_due: wp.array2d(dtype=wp.int32)
        hello_pending: wp.array2d(dtype=wp.int32)
        hello_dv_hop: wp.array3d(dtype=wp.int32)
        hello_dv_dist: wp.array3d(dtype=wp.int32)
        hello_dv_mask: wp.array3d(dtype=wp.int32)
        active_slot: wp.array2d(dtype=wp.int32)
        next_hop: wp.array2d(dtype=wp.int32)
        tx_pkt_id: wp.array2d(dtype=wp.int32)
        ack_dst: wp.array2d(dtype=wp.int32)
        ack_pid: wp.array2d(dtype=wp.int32)
        retries: wp.array2d(dtype=wp.int32)
        backoff_rem: wp.array2d(dtype=wp.int32)
        frozen: wp.array2d(dtype=wp.int32)
        rx_lock: wp.array2d(dtype=wp.int32)
        rx_corrupt: wp.array2d(dtype=wp.int32)
        rx_interf: wp.array2d(dtype=wp.float32)
        rx_sig: wp.array2d(dtype=wp.float32)
        tx_airtime: wp.array2d(dtype=wp.int32)
        q_head: wp.array2d(dtype=wp.int32)
        q_count: wp.array2d(dtype=wp.int32)
        q_type: wp.array3d(dtype=wp.int32)
        q_pid: wp.array3d(dtype=wp.int32)
        q_src: wp.array3d(dtype=wp.int32)
        q_dst: wp.array3d(dtype=wp.int32)
        q_seq: wp.array3d(dtype=wp.int32)
        q_expiry: wp.array3d(dtype=wp.int32)
        q_visited: wp.array3d(dtype=wp.int32)
        q_hops: wp.array3d(dtype=wp.int32)
        q_entries: wp.array3d(dtype=wp.int32)
        q_word0: wp.array4d(dtype=wp.uint32)
        q_word1: wp.array4d(dtype=wp.uint32)
        route_hop: wp.array3d(dtype=wp.int32)
        route_dist: wp.array3d(dtype=wp.int32)
        route_mask: wp.array3d(dtype=wp.int32)
        route_expiry: wp.array3d(dtype=wp.int32)
        # Separate bounded dedup rings for DATA and E2E_ACK (prevents key collision)
        delivered_seq: wp.array4d(dtype=wp.int32)
        delivered_ptr: wp.array3d(dtype=wp.int32)
        delivered_ack_seq: wp.array4d(dtype=wp.int32)
        delivered_ack_ptr: wp.array3d(dtype=wp.int32)
        ledger_pos: wp.array4d(dtype=wp.uint16)
        ledger_neg: wp.array4d(dtype=wp.uint16)
        ledger_gen: wp.array4d(dtype=wp.int32)
        ledger_rx_age_s: wp.array4d(dtype=wp.float32)
        ack_pos: wp.array4d(dtype=wp.uint16)
        ack_neg: wp.array4d(dtype=wp.uint16)
        ack_gen: wp.array4d(dtype=wp.int32)
        pending_seq: wp.array3d(dtype=wp.int32)
        pending_dst: wp.array3d(dtype=wp.int32)
        pending_valid: wp.array3d(dtype=wp.int32)
        pending_expiry: wp.array3d(dtype=wp.int32)
        pending_entries: wp.array3d(dtype=wp.int32)
        pending_word0: wp.array4d(dtype=wp.uint32)
        pending_word1: wp.array4d(dtype=wp.uint32)
        gcs_latest: wp.array2d(dtype=wp.int32)
        gcs_last_rx_us: wp.array2d(dtype=wp.int32)
        stats_gcs: wp.array(dtype=wp.int32)
        stats_peer: wp.array(dtype=wp.int32)
        stats_mac_ack: wp.array(dtype=wp.int32)
        stats_e2e: wp.array(dtype=wp.int32)
        stats_retry: wp.array(dtype=wp.int32)
        stats_drop: wp.array(dtype=wp.int32)
        stats_ttl: wp.array(dtype=wp.int32)
        stats_loop: wp.array(dtype=wp.int32)
