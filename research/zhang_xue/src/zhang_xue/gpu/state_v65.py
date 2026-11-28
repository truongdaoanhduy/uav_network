"""v6.5 Parameterized Compressed Warp SoA Buffers (Dual-Path, Non-Breaking).

Keeps legacy `gpu/state.py` (`v61_uncompressed`) untouched for backward compatibility,
while providing an *allocation-only* compressed schema (`make_shapes_v63`:
379,848 B/env at 6 UAV / 20x20). This omits data-mule multi-origin ACK
watermarks and cannot execute the packet network. Selected UavNetSim PHY parameters
(0.1 W, CCA=-82 dBm, Noise=4.39e-13 W) and scalable (num_uavs, grid_w).
"""
from __future__ import annotations

import math

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

# Exact UavNetSim-v1 IEEE 802.11b PHY defaults (utils/config.py)
UAVNETSIM_TX_POWER_W: float = 0.1
UAVNETSIM_CCA_DBM: float = -82.0
UAVNETSIM_CCA_W: float = 10.0 ** ((UAVNETSIM_CCA_DBM - 30.0) / 10.0)  # 6.3095734448e-12 W
UAVNETSIM_NOISE_W: float = 1e-3 * 10.0 ** ((-174.0 + 10.0 * math.log10(22.0e6) + 7.0) / 10.0)  # 4.389577e-13 W
UAVNETSIM_SNR_TH_LIN: float = 10.0 ** (6.0 / 10.0)  # 3.9810717

Q_V65: int = 32
K_V65: int = 62
NEVER_US: int = 2_000_000_000

if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

    @wp.struct
    class SimDimsV65:
        n_uav: wp.int32
        n_radio: wp.int32
        gcs_id: wp.int32
        grid_w: wp.int32
        num_cells: wp.int32
        q_cap: wp.int32
        p_cap: wp.int32
        k_entries: wp.int32
        scalable_wire: wp.int32
        node_mask_all: wp.uint32

    @wp.struct
    class NetworkBuffersV65:
        now: wp.array(dtype=wp.int32)
        mission_epoch_us: wp.array(dtype=wp.int32)
        episode: wp.array(dtype=wp.int32)
        overflow: wp.array(dtype=wp.uint8)
        event_count: wp.array(dtype=wp.int32)
        next_pid: wp.array(dtype=wp.int32)
        next_seq: wp.array(dtype=wp.int32)
        rng: wp.array2d(dtype=wp.uint32)
        radio_alive: wp.array2d(dtype=wp.uint8)
        gain: wp.array3d(dtype=wp.float32)
        state: wp.array2d(dtype=wp.int8)
        deadline: wp.array2d(dtype=wp.int32)
        hello_due: wp.array2d(dtype=wp.int32)
        hello_pending: wp.array2d(dtype=wp.uint8)
        hello_dv_hop: wp.array3d(dtype=wp.int8)
        hello_dv_dist: wp.array3d(dtype=wp.int8)
        hello_dv_mask: wp.array3d(dtype=wp.uint32)
        active_slot: wp.array2d(dtype=wp.int8)
        next_hop: wp.array2d(dtype=wp.int8)
        tx_pkt_id: wp.array2d(dtype=wp.int32)
        ack_dst: wp.array2d(dtype=wp.int8)
        ack_pid: wp.array2d(dtype=wp.int32)
        retries: wp.array2d(dtype=wp.uint8)
        backoff_rem: wp.array2d(dtype=wp.int32)
        frozen: wp.array2d(dtype=wp.uint8)
        rx_lock: wp.array2d(dtype=wp.int8)
        rx_corrupt: wp.array2d(dtype=wp.uint8)
        rx_interf: wp.array2d(dtype=wp.float32)
        rx_sig: wp.array2d(dtype=wp.float32)
        tx_airtime: wp.array2d(dtype=wp.int32)
        q_order: wp.array3d(dtype=wp.uint8)
        q_used: wp.array3d(dtype=wp.uint8)
        q_count: wp.array2d(dtype=wp.uint8)
        q_type: wp.array3d(dtype=wp.uint8)
        q_pid: wp.array3d(dtype=wp.int32)
        q_src: wp.array3d(dtype=wp.int8)
        q_dst: wp.array3d(dtype=wp.int8)
        q_seq: wp.array3d(dtype=wp.int32)
        q_expiry: wp.array3d(dtype=wp.int32)
        q_visited: wp.array3d(dtype=wp.uint32)
        q_hops: wp.array3d(dtype=wp.uint8)
        q_entries: wp.array3d(dtype=wp.uint8)
        q_word0: wp.array4d(dtype=wp.uint32)
        q_word1: wp.array4d(dtype=wp.uint32)
        route_hop: wp.array3d(dtype=wp.int8)
        route_dist: wp.array3d(dtype=wp.int8)
        route_mask: wp.array3d(dtype=wp.uint32)
        route_expiry: wp.array3d(dtype=wp.int32)
        delivered_seq: wp.array4d(dtype=wp.int32)
        delivered_ptr: wp.array3d(dtype=wp.uint8)
        delivered_ack_seq: wp.array4d(dtype=wp.int32)
        delivered_ack_ptr: wp.array3d(dtype=wp.uint8)
        ledger_pos: wp.array4d(dtype=wp.uint16)
        ledger_neg: wp.array4d(dtype=wp.uint16)
        ledger_gen: wp.array4d(dtype=wp.int16)
        ledger_rx_age_s: wp.array4d(dtype=wp.float16)
        ack_pos: wp.array4d(dtype=wp.uint16)
        ack_neg: wp.array4d(dtype=wp.uint16)
        ack_gen: wp.array4d(dtype=wp.int16)
        pending_seq: wp.array3d(dtype=wp.int32)
        pending_dst: wp.array3d(dtype=wp.int8)
        pending_valid: wp.array3d(dtype=wp.uint8)
        pending_expiry: wp.array3d(dtype=wp.int32)
        pending_entries: wp.array3d(dtype=wp.uint8)
        pending_word0: wp.array4d(dtype=wp.uint32)
        pending_word1: wp.array4d(dtype=wp.uint32)
        gcs_latest: wp.array2d(dtype=wp.int16)
        gcs_last_rx_us: wp.array2d(dtype=wp.int32)
        step_peer_info_gain: wp.array(dtype=wp.float32)
        stats_gcs: wp.array(dtype=wp.int32)
        stats_peer: wp.array(dtype=wp.int32)
        stats_mac_ack: wp.array(dtype=wp.int32)
        stats_e2e: wp.array(dtype=wp.int32)
        stats_retry: wp.array(dtype=wp.int32)
        stats_drop: wp.array(dtype=wp.int32)
        stats_ttl: wp.array(dtype=wp.int32)
        stats_loop: wp.array(dtype=wp.int32)
