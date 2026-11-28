"""v6.7 Warp Structs (`SimDimsV67`, `NetworkBuffersV67`) & UavNetSim PHY Constants."""
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
UAVNETSIM_NOISE_W: float = 1e-3 * 10.0 ** ((-174.0 + 10.0 * math.log10(22.0e6) + 7.0) / 10.0)
UAVNETSIM_SNR_TH_LIN: float = 10.0 ** (6.0 / 10.0)  # 3.9810717

PKT_DATA: int = 0
PKT_E2E_ACK: int = 1
PKT_HELLO: int = 2
FAULT_GENERATION: int = 3

ST_IDLE: int = 0
ST_DIFS: int = 1
ST_BACKOFF: int = 2
ST_TX: int = 3
ST_WAIT_ACK: int = 4
ST_SIFS: int = 5
ST_TX_ACK: int = 6

NEVER_US: int = 2_000_000_000
DIFS_US: int = 50
SIFS_US: int = 10
SLOT_US: int = 20
TX_MAC_ACK_US: int = 120
ACK_TIMEOUT_US: int = 180
CW_MIN: int = 31
MAX_RETX: int = 5
PKT_TTL_US: int = 10_000_000
HELLO_INT_US: int = 500_000
HELLO_V89_TX_US: int = 440  # 360us legacy + 20B telemetry at 2Mbps = 80us
ROUTE_TTL_US: int = 2_000_000

if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

    @wp.struct
    class SimDimsV67:
        n_uav: wp.int32
        n_radio: wp.int32
        gcs_id: wp.int32
        grid_w: wp.int32
        num_cells: wp.int32
        q_cap: wp.int32
        p_cap: wp.int32
        k_entries: wp.int32
        max_hops: wp.int32
        scalable_wire: wp.int32
        enable_data_mule: wp.int32
        enable_peer: wp.int32
        enable_peer_telemetry: wp.int32
        decentralized_peer_select: wp.int32
        aoi_ewma: wp.int32
        aoi_eta: wp.float32
        rx_lock_on_cca: wp.int32
        node_mask_all: wp.uint32
        difs_us: wp.int32
        sifs_us: wp.int32
        slot_us: wp.int32
        cw_min: wp.int32
        cw_inclusive_upper: wp.int32
        max_retx: wp.int32
        ack_timeout_us: wp.int32

    @wp.struct
    class NetworkBuffersV67:
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
        hello_offer_pose: wp.array3d(dtype=wp.float32)
        hello_active_pose: wp.array3d(dtype=wp.float32)
        hello_active_at_us: wp.array2d(dtype=wp.int32)
        hello_rx_pose: wp.array4d(dtype=wp.float32)
        hello_rx_at_us: wp.array3d(dtype=wp.int32)
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
        received_pid: wp.array3d(dtype=wp.int32)
        received_ptr: wp.array2d(dtype=wp.uint8)
        delivered_seq: wp.array4d(dtype=wp.int32)
        delivered_ptr: wp.array3d(dtype=wp.uint8)
        delivered_ack_seq: wp.array4d(dtype=wp.int32)
        delivered_ack_ptr: wp.array3d(dtype=wp.uint8)
        ledger_pos: wp.array4d(dtype=wp.uint16)
        ledger_neg: wp.array4d(dtype=wp.uint16)
        ledger_gen: wp.array4d(dtype=wp.int16)
        ledger_rx_age_s: wp.array4d(dtype=wp.float16)
        ack_origin_gen: wp.array4d(dtype=wp.int16)
        dirty_dest: wp.array3d(dtype=wp.uint8)
        pending_seq: wp.array3d(dtype=wp.int32)
        pending_dst: wp.array3d(dtype=wp.int8)
        pending_valid: wp.array3d(dtype=wp.uint8)
        pending_expiry: wp.array3d(dtype=wp.int32)
        pending_entries: wp.array3d(dtype=wp.uint8)
        pending_word0: wp.array4d(dtype=wp.uint32)
        pending_word1: wp.array4d(dtype=wp.uint32)
        gcs_latest: wp.array2d(dtype=wp.int16)
        gcs_last_rx_us: wp.array2d(dtype=wp.int32)
        stats_gcs: wp.array(dtype=wp.int32)
        stats_peer: wp.array(dtype=wp.int32)
        stats_mac_ack: wp.array(dtype=wp.int32)
        stats_e2e: wp.array(dtype=wp.int32)
        stats_retry: wp.array(dtype=wp.int32)
        stats_drop: wp.array(dtype=wp.int32)
        stats_ttl: wp.array(dtype=wp.int32)
        stats_loop: wp.array(dtype=wp.int32)
