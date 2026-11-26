"""Proposed (NOT ACTIVE) compressed Warp seven-radio SoA schema.

Theoretical design footprint: 384,549 B/env vs 635,931 B/env (v6.1).
Includes proposed per-UAV `tx_power_w` and `q_order`/`q_used` FIFO scheme,
which are NOT used by gpu.runtime/state/mac_functions and do not eliminate payload swaps, and `step_peer_info_gain`, keeping `gain`/`rx_sig`/`rx_interf`
strictly in float32 to prevent subnormal underflow at 2.4 GHz.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

Q_V62 = 32
P_V62 = 6
K_V62 = 62

SHAPES_V62 = {
    "now": ((1,), np.int32),
    "mission_epoch_us": ((1,), np.int32),
    "episode": ((1,), np.int32),
    "overflow": ((1,), np.uint8),
    "event_count": ((1,), np.int32),
    "next_pid": ((1,), np.int32),
    "next_seq": ((1,), np.int32),
    "rng": ((7,), np.uint32),
    "radio_alive": ((7,), np.uint8),
    "tx_power_w": ((7,), np.float32),
    "gain": ((7, 7), np.float32),
    "state": ((7,), np.int8),
    "deadline": ((7,), np.int32),
    "hello_due": ((7,), np.int32),
    "hello_pending": ((7,), np.uint8),
    "hello_dv_hop": ((7, 7), np.int8),
    "hello_dv_dist": ((7, 7), np.int8),
    "hello_dv_mask": ((7, 7), np.uint8),
    "active_slot": ((7,), np.int8),
    "next_hop": ((7,), np.int8),
    "tx_pkt_id": ((7,), np.int32),
    "ack_dst": ((7,), np.int8),
    "ack_pid": ((7,), np.int32),
    "retries": ((7,), np.uint8),
    "backoff_rem": ((7,), np.int32),
    "frozen": ((7,), np.uint8),
    "rx_lock": ((7,), np.int8),
    "rx_corrupt": ((7,), np.uint8),
    "rx_interf": ((7,), np.float32),
    "rx_sig": ((7,), np.float32),
    "tx_airtime": ((7,), np.int32),
    # Zero-copy FIFO order table (32 B/radio instead of 992 B payload swap)
    "q_order": ((7, Q_V62), np.uint8),
    "q_used": ((7, Q_V62), np.uint8),
    "q_count": ((7,), np.uint8),
    "q_type": ((7, Q_V62), np.uint8),
    "q_pid": ((7, Q_V62), np.int32),
    "q_src": ((7, Q_V62), np.int8),
    "q_dst": ((7, Q_V62), np.int8),
    "q_seq": ((7, Q_V62), np.int32),
    "q_expiry": ((7, Q_V62), np.int32),
    "q_visited": ((7, Q_V62), np.uint8),
    "q_hops": ((7, Q_V62), np.uint8),
    "q_entries": ((7, Q_V62), np.uint8),
    "q_word0": ((7, Q_V62, K_V62), np.uint32),
    "q_word1": ((7, Q_V62, K_V62), np.uint32),
    "route_hop": ((7, 7), np.int8),
    "route_dist": ((7, 7), np.int8),
    "route_mask": ((7, 7), np.uint8),
    "route_expiry": ((7, 7), np.int32),
    "delivered_seq": ((7, 7, 16), np.int32),
    "delivered_ptr": ((7, 7), np.uint8),
    "delivered_ack_seq": ((7, 7, 16), np.int32),
    "delivered_ack_ptr": ((7, 7), np.uint8),
    # Provenance ledger: uint16 stores either raw counts or UQ8.8 EWMA effective counts!
    "ledger_pos": ((7, 6, 400), np.uint16),
    "ledger_neg": ((7, 6, 400), np.uint16),
    "ledger_gen": ((7, 6, 400), np.int16),
    "ledger_rx_age_s": ((7, 6, 400), np.float16),
    "ack_pos": ((6, 7, 400), np.uint16),
    "ack_neg": ((6, 7, 400), np.uint16),
    "ack_gen": ((6, 7, 400), np.int16),
    # 1 pending slot per destination (P_V62=6 eliminates O(16) linear scans)
    "pending_seq": ((6, P_V62), np.int32),
    "pending_dst": ((6, P_V62), np.int8),
    "pending_valid": ((6, P_V62), np.uint8),
    "pending_expiry": ((6, P_V62), np.int32),
    "pending_entries": ((6, P_V62), np.uint8),
    "pending_word0": ((6, P_V62, K_V62), np.uint32),
    "pending_word1": ((6, P_V62, K_V62), np.uint32),
    "gcs_latest": ((400,), np.int16),
    "gcs_last_rx_us": ((400,), np.int32),
    "step_peer_info_gain": ((1,), np.float32),
    "stats_gcs": ((1,), np.int32),
    "stats_peer": ((1,), np.int32),
    "stats_mac_ack": ((1,), np.int32),
    "stats_e2e": ((1,), np.int32),
    "stats_retry": ((1,), np.int32),
    "stats_drop": ((1,), np.int32),
    "stats_ttl": ((1,), np.int32),
    "stats_loop": ((1,), np.int32),
}


def backing_bytes_per_environment_v62() -> int:
    """Design bytes/env ONLY; not the currently allocated Warp state."""
    return sum(math.prod(sh) * np.dtype(dtype).itemsize for sh, dtype in SHAPES_V62.values())


def v62_gpu_memory_budget(
    num_envs_per_gpu: int = 2048,
    replay_capacity: int = 300_000,
    obs_dim: int = 522,
) -> Dict[str, float]:
    """Planning-only compressed layout, NOT measured or enabled by runtime."""
    from zhang_xue.learners.state_derived_replay import estimate_replay_bytes

    soa_bytes = backing_bytes_per_environment_v62()
    network_gib = (soa_bytes * num_envs_per_gpu) / (1024 ** 3)
    replay_gib = estimate_replay_bytes(replay_capacity, 6, obs_dim, 3, 7, store_state=False) / (1024 ** 3)
    pytorch_cuda_reserve_gib = 3.0
    total_gib = network_gib + replay_gib + pytorch_cuda_reserve_gib
    return {
        "num_envs_per_gpu": float(num_envs_per_gpu),
        "soa_bytes_per_env_v62": float(soa_bytes),
        "network_soa_gib": round(network_gib, 3),
        "replay_buffer_gib": round(replay_gib, 3),
        "pytorch_cuda_reserve_gib": pytorch_cuda_reserve_gib,
        "estimated_total_gib": round(total_gib, 3),
        "t4_vram_gib": 16.0,
        "fits_single_t4_16gib": float(total_gib < 14.5),
    }


def make_shapes_v63(num_uavs: int = 6, grid_w: int = 20, q_cap: int = Q_V62, k_entries: int = K_V62) -> dict:
    """Parameterized compressed SoA shapes for any (num_uavs, grid_w) on Warp GPU (no RF power bloat)."""
    r_n = int(num_uavs) + 1
    cells = int(grid_w) * int(grid_w)
    p_cap = max(1, int(num_uavs))
    return {
        "now": ((1,), np.int32),
        "mission_epoch_us": ((1,), np.int32),
        "episode": ((1,), np.int32),
        "overflow": ((1,), np.uint8),
        "event_count": ((1,), np.int32),
        "next_pid": ((1,), np.int32),
        "next_seq": ((1,), np.int32),
        "rng": ((r_n,), np.uint32),
        "radio_alive": ((r_n,), np.uint8),
        "gain": ((r_n, r_n), np.float32),
        "state": ((r_n,), np.int8),
        "deadline": ((r_n,), np.int32),
        "hello_due": ((r_n,), np.int32),
        "hello_pending": ((r_n,), np.uint8),
        "hello_dv_hop": ((r_n, r_n), np.int8),
        "hello_dv_dist": ((r_n, r_n), np.int8),
        "hello_dv_mask": ((r_n, r_n), np.uint32),
        "active_slot": ((r_n,), np.int8),
        "next_hop": ((r_n,), np.int8),
        "tx_pkt_id": ((r_n,), np.int32),
        "ack_dst": ((r_n,), np.int8),
        "ack_pid": ((r_n,), np.int32),
        "retries": ((r_n,), np.uint8),
        "backoff_rem": ((r_n,), np.int32),
        "frozen": ((r_n,), np.uint8),
        "rx_lock": ((r_n,), np.int8),
        "rx_corrupt": ((r_n,), np.uint8),
        "rx_interf": ((r_n,), np.float32),
        "rx_sig": ((r_n,), np.float32),
        "tx_airtime": ((r_n,), np.int32),
        "q_order": ((r_n, q_cap), np.uint8),
        "q_used": ((r_n, q_cap), np.uint8),
        "q_count": ((r_n,), np.uint8),
        "q_type": ((r_n, q_cap), np.uint8),
        "q_pid": ((r_n, q_cap), np.int32),
        "q_src": ((r_n, q_cap), np.int8),
        "q_dst": ((r_n, q_cap), np.int8),
        "q_seq": ((r_n, q_cap), np.int32),
        "q_expiry": ((r_n, q_cap), np.int32),
        "q_visited": ((r_n, q_cap), np.uint32),
        "q_hops": ((r_n, q_cap), np.uint8),
        "q_entries": ((r_n, q_cap), np.uint8),
        "q_word0": ((r_n, q_cap, k_entries), np.uint32),
        "q_word1": ((r_n, q_cap, k_entries), np.uint32),
        "route_hop": ((r_n, r_n), np.int8),
        "route_dist": ((r_n, r_n), np.int8),
        "route_mask": ((r_n, r_n), np.uint32),
        "route_expiry": ((r_n, r_n), np.int32),
        "delivered_seq": ((r_n, r_n, 16), np.int32),
        "delivered_ptr": ((r_n, r_n), np.uint8),
        "delivered_ack_seq": ((r_n, r_n, 16), np.int32),
        "delivered_ack_ptr": ((r_n, r_n), np.uint8),
        "ledger_pos": ((r_n, num_uavs, cells), np.uint16),
        "ledger_neg": ((r_n, num_uavs, cells), np.uint16),
        "ledger_gen": ((r_n, num_uavs, cells), np.int16),
        "ledger_rx_age_s": ((r_n, num_uavs, cells), np.float16),
        "ack_pos": ((num_uavs, r_n, cells), np.uint16),
        "ack_neg": ((num_uavs, r_n, cells), np.uint16),
        "ack_gen": ((num_uavs, r_n, cells), np.int16),
        "pending_seq": ((num_uavs, p_cap), np.int32),
        "pending_dst": ((num_uavs, p_cap), np.int8),
        "pending_valid": ((num_uavs, p_cap), np.uint8),
        "pending_expiry": ((num_uavs, p_cap), np.int32),
        "pending_entries": ((num_uavs, p_cap), np.uint8),
        "pending_word0": ((num_uavs, p_cap, k_entries), np.uint32),
        "pending_word1": ((num_uavs, p_cap, k_entries), np.uint32),
        "gcs_latest": ((cells,), np.int16),
        "gcs_last_rx_us": ((cells,), np.int32),
        "step_peer_info_gain": ((1,), np.float32),
        "stats_gcs": ((1,), np.int32),
        "stats_peer": ((1,), np.int32),
        "stats_mac_ack": ((1,), np.int32),
        "stats_e2e": ((1,), np.int32),
        "stats_retry": ((1,), np.int32),
        "stats_drop": ((1,), np.int32),
        "stats_ttl": ((1,), np.int32),
        "stats_loop": ((1,), np.int32),
    }


def backing_bytes_per_environment_v63(num_uavs: int = 6, grid_w: int = 20) -> int:
    shapes = make_shapes_v63(num_uavs=num_uavs, grid_w=grid_w)
    return sum(math.prod(sh) * np.dtype(dtype).itemsize for sh, dtype in shapes.values())
