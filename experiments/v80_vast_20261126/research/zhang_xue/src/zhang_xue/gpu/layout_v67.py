"""v6.7 Parameterized Compressed Warp SoA Schema with 4D `ack_origin_gen` (`int16`).

Replaces the 3D `ack_pos/ack_neg/ack_gen` (100,800 B/env) with a single packed 4D
`ack_origin_gen` (`int16`, shape `(num_uavs, radio_n, num_uavs * num_cells)` = 201,600 B/env),
which represents freshness only when each source/cell has a strictly increasing
generation timestamp. Same-timestamp count changes cannot be represented. The
ACK watermark alone saves 604,800 B/env versus naive 4D `(pos, neg, ts)` at 6/20x20;
this does not prove event, raw-count or whole-simulator equivalence.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

Q_V67: int = 32
Q_V67_MAX: int = 127  # active_slot is int8 (-1 sentinel -> valid slots 0..126)
K_V67: int = 62
DEDUP_RING_V67: int = 16
ENV_SCALAR_FIELDS_V67 = frozenset((
    "now", "mission_epoch_us", "episode", "overflow", "event_count", "next_pid", "next_seq",
    "stats_gcs", "stats_peer", "stats_mac_ack", "stats_e2e", "stats_retry", "stats_drop",
    "stats_ttl", "stats_loop",
))


def make_shapes_v67(
    num_uavs: int = 6,
    grid_w: int = 20,
    q_cap: int = Q_V67,
    k_entries: int = K_V67,
) -> dict:
    """Parameterized compressed SoA shapes for any (num_uavs <= 30, grid_w^2 <= 4096)."""
    u_n = int(num_uavs)
    r_n = u_n + 1
    cells = int(grid_w) * int(grid_w)
    p_cap = max(1, u_n)
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
        "hello_offer_pose": ((r_n, 4), np.float32),
        "hello_active_pose": ((r_n, 4), np.float32),
        "hello_active_at_us": ((r_n,), np.int32),
        "hello_rx_pose": ((r_n, r_n, 4), np.float32),
        "hello_rx_at_us": ((r_n, r_n), np.int32),
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
        # Hop-local admission dedup is separate from application delivery dedup.
        "received_pid": ((r_n, DEDUP_RING_V67), np.int32),
        "received_ptr": ((r_n,), np.uint8),
        "delivered_seq": ((r_n, r_n, DEDUP_RING_V67), np.int32),
        "delivered_ptr": ((r_n, r_n), np.uint8),
        "delivered_ack_seq": ((r_n, r_n, DEDUP_RING_V67), np.int32),
        "delivered_ack_ptr": ((r_n, r_n), np.uint8),
        "ledger_pos": ((r_n, u_n, cells), np.uint16),
        "ledger_neg": ((r_n, u_n, cells), np.uint16),
        "ledger_gen": ((r_n, u_n, cells), np.int16),
        "ledger_rx_age_s": ((r_n, u_n, cells), np.float16),
        # Packed 4D multi-origin E2E ACK watermark: (src, dst, origin * cells + cid)
        "ack_origin_gen": ((u_n, r_n, u_n * cells), np.int16),
        "dirty_dest": ((u_n, r_n), np.uint8),
        "pending_seq": ((u_n, p_cap), np.int32),
        "pending_dst": ((u_n, p_cap), np.int8),
        "pending_valid": ((u_n, p_cap), np.uint8),
        "pending_expiry": ((u_n, p_cap), np.int32),
        "pending_entries": ((u_n, p_cap), np.uint8),
        "pending_word0": ((u_n, p_cap, k_entries), np.uint32),
        "pending_word1": ((u_n, p_cap, k_entries), np.uint32),
        "gcs_latest": ((cells,), np.int16),
        "gcs_last_rx_us": ((cells,), np.int32),
        "stats_gcs": ((1,), np.int32),
        "stats_peer": ((1,), np.int32),
        "stats_mac_ack": ((1,), np.int32),
        "stats_e2e": ((1,), np.int32),
        "stats_retry": ((1,), np.int32),
        "stats_drop": ((1,), np.int32),
        "stats_ttl": ((1,), np.int32),
        "stats_loop": ((1,), np.int32),
    }


def backing_bytes_per_environment_v67(
    num_uavs: int = 6, grid_w: int = 20, q_cap: int = Q_V67
) -> int:
    if isinstance(q_cap, bool) or not isinstance(q_cap, int) or not (1 <= q_cap <= Q_V67_MAX):
        raise ValueError(f"q_cap must be an integer in [1, {Q_V67_MAX}]")
    shapes = make_shapes_v67(num_uavs=num_uavs, grid_w=grid_w, q_cap=q_cap)
    return sum(math.prod(sh) * np.dtype(dtype).itemsize for sh, dtype in shapes.values())


def assert_network_allocation_fits(
    num_envs: int, num_uavs: int, grid_w: int, *, free_bytes: int,
    q_cap: int = Q_V67, reserve_bytes: int = 3 * 1024**3, safety_fraction: float = 0.85,
) -> dict:
    """Preflight *network SoA* against measured free VRAM, including headroom.

    Does NOT estimate model activations, temporary buffers, training replay or
    CUDA compilation overhead. Passing is necessary, never sufficient.
    """
    if not (isinstance(num_envs, int) and num_envs > 0
            and isinstance(free_bytes, int) and free_bytes >= 0
            and isinstance(reserve_bytes, int) and reserve_bytes >= 0
            and 0.0 < safety_fraction <= 1.0):
        raise ValueError("invalid memory preflight arguments")
    estimated = backing_bytes_per_environment_v67(num_uavs, grid_w, q_cap=q_cap) * num_envs
    available = max(0, int(free_bytes * safety_fraction) - reserve_bytes)
    if estimated > available:
        raise MemoryError(
            f"Network SoA {estimated} B exceeds safe {available} B; "
            f"reduce num_envs/num_uavs/grid_w or use another GPU"
        )
    return {'estimated_network_bytes': estimated,
            'safe_allocation_limit_bytes': available,
            'raw_free_bytes': free_bytes,
            'note': 'Necessary network-only check, not full training fit'}


def v67_gpu_memory_budget(
    num_envs_per_gpu: int = 2048,
    num_uavs: int = 6,
    grid_w: int = 20,
    replay_capacity: int = 300_000,
    obs_dim: int = 514,
    q_cap: int = Q_V67,
) -> Dict[str, float]:
    """VRAM budget for executable v6.7 Warp Data-Mule runtime on 16 GiB T4."""
    from zhang_xue.learners.state_derived_replay import estimate_replay_bytes

    soa_bytes = backing_bytes_per_environment_v67(num_uavs=num_uavs, grid_w=grid_w, q_cap=q_cap)
    network_gib = (soa_bytes * num_envs_per_gpu) / (1024 ** 3)
    replay_gib = estimate_replay_bytes(
        replay_capacity, num_uavs, obs_dim, 2, 3, store_state=False
    ) / (1024 ** 3)
    pytorch_cuda_reserve_gib = 3.0
    total_gib = network_gib + replay_gib + pytorch_cuda_reserve_gib
    return {
        "layout_id": "v67_datamule_compressed",
        "num_envs_per_gpu": float(num_envs_per_gpu),
        "num_uavs": float(num_uavs),
        "grid_w": float(grid_w),
        "soa_bytes_per_env": float(soa_bytes),
        "network_soa_gib": round(network_gib, 3),
        "replay_buffer_gib": round(replay_gib, 3),
        "pytorch_cuda_reserve_gib": pytorch_cuda_reserve_gib,
        "estimated_total_gib": round(total_gib, 3),
        "t4_vram_gib": 16.0,
        "fits_single_t4_16gib": float(total_gib < 14.5),
    }
