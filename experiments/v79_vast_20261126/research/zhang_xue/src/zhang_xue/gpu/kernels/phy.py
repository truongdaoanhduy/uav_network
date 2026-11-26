"""v5.7 On-GPU 3D Building NLoS & Blackout PHY Kernel (NVIDIA Warp, FP32, enable_backward=False).

Eliminates per-step CPU->GPU channel-gain copies (required for CUDA Graph capture at E=4096).
Matches `uav_v54_7radio_csmaca_sim.py` (`segment_hits_cylinder_3d_fp32`, `is_in_blackout`,
2.4 GHz Friis + alpha=3.0 NLoS + 12 dB excess loss, radio_alive gating) in FP32.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

# 2.4 GHz constants matching uav_v54_7radio_csmaca_sim.py
C_LIGHT_MPS = 299_792_458.0
FREQ_HZ = 2.4e9
LAMBDA_M = C_LIGHT_MPS / FREQ_HZ
G0_1M = (LAMBDA_M / (4.0 * math.pi)) ** 2
EXCESS_NLOS_LIN = 10.0 ** (-12.0 / 10.0)  # -12 dB
MAX_RANGE_M = 500.0
CELL_SIZE_M = 50.0
BLACKOUT_MANHATTAN_RADIUS = 3

# 5 canonical buildings: (bx, by, radius_m, height_m)
BUILDINGS_5_NP = np.array(
    [
        [280.0, 280.0, 45.0, 75.0],
        [520.0, 320.0, 45.0, 80.0],
        [480.0, 560.0, 45.0, 72.0],
        [740.0, 520.0, 45.0, 78.0],
        [320.0, 740.0, 45.0, 70.0],
    ],
    dtype=np.float32,
)

# 2 canonical blackout centers (cx_cell, cy_cell) matching CANONICAL_BLACKOUT_CENTERS_2
BLACKOUTS_2_NP = np.array([[15, 5], [14, 15]], dtype=np.int32)


def compute_channel_gain_3d_nlos_numpy(
    positions_xyz: np.ndarray,
    radio_alive: np.ndarray | None = None,
) -> np.ndarray:
    """Pure-NumPy FP32 reference matching the Warp GPU kernel for CPU-GPU parity tests.

    Args:
      positions_xyz: shape (E, 7, 3) float32
      radio_alive: optional shape (E, 7) uint8 (1=alive, 0=dead/battery-depleted)
    Returns:
      gain: shape (E, 7, 7) float32
    """
    pos = np.asarray(positions_xyz, dtype=np.float32)
    e_n = pos.shape[0]
    alive = np.ones((e_n, 7), dtype=np.uint8) if radio_alive is None else np.asarray(radio_alive, dtype=np.uint8)
    gain = np.zeros((e_n, 7, 7), dtype=np.float32)

    for e in range(e_n):
        in_blk = np.zeros(7, dtype=bool)
        for r in range(7):
            cx = int(np.clip(pos[e, r, 0] // CELL_SIZE_M, 0, 19))
            cy = int(np.clip(pos[e, r, 1] // CELL_SIZE_M, 0, 19))
            for k in range(2):
                if abs(cx - int(BLACKOUTS_2_NP[k, 0])) + abs(cy - int(BLACKOUTS_2_NP[k, 1])) <= BLACKOUT_MANHATTAN_RADIUS:
                    in_blk[r] = True
        for i in range(7):
            if alive[e, i] == 0 or in_blk[i]:
                continue
            for j in range(i + 1, 7):
                if alive[e, j] == 0 or in_blk[j]:
                    continue
                dx = float(pos[e, j, 0] - pos[e, i, 0])
                dy = float(pos[e, j, 1] - pos[e, i, 1])
                dz = float(pos[e, j, 2] - pos[e, i, 2])
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 1.0:
                    dist = 1.0
                nlos = False
                for b in range(5):
                    bx, by, br, bh = (
                        float(BUILDINGS_5_NP[b, 0]),
                        float(BUILDINGS_5_NP[b, 1]),
                        float(BUILDINGS_5_NP[b, 2]),
                        float(BUILDINGS_5_NP[b, 3]),
                    )
                    fx = float(pos[e, i, 0]) - bx
                    fy = float(pos[e, i, 1]) - by
                    a = dx * dx + dy * dy
                    if a < 1e-9:
                        if fx * fx + fy * fy <= br * br and min(float(pos[e, i, 2]), float(pos[e, j, 2])) <= bh and max(float(pos[e, i, 2]), float(pos[e, j, 2])) >= 0.0:
                            nlos = True
                            break
                        continue
                    bb = 2.0 * (fx * dx + fy * dy)
                    cc = fx * fx + fy * fy - br * br
                    disc = bb * bb - 4.0 * a * cc
                    if disc < 0.0:
                        continue
                    sd = math.sqrt(disc)
                    t1 = (-bb - sd) / (2.0 * a)
                    t2 = (-bb + sd) / (2.0 * a)
                    t_lo = max(0.0, min(t1, t2))
                    t_hi = min(1.0, max(t1, t2))
                    if t_lo <= t_hi:
                        z_lo = float(pos[e, i, 2]) + t_lo * dz
                        z_hi = float(pos[e, i, 2]) + t_hi * dz
                        if min(z_lo, z_hi) <= bh and max(z_lo, z_hi) >= 0.0:
                            nlos = True
                            break
                if nlos:
                    g = float(G0_1M / (dist ** 3.0) * EXCESS_NLOS_LIN)
                else:
                    g = float(G0_1M / (dist * dist))
                gain[e, i, j] = np.float32(g)
                gain[e, j, i] = np.float32(g)
    return gain


def v57_gpu_memory_budget(num_envs_per_gpu: int = 2048, replay_capacity: int = 300_000, obs_dim: int = 514) -> Dict[str, float]:
    """Audit ACTUALLY ALLOCATED Warp SoA, not a proposed compressed layout.

    The older v5.7 calculation claimed 315710 B/env as though a compressed
    payload pool had already been implemented. `WarpSevenRadioNetwork` still
    allocates SHAPES with two full 7x64x62 uint32 payload arrays; using the
    projected compressed value would undercount the actual device allocation.

    This remains a *planning estimate* because real peak PyTorch/Warp CUDA
    memory, graph capture, intermediates and fragmentation need T4 measurement.
    """
    from zhang_xue.learners.state_derived_replay import estimate_replay_bytes
    from zhang_xue.compat.warp_mac import backing_bytes_per_environment

    if min(num_envs_per_gpu, replay_capacity, obs_dim) <= 0:
        raise ValueError("All dimensions must be positive")
    actual_bytes_per_env = backing_bytes_per_environment()
    network_gib = actual_bytes_per_env * num_envs_per_gpu / (1024 ** 3)
    replay_gib = estimate_replay_bytes(replay_capacity, 6, obs_dim, 2, 7, store_state=False) / (1024 ** 3)
    provisional_other_gib = 3.0  # assumption, NOT measured CUDA allocator peak
    total_gib = network_gib + replay_gib + provisional_other_gib
    return {
        "num_envs_per_gpu": float(num_envs_per_gpu),
        "compressed_layout_implemented": 0.0,
        "actual_soa_bytes_per_env": float(actual_bytes_per_env),
        "network_soa_gib": round(network_gib, 3),
        "replay_buffer_gib": round(replay_gib, 3),
        "pytorch_cuda_overhead_assumed_gib": provisional_other_gib,
        "estimated_total_gib": round(total_gib, 3),
        "t4_vram_gib": 16.0,
        "fits_single_t4_16gib": float(total_gib < 14.5),
        "cuda_peak_verified": 0.0,
        "proposed_compressed_soa_bytes_per_env_unimplemented": 315_710.0,
    }


if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

    @wp.func
    def _is_blackout_wp(px: float, py: float) -> int:
        cx = wp.clamp(wp.int32(wp.floor(px / 50.0)), 0, 19)
        cy = wp.clamp(wp.int32(wp.floor(py / 50.0)), 0, 19)
        if wp.abs(cx - 15) + wp.abs(cy - 5) <= 3:
            return 1
        if wp.abs(cx - 14) + wp.abs(cy - 15) <= 3:
            return 1
        return 0

    @wp.func
    def _hits_cyl_3d_wp(
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
        bx: float, by: float, br: float, bh: float,
    ) -> int:
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        fx = x0 - bx
        fy = y0 - by
        a = dx * dx + dy * dy
        if a < 1.0e-9:
            if fx * fx + fy * fy <= br * br and wp.min(z0, z1) <= bh:
                return 1
            return 0
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - br * br
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return 0
        sd = wp.sqrt(disc)
        t1 = (-b - sd) / (2.0 * a)
        t2 = (-b + sd) / (2.0 * a)
        t_lo = wp.max(0.0, wp.min(t1, t2))
        t_hi = wp.min(1.0, wp.max(t1, t2))
        if t_lo <= t_hi:
            z_lo = z0 + t_lo * dz
            z_hi = z0 + t_hi * dz
            if wp.min(z_lo, z_hi) <= bh and wp.max(z_lo, z_hi) >= 0.0:
                return 1
        return 0

    @wp.kernel(enable_backward=False)
    def compute_channel_gain_3d_nlos_wp(
        positions_xyz: wp.array3d(dtype=wp.float32),  # (E, 7, 3)
        radio_alive: wp.array2d(dtype=wp.uint8),      # (E, 7)
        buildings: wp.array2d(dtype=wp.float32),      # (5, 4)
        gain_out: wp.array3d(dtype=wp.float32),       # (E, 7, 7)
    ):
        e, i, j = wp.tid()
        if i == j:
            gain_out[e, i, j] = 0.0
            return
        if radio_alive[e, i] == wp.uint8(0) or radio_alive[e, j] == wp.uint8(0):
            gain_out[e, i, j] = 0.0
            return
        x0 = positions_xyz[e, i, 0]
        y0 = positions_xyz[e, i, 1]
        z0 = positions_xyz[e, i, 2]
        x1 = positions_xyz[e, j, 0]
        y1 = positions_xyz[e, j, 1]
        z1 = positions_xyz[e, j, 2]
        if _is_blackout_wp(x0, y0) != 0 or _is_blackout_wp(x1, y1) != 0:
            gain_out[e, i, j] = 0.0
            return
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1.0:
            dist = 1.0
        nlos = int(0)
        for b in range(5):
            if _hits_cyl_3d_wp(
                x0, y0, z0, x1, y1, z1,
                buildings[b, 0], buildings[b, 1], buildings[b, 2], buildings[b, 3]
            ) != 0:
                nlos = 1
        g0 = float(9.8902975e-5)  # (c / (4 * pi * 2.4e9))^2
        if nlos != 0:
            gain_out[e, i, j] = g0 / (dist * dist * dist) * float(0.063095734)
        else:
            gain_out[e, i, j] = g0 / (dist * dist)
