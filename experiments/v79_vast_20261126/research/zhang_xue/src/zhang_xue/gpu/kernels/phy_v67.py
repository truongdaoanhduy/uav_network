"""v6.7/v7.1 Parameterized 3D Urban NLoS + Blackout PHY Kernels (race-free row ownership)."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from zhang_xue.gpu.state_v67 import HAS_WARP, wp

G0_1M: float = (299_792_458.0 / (4.0 * math.pi * 2.4e9)) ** 2
EXCESS_NLOS_LIN: float = 10.0 ** (-12.0 / 10.0)


def compute_channel_gain_v67_np(
    pos: np.ndarray,
    buildings: Sequence[Tuple[float, float, float, float]],
    blackout_centers: Sequence[Tuple[int, int]],
    blackout_r: int,
    grid_w: int,
    cell_size_m: float,
    alive: np.ndarray | None = None,
    *,
    obstacle_model: str = "binary_nlos",
    per_building_loss_db: float = 12.0,
) -> np.ndarray:
    """Parameterized NumPy reference matching `compute_channel_gain_v67` bit-for-bit."""
    if obstacle_model not in ("binary_nlos", "multi_obstacle", "rf_opaque"):
        raise ValueError(f"Invalid obstacle_model: {obstacle_model}")
    if not math.isfinite(per_building_loss_db) or per_building_loss_db < 0:
        raise ValueError("per_building_loss_db must be finite and non-negative")
    p = np.asarray(pos, dtype=np.float32)
    e_n, r_n, _ = p.shape
    gain = np.zeros((e_n, r_n, r_n), dtype=np.float32)
    b_arr = np.asarray(buildings, dtype=np.float32)
    bo_arr = np.asarray(blackout_centers, dtype=np.int32)

    def _in_bo(x: float, y: float) -> bool:
        cx = max(0, min(grid_w - 1, int(math.floor(x / cell_size_m))))
        cy = max(0, min(grid_w - 1, int(math.floor(y / cell_size_m))))
        for m in range(len(bo_arr)):
            if abs(cx - int(bo_arr[m, 0])) + abs(cy - int(bo_arr[m, 1])) <= blackout_r:
                return True
        return False

    for e in range(e_n):
        for i in range(r_n):
            if alive is not None and not bool(alive[e, i]):
                continue
            if _in_bo(float(p[e, i, 0]), float(p[e, i, 1])):
                continue
            for j in range(i + 1, r_n):
                if alive is not None and not bool(alive[e, j]):
                    continue
                if _in_bo(float(p[e, j, 0]), float(p[e, j, 1])):
                    continue
                dx = float(p[e, j, 0]) - float(p[e, i, 0])
                dy = float(p[e, j, 1]) - float(p[e, i, 1])
                dz = float(p[e, j, 2]) - float(p[e, i, 2])
                dist = max(1.0, math.sqrt(dx * dx + dy * dy + dz * dz))
                n_blocks = 0
                for b in range(len(b_arr)):
                    bx, by, br, bh = float(b_arr[b, 0]), float(b_arr[b, 1]), float(b_arr[b, 2]), float(b_arr[b, 3])
                    fx = float(p[e, i, 0]) - bx
                    fy = float(p[e, i, 1]) - by
                    a = dx * dx + dy * dy
                    if a < 1e-9:
                        if fx * fx + fy * fy <= br * br and min(float(p[e, i, 2]), float(p[e, j, 2])) <= bh and max(float(p[e, i, 2]), float(p[e, j, 2])) >= 0.0:
                            n_blocks += 1
                            if obstacle_model != "multi_obstacle":
                                break
                        continue
                    bb = 2.0 * (fx * dx + fy * dy)
                    cc = fx * fx + fy * fy - br * br
                    disc = bb * bb - 4.0 * a * cc
                    if disc < 0.0:
                        continue
                    sd = math.sqrt(disc)
                    t_lo = max(0.0, min((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
                    t_hi = min(1.0, max((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
                    if t_lo <= t_hi:
                        z_lo = float(p[e, i, 2]) + t_lo * dz
                        z_hi = float(p[e, i, 2]) + t_hi * dz
                        if min(z_lo, z_hi) <= bh and max(z_lo, z_hi) >= 0.0:
                            n_blocks += 1
                            if obstacle_model != "multi_obstacle":
                                break
                if obstacle_model == "rf_opaque":
                    g = 0.0 if n_blocks else G0_1M / (dist * dist)
                elif obstacle_model == "multi_obstacle":
                    g = G0_1M / (dist * dist) * 10.0 ** (-n_blocks * per_building_loss_db / 10.0)
                else:
                    g = (G0_1M / (dist ** 3.0) * EXCESS_NLOS_LIN) if n_blocks else (G0_1M / (dist * dist))
                gain[e, i, j] = np.float32(g)
                gain[e, j, i] = np.float32(g)
    return gain


if HAS_WARP:
    @wp.func
    def _in_blackout_v67(
        x: float, y: float, grid_w: int, cell_size_m: float,
        blackout_xy: wp.array2d(dtype=wp.int32), n_bo: int, blackout_r: int,
    ) -> int:
        cx = wp.clamp(int(wp.floor(x / cell_size_m)), 0, grid_w - 1)
        cy = wp.clamp(int(wp.floor(y / cell_size_m)), 0, grid_w - 1)
        for m in range(n_bo):
            if wp.abs(cx - blackout_xy[m, 0]) + wp.abs(cy - blackout_xy[m, 1]) <= blackout_r:
                return 1
        return 0

    @wp.kernel
    def compute_channel_gain_v67(
        pos: wp.array3d(dtype=wp.float32),
        alive: wp.array2d(dtype=wp.uint8),
        buildings: wp.array2d(dtype=wp.float32),
        blackout_xy: wp.array2d(dtype=wp.int32),
        gain: wp.array3d(dtype=wp.float32),
        n_radio: int,
        n_bld: int,
        n_bo: int,
        blackout_r: int,
        grid_w: int,
        cell_size_m: float,
        obstacle_model_id: int,
        per_building_loss_db: float,
    ):
        # Thread (e, i) owns row gain[e, i, :] exclusively to avoid cross-row write races.
        e, i = wp.tid()
        if i >= n_radio:
            return
        if alive[e, i] == wp.uint8(0):
            for j in range(n_radio):
                gain[e, i, j] = 0.0
            return
        xi, yi, zi = pos[e, i, 0], pos[e, i, 1], pos[e, i, 2]
        bo_i = _in_blackout_v67(xi, yi, grid_w, cell_size_m, blackout_xy, n_bo, blackout_r)
        for j in range(n_radio):
            if j == i or alive[e, j] == wp.uint8(0) or bo_i == 1:
                gain[e, i, j] = 0.0
                continue
            xj, yj, zj = pos[e, j, 0], pos[e, j, 1], pos[e, j, 2]
            if _in_blackout_v67(xj, yj, grid_w, cell_size_m, blackout_xy, n_bo, blackout_r) == 1:
                gain[e, i, j] = 0.0
                continue
            dx, dy, dz = xj - xi, yj - yi, zj - zi
            dist = wp.max(1.0, wp.sqrt(dx * dx + dy * dy + dz * dz))
            n_blocks = int(0)
            a = dx * dx + dy * dy
            for b in range(n_bld):
                bx, by, br, bh = buildings[b, 0], buildings[b, 1], buildings[b, 2], buildings[b, 3]
                fx, fy = xi - bx, yi - by
                if a < 1.0e-9:
                    if fx * fx + fy * fy <= br * br and wp.min(zi, zj) <= bh and wp.max(zi, zj) >= 0.0:
                        n_blocks += 1
                        if obstacle_model_id != 1:
                            break
                    continue
                bb = 2.0 * (fx * dx + fy * dy)
                cc = fx * fx + fy * fy - br * br
                disc = bb * bb - 4.0 * a * cc
                if disc < 0.0:
                    continue
                sd = wp.sqrt(disc)
                t_lo = wp.max(0.0, wp.min((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
                t_hi = wp.min(1.0, wp.max((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
                if t_lo <= t_hi:
                    z_lo, z_hi = zi + t_lo * dz, zi + t_hi * dz
                    if wp.min(z_lo, z_hi) <= bh and wp.max(z_lo, z_hi) >= 0.0:
                        n_blocks += 1
                        if obstacle_model_id != 1:
                            break
            los_gain = float(G0_1M) / (dist * dist)
            g = los_gain
            if obstacle_model_id == 0 and n_blocks > 0:
                g = float(G0_1M) / (dist * dist * dist) * float(EXCESS_NLOS_LIN)
            elif obstacle_model_id == 1 and n_blocks > 0:
                # 10^(-n*L/10) = exp(-ln(10)/10 * n * L), no NLoS double count.
                g = los_gain * wp.exp(-0.2302585092994046 * float(n_blocks) * per_building_loss_db)
            elif obstacle_model_id == 2 and n_blocks > 0:
                g = 0.0
            gain[e, i, j] = g
