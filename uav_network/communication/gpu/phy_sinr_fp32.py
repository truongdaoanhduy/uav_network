"""Pure FP32 NVIDIA Warp PHY/Channel Kernel for 2.5D Urban LOS/NLOS & O(N^2) SINR.

Designed for Kaggle Tesla T4 (Turing SM 7.5, FP32:FP64 = 32:1, 32 warps/SM).
- Strictly wp.float32 / wp.int32 (strictly single-precision, zero FP64).
- 100% CUDA Graph safe (static arrays, zero dynamic allocations, zero host sync).
- Log-Distance Urban 2.4 GHz / 22 MHz / 0.1 W (20 dBm):
    * FSPL(1m) = 40.05 dB, alpha_LOS = 2.5 -> natural d_LOS ~ 503 m at SINR >= 6 dB (NF=7 dB)
    * NLOS building shadowing (+20 dB) -> natural d_NLOS ~ 80 m (forces genuine 2-3 hop relays)
    * CCA carrier-sense threshold = -82 dBm (creates realistic hidden/exposed terminals).
"""

from __future__ import annotations

import math

try:
    import warp as wp
except ImportError:  # pragma: no cover - allows syntax/unit import on CPU hosts without warp
    wp = None

NUM_UAVS = 6
NUM_NODES = 7  # 0..5 = UAVs, 6 = GCS (sink)
GCS_IDX = 6

# Precomputed IEEE 802.11b 2.4 GHz / 22 MHz constants (FP32)
TX_POWER_W = 0.1                               # 20 dBm fixed (no continuous tx_power RL action)
FSPL_1M_DB = 40.05                             # 20*log10(4*pi*2.4e9 / 3e8)
LOS_EXPONENT = 2.5                             # Log-distance urban LOS -> d_LOS ~ 503 m
NLOS_EXTRA_LOSS_DB = 20.0                      # Urban building occlusion -> d_NLOS ~ 80 m
NOISE_FLOOR_DBM = -93.58                       # -174 + 10*log10(22e6) + 7 dB NF
NOISE_FLOOR_W = 10.0 ** ((NOISE_FLOOR_DBM - 30.0) / 10.0)
SINR_THRESHOLD_LIN = 10.0 ** (6.0 / 10.0)      # 6 dB -> ~3.981 linear
CCA_THRESHOLD_W = 10.0 ** ((-82.0 - 30.0) / 10.0)  # -82 dBm carrier sense threshold


if wp is not None:

    @wp.func
    def segment_hits_cylinder_2d(
        ax: wp.float32,
        ay: wp.float32,
        bx: wp.float32,
        by: wp.float32,
        cx: wp.float32,
        cy: wp.float32,
        radius: wp.float32,
    ) -> wp.int32:
        """Return 1 if 2D segment A->B intersects circle (C, radius), else 0."""
        dx = bx - ax
        dy = by - ay
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-6:
            return wp.int32(0)
        t = ((cx - ax) * dx + (cy - ay) * dy) / len_sq
        t = wp.clamp(t, wp.float32(0.0), wp.float32(1.0))
        px = ax + t * dx - cx
        py = ay + t * dy - cy
        if px * px + py * py <= radius * radius:
            return wp.int32(1)
        return wp.int32(0)

    @wp.func
    def check_link_los_fp32(
        env_id: wp.int32,
        ax: wp.float32,
        ay: wp.float32,
        az: wp.float32,
        bx: wp.float32,
        by: wp.float32,
        bz: wp.float32,
        obstacles: wp.array2d(dtype=wp.vec4f),  # (num_envs, max_obs) -> (cx, cy, r, h)
        num_obstacles: wp.int32,
    ) -> wp.int32:
        """Return 1 if 2.5D link A->B is Line-of-Sight (clear of all tall buildings), else 0 (NLOS)."""
        min_z = wp.min(az, bz)
        for o in range(num_obstacles):
            ob = obstacles[env_id, o]
            if ob[3] > min_z:
                if segment_hits_cylinder_2d(ax, ay, bx, by, ob[0], ob[1], ob[2]) == wp.int32(1):
                    return wp.int32(0)
        return wp.int32(1)

    @wp.func
    def rx_power_watts_fp32(dist_m: wp.float32, is_los: wp.int32) -> wp.float32:
        """Compute received power in Watts under Log-Distance Urban LOS/NLOS model."""
        d = wp.max(dist_m, wp.float32(1.0))
        pl_db = wp.float32(FSPL_1M_DB) + wp.float32(10.0 * LOS_EXPONENT) * wp.log(d) / wp.float32(2.302585093)
        if is_los == wp.int32(0):
            pl_db = pl_db + wp.float32(NLOS_EXTRA_LOSS_DB)
        # P_rx(W) = P_tx(W) * 10^(-PL_dB / 10)
        return wp.float32(TX_POWER_W) * wp.pow(wp.float32(10.0), -pl_db * wp.float32(0.1))

    @wp.kernel
    def compute_phy_matrices_fp32(
        node_pos: wp.array2d(dtype=wp.vec3f),      # (num_envs, 7): 0..5 UAVs, 6 GCS
        obstacles: wp.array2d(dtype=wp.vec4f),     # (num_envs, max_obs): (cx, cy, r, h)
        num_obstacles: wp.int32,
        # Outputs (static pre-allocated GPU buffers, zero dynamic alloc)
        dist_mat: wp.array3d(dtype=wp.float32),    # (num_envs, 7, 7)
        los_mat: wp.array3d(dtype=wp.int32),       # (num_envs, 7, 7)
        rx_pwr_mat: wp.array3d(dtype=wp.float32),  # (num_envs, 7, 7) in Watts
        cca_hear_mat: wp.array3d(dtype=wp.int32),  # (num_envs, 7, 7): 1 if Rx >= CCA (-82 dBm)
    ):
        """1 thread = 1 env; unrolled 7x7 symmetric PHY precomputation with zero atomics."""
        e = wp.tid()
        for i in range(wp.static(NUM_NODES)):
            pi = node_pos[e, i]
            dist_mat[e, i, i] = wp.float32(0.0)
            los_mat[e, i, i] = wp.int32(1)
            rx_pwr_mat[e, i, i] = wp.float32(TX_POWER_W)
            cca_hear_mat[e, i, i] = wp.int32(1)
            for j in range(i + 1, wp.static(NUM_NODES)):
                pj = node_pos[e, j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                dz = pi[2] - pj[2]
                d = wp.sqrt(dx * dx + dy * dy + dz * dz)
                los = check_link_los_fp32(e, pi[0], pi[1], pi[2], pj[0], pj[1], pj[2], obstacles, num_obstacles)
                prx = rx_power_watts_fp32(d, los)
                hear = wp.int32(1) if prx >= wp.float32(CCA_THRESHOLD_W) else wp.int32(0)

                dist_mat[e, i, j] = d
                dist_mat[e, j, i] = d
                los_mat[e, i, j] = los
                los_mat[e, j, i] = los
                rx_pwr_mat[e, i, j] = prx
                rx_pwr_mat[e, j, i] = prx
                cca_hear_mat[e, i, j] = hear
                cca_hear_mat[e, j, i] = hear


def analytical_link_budget_summary() -> dict:
    """Pure-Python analytical verification of the FP32 PHY link budget constants."""
    mapl_db = 20.0 - (NOISE_FLOOR_DBM + 6.0)
    d_los = 10.0 ** ((mapl_db - FSPL_1M_DB) / (10.0 * LOS_EXPONENT))
    d_nlos = 10.0 ** ((mapl_db - FSPL_1M_DB - NLOS_EXTRA_LOSS_DB) / (10.0 * LOS_EXPONENT))
    return {
        "tx_power_dbm": 20.0,
        "noise_floor_dbm": round(NOISE_FLOOR_DBM, 2),
        "rx_sensitivity_dbm": round(NOISE_FLOOR_DBM + 6.0, 2),
        "mapl_db": round(mapl_db, 2),
        "d_los_m": round(d_los, 1),
        "d_nlos_m": round(d_nlos, 1),
        "cca_threshold_dbm": -82.0,
    }
