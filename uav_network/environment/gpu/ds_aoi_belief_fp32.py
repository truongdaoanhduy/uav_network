"""Pure FP32 NVIDIA Warp Dempster-Shafer Belief Map + Zeng (2019) Rotary-Wing Energy + CUDA Graph Wrapper.

Combines:
1. Timing-independent Dempster-Shafer (DS) belief map (Zhang et al., IEEE TIV 2024, Theorem 1)
   in pure wp.float32 — eliminates Bayesian prior=0.5 false-positive floods.
2. Zeng et al. (IEEE TWC 2019) rotary-wing propulsion power scaled to a 2 kg class UAV
   (Hover P(0)=168 W > Optimal Cruise P(11 m/s)~132 W) — eliminates hover-camping reward hack.
3. WarpSlottedUavEnvGraph: reference CUDA Graph capture harness (`wp.capture_begin`) using
   a non-default PyTorch stream (`wp.stream_from_torch`) and zero-copy static tensors.
"""

from __future__ import annotations

import math

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

GRID_W = 20
GRID_H = 20
NUM_CELLS = GRID_W * GRID_H
CELL_SIZE_M = 40.0
NUM_UAVS = 6

# Zeng et al. (2019) 2kg-class tactical quadrotor constants (Hover 168 W > Cruise ~132 W)
ZENG_P0 = 80.0
ZENG_PI = 88.0
ZENG_U_TIP = 120.0
ZENG_V0 = 4.03
ZENG_PARASITE_COEFF = 0.5 * 0.6 * 1.225 * 0.05 * 0.503  # 0.5 * d0 * rho * s * A


def zeng_rotary_power_w(speed_mps: float) -> float:
    """Pure-Python reference for Zeng (2019) rotary-wing propulsion power (Watts)."""
    v = max(0.0, float(speed_mps))
    p_blade = ZENG_P0 * (1.0 + 3.0 * (v * v) / (ZENG_U_TIP * ZENG_U_TIP))
    v_ratio = (v * v) / (2.0 * ZENG_V0 * ZENG_V0)
    p_induced = ZENG_PI * math.sqrt(max(0.0, math.sqrt(1.0 + v_ratio * v_ratio) - v_ratio))
    p_parasite = ZENG_PARASITE_COEFF * (v ** 3)
    return p_blade + p_induced + p_parasite


def ds_closed_form(n_pos: int, n_neg: int, mf: float = 0.55, me: float = 0.55) -> tuple[float, float, float]:
    """Pure-Python reference for Zhang et al. (IEEE TIV 2024) Eq. (17) closed-form DS fusion."""
    f = (1.0 - mf) ** n_pos
    e = (1.0 - me) ** n_neg
    denom = f + e - e * f
    m_e = (f - e * f) / denom
    m_f = (e - e * f) / denom
    m_u = (e * f) / denom
    return m_f, m_e, m_u


if wp is not None:

    @wp.func
    def ds_update_cell_fp32(
        n_pos: wp.int32,
        n_neg: wp.int32,
        mf: wp.float32,
        me: wp.float32,
    ) -> wp.vec3f:
        """Closed-form timing-independent DS belief (m_F, m_E, m_U) in pure FP32."""
        f = wp.pow(wp.float32(1.0) - mf, wp.float32(n_pos))
        e = wp.pow(wp.float32(1.0) - me, wp.float32(n_neg))
        denom = wp.max(f + e - e * f, wp.float32(1e-6))
        m_f = (e - e * f) / denom
        m_e = (f - e * f) / denom
        m_u = (e * f) / denom
        return wp.vec3f(m_f, m_e, m_u)

    @wp.func
    def zeng_power_fp32(speed: wp.float32) -> wp.float32:
        """Zeng et al. (2019) rotary-wing power in Watts (FP32)."""
        v2 = speed * speed
        p_blade = wp.float32(ZENG_P0) * (wp.float32(1.0) + wp.float32(3.0) * v2 / wp.float32(ZENG_U_TIP * ZENG_U_TIP))
        vr = v2 / wp.float32(2.0 * ZENG_V0 * ZENG_V0)
        inner = wp.max(wp.float32(0.0), wp.sqrt(wp.float32(1.0) + vr * vr) - vr)
        p_ind = wp.float32(ZENG_PI) * wp.sqrt(inner)
        p_par = wp.float32(ZENG_PARASITE_COEFF) * v2 * speed
        return p_blade + p_ind + p_par

    @wp.kernel
    def ds_sense_and_energy_step_fp32(
        uav_pos: wp.array2d(dtype=wp.vec3f),          # (E, 6)
        uav_vel: wp.array2d(dtype=wp.vec3f),          # (E, 6)
        uav_battery_j: wp.array2d(dtype=wp.float32),  # (E, 6)
        uav_alive: wp.array2d(dtype=wp.int32),        # (E, 6)
        cell_has_target: wp.array2d(dtype=wp.int32),  # (E, 400): -1 if empty else target_id
        # Global team-merged DS counters (Anti-Reward-Hack Guard #2: global marginal uncertainty!)
        ds_pos_cnt: wp.array2d(dtype=wp.int32),       # (E, 400)
        ds_neg_cnt: wp.array2d(dtype=wp.int32),       # (E, 400)
        ds_m_f: wp.array2d(dtype=wp.float32),         # (E, 400)
        ds_m_u: wp.array2d(dtype=wp.float32),         # (E, 400)
        cell_reported: wp.array2d(dtype=wp.int32),    # (E, 400): 1 if already confirmed & enqueued
        # Per-UAV report queue (shared with mac_slotted_fp32.py)
        q_count: wp.array2d(dtype=wp.int32),          # (E, 6)
        q_target_id: wp.array3d(dtype=wp.int32),      # (E, 6, 32)
        q_timestamp: wp.array3d(dtype=wp.float32),    # (E, 6, 32)
        q_retries: wp.array3d(dtype=wp.int32),        # (E, 6, 32)
        q_hops: wp.array3d(dtype=wp.int32),           # (E, 6, 32)
        q_visited_mask: wp.array3d(dtype=wp.int32),   # (E, 6, 32)
        step_time_s: wp.array(dtype=wp.float32),      # (1,)
        rng_states: wp.array(dtype=wp.uint32),        # (E,)
        # Step outputs for reward assembly
        step_delta_u_global: wp.array(dtype=wp.float32),  # (E,)
        step_propulsion_kj: wp.array(dtype=wp.float32),   # (E,)
        sensing_radius_m: wp.float32,
        dt_s: wp.float32,
    ):
        """1 thread = 1 env. Updates propulsion battery, DS belief cells, and enqueues confirmed reports."""
        e = wp.tid()
        now = step_time_s[0]
        rng = rng_states[e]

        delta_u_sum = wp.float32(0.0)
        prop_j_sum = wp.float32(0.0)
        r_s2 = sensing_radius_m * sensing_radius_m

        for u in range(wp.static(NUM_UAVS)):
            if uav_alive[e, u] == wp.int32(0):
                continue
            vel = uav_vel[e, u]
            speed = wp.sqrt(vel[0] * vel[0] + vel[1] * vel[1])
            p_w = zeng_power_fp32(speed)
            e_j = p_w * dt_s
            rem = wp.max(wp.float32(0.0), uav_battery_j[e, u] - e_j)
            uav_battery_j[e, u] = rem
            if rem <= wp.float32(0.0):
                uav_alive[e, u] = wp.int32(0)
            prop_j_sum = prop_j_sum + e_j

            # Sense grid cells within local bounding box of sensing_radius_m
            px = uav_pos[e, u][0]
            py = uav_pos[e, u][1]
            cx_min = wp.max(wp.int32(0), wp.int32((px - sensing_radius_m) / wp.float32(CELL_SIZE_M)))
            cx_max = wp.min(wp.int32(GRID_W - 1), wp.int32((px + sensing_radius_m) / wp.float32(CELL_SIZE_M)))
            cy_min = wp.max(wp.int32(0), wp.int32((py - sensing_radius_m) / wp.float32(CELL_SIZE_M)))
            cy_max = wp.min(wp.int32(GRID_H - 1), wp.int32((py + sensing_radius_m) / wp.float32(CELL_SIZE_M)))

            for cy in range(cy_min, cy_max + wp.int32(1)):
                cell_y = (wp.float32(cy) + wp.float32(0.5)) * wp.float32(CELL_SIZE_M)
                dy = cell_y - py
                for cx in range(cx_min, cx_max + wp.int32(1)):
                    cell_x = (wp.float32(cx) + wp.float32(0.5)) * wp.float32(CELL_SIZE_M)
                    dx = cell_x - px
                    if dx * dx + dy * dy <= r_s2:
                        cid = cy * wp.int32(GRID_W) + cx
                        tid = cell_has_target[e, cid]
                        # Stochastic sensor reading (Pd=0.82, Pf=0.08 at cruise z=50m)
                        draw = wp.randf(rng)
                        detected = (draw < wp.float32(0.82)) if tid >= wp.int32(0) else (draw < wp.float32(0.08))
                        if detected:
                            ds_pos_cnt[e, cid] = ds_pos_cnt[e, cid] + wp.int32(1)
                        else:
                            ds_neg_cnt[e, cid] = ds_neg_cnt[e, cid] + wp.int32(1)

                        old_u = ds_m_u[e, cid]
                        belief = ds_update_cell_fp32(
                            ds_pos_cnt[e, cid], ds_neg_cnt[e, cid], wp.float32(0.55), wp.float32(0.55)
                        )
                        ds_m_f[e, cid] = belief[0]
                        ds_m_u[e, cid] = belief[2]
                        if old_u > belief[2]:
                            delta_u_sum = delta_u_sum + (old_u - belief[2])

                        # Confirm target cell & enqueue report with AoI timestamp t_s
                        if (
                            cell_reported[e, cid] == wp.int32(0)
                            and belief[0] >= wp.float32(0.90)
                            and belief[2] <= wp.float32(0.15)
                        ):
                            cell_reported[e, cid] = wp.int32(1)
                            cnt = q_count[e, u]
                            if cnt < wp.int32(32) and tid >= wp.int32(0):
                                q_target_id[e, u, cnt] = tid
                                q_timestamp[e, u, cnt] = now
                                q_retries[e, u, cnt] = wp.int32(0)
                                q_hops[e, u, cnt] = wp.int32(0)
                                q_visited_mask[e, u, cnt] = wp.int32(1) << u
                                q_count[e, u] = cnt + wp.int32(1)

        step_delta_u_global[e] = delta_u_sum
        step_propulsion_kj[e] = prop_j_sum * wp.float32(0.001)
        rng_states[e] = rng


class WarpSlottedUavEnvGraph:
    """Reference CUDA Graph capture wrapper for zero-copy PyTorch <-> NVIDIA Warp stepping on T4.

    Usage on Kaggle Tesla T4:
        1. Allocate all static PyTorch CUDA tensors once in __init__ and wrap via wp.from_torch().
        2. Run 2 warm-up steps on a non-default torch.cuda.Stream() so Warp JIT-compiles sm_75 kernels.
        3. Capture the full step sequence (motion -> phy_sinr_fp32 -> ds_sense_and_energy ->
           mac_slotted_step_fp32 -> reward -> auto_reset) inside wp.ScopedCapture().
        4. At runtime: static_actions_torch.copy_(actions); wp.capture_launch(graph) with zero CPU sync.
    """

    def __init__(self, step_fn, torch_stream) -> None:
        if wp is None:
            raise RuntimeError("nvidia-warp is required to capture CUDA graphs.")
        self._step_fn = step_fn
        self._torch_stream = torch_stream
        self._wp_stream = wp.stream_from_torch(torch_stream)
        self._graph = None

    def capture(self) -> None:
        with wp.ScopedStream(self._wp_stream):
            # Mandatory warm-up before capture to finish JIT compilation and pool allocation
            self._step_fn()
            self._step_fn()
            with wp.ScopedCapture(stream=self._wp_stream) as cap:
                self._step_fn()
            self._graph = cap.graph

    def launch(self) -> None:
        if self._graph is None:
            raise RuntimeError("Call capture() before launch().")
        with wp.ScopedStream(self._wp_stream):
            wp.capture_launch(self._graph)
