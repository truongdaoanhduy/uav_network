"""Pure FP32 NVIDIA Warp Dempster-Shafer Belief Map (with Building Occlusion & Local Dec-POMDP) + Zeng Energy.

Fixes Gap B & Gap C from the 2026-11-24 audit:
1. Building LOS Occlusion (Gap B): `ds_sense_and_energy_step_fp32` now checks
   `segment_hits_cylinder_2d` against all 22 urban buildings before generating any sensor reading,
   matching CPU camera occlusion 100% (no seeing through buildings).
2. Strict Local Dec-POMDP Belief Mode (Gap C): Each UAV `u` maintains private
   `local_ds_pos[E, 6, 400]`, `local_ds_neg[E, 6, 400]`, `local_ds_m_f[E, 6, 400]`,
   `local_ds_m_u[E, 6, 400]`, `local_cell_reported[E, 6, 400]` (~93.8 MiB VRAM at E=2048, <0.6% T4).
   When `belief_sharing_mode == 1 ("local_decpomdp")`, UAV `u` confirms hypotheses and builds its
   96D Actor observation strictly from its own local belief map (zero instantaneous telepathy).
   The team-merged `ds_pos_cnt/ds_neg_cnt/ds_m_f/ds_m_u` is updated strictly for CTDE global reward
   `w_1 * Delta_U_global` and 256D Centralized Critic state.
"""

from __future__ import annotations

import math

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from uav_network.communication.gpu.phy_sinr_fp32 import NUM_UAVS
if wp is not None:
    from uav_network.communication.gpu.phy_sinr_fp32 import segment_hits_cylinder_2d

GRID_W = 20
GRID_H = 20
NUM_CELLS = GRID_W * GRID_H
CELL_SIZE_M = 40.0

# Zeng et al. (2019) 2kg-class tactical quadrotor constants (Hover 168 W > Cruise ~132 W)
ZENG_P0 = 80.0
ZENG_PI = 88.0
ZENG_U_TIP = 120.0
ZENG_V0 = 4.03
ZENG_PARASITE_COEFF = 0.5 * 0.6 * 1.225 * 0.05 * 0.503


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
    # Stable log-space evaluation: direct powers underflow after many updates.
    lf = n_pos * math.log1p(-mf)
    le = n_neg * math.log1p(-me)
    pivot = max(lf, le)
    f = math.exp(lf - pivot)
    e = math.exp(le - pivot)
    ef = math.exp(lf + le - pivot)
    denom = f + e - ef
    m_e = (f - ef) / denom
    m_f = (e - ef) / denom
    m_u = ef / denom
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
        lf = wp.float32(n_pos) * wp.log(wp.float32(1.0) - mf)
        le = wp.float32(n_neg) * wp.log(wp.float32(1.0) - me)
        pivot = wp.max(lf, le)
        f = wp.exp(lf - pivot)
        e = wp.exp(le - pivot)
        ef = wp.exp(lf + le - pivot)
        denom = wp.max(f + e - ef, wp.float32(1.0e-30))
        m_f = (e - ef) / denom
        m_e = (f - ef) / denom
        m_u = ef / denom
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
        uav_pos: wp.array2d(dtype=wp.vec3f),             # (E, 6)
        uav_vel: wp.array2d(dtype=wp.vec3f),             # (E, 6)
        uav_battery_j: wp.array2d(dtype=wp.float32),     # (E, 6)
        uav_alive: wp.array2d(dtype=wp.int32),           # (E, 6)
        obstacles: wp.array2d(dtype=wp.vec4f),           # (E, max_obs) (cx, cy, r, h) — Gap B fix!
        num_obstacles: wp.int32,
        cell_has_target: wp.array2d(dtype=wp.int32),     # (E, 400): -1 if empty else target_id
        # Team-merged DS maps (for CTDE global marginal uncertainty reward & 256D Critic state)
        ds_pos_cnt: wp.array2d(dtype=wp.int32),          # (E, 400)
        ds_neg_cnt: wp.array2d(dtype=wp.int32),          # (E, 400)
        ds_m_f: wp.array2d(dtype=wp.float32),            # (E, 400)
        ds_m_u: wp.array2d(dtype=wp.float32),            # (E, 400)
        cell_reported: wp.array2d(dtype=wp.int32),       # (E, 400)
        # Per-UAV private local DS maps (Gap C fix: strict local_decpomdp!)
        local_ds_pos: wp.array3d(dtype=wp.int32),        # (E, 6, 400)
        local_ds_neg: wp.array3d(dtype=wp.int32),        # (E, 6, 400)
        local_ds_m_f: wp.array3d(dtype=wp.float32),      # (E, 6, 400)
        local_ds_m_u: wp.array3d(dtype=wp.float32),      # (E, 6, 400)
        local_cell_reported: wp.array3d(dtype=wp.int32), # (E, 6, 400)
        belief_sharing_mode: wp.int32,                   # 1 = local_decpomdp, 0 = shared_team_ideal
        # Per-UAV hypothesis report queue (observed cell ID, strictly NoOracle)
        q_count: wp.array2d(dtype=wp.int32),             # (E, 6)
        q_cell_id: wp.array3d(dtype=wp.int32),           # (E, 6, 32)
        q_timestamp: wp.array3d(dtype=wp.float32),       # (E, 6, 32)
        q_retries: wp.array3d(dtype=wp.int32),           # (E, 6, 32)
        q_hops: wp.array3d(dtype=wp.int32),              # (E, 6, 32)
        q_visited_mask: wp.array3d(dtype=wp.int32),      # (E, 6, 32)
        step_time_s: wp.array(dtype=wp.float32),         # (1,)
        rng_states: wp.array(dtype=wp.uint32),           # (E,)
        # Step outputs for reward assembly
        step_delta_u_global: wp.array(dtype=wp.float32), # (E,)
        step_propulsion_kj: wp.array(dtype=wp.float32),  # (E,)
        sensing_radius_m: wp.float32,
        dt_s: wp.float32,
    ):
        """1 thread = 1 env. Building-occluded DS sensing (local + team) & Zeng rotary-wing energy."""
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
                continue
            prop_j_sum = prop_j_sum + e_j

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
                    if dx * dx + dy * dy > r_s2:
                        continue

                    # Gap B fix: Check building occlusion between UAV (px, py) and ground cell (cell_x, cell_y)
                    occluded = wp.int32(0)
                    for o in range(num_obstacles):
                        ob = obstacles[e, o]
                        if ob[3] > wp.float32(0.0):
                            if segment_hits_cylinder_2d(px, py, cell_x, cell_y, ob[0], ob[1], ob[2]) == wp.int32(1):
                                occluded = wp.int32(1)
                                break
                    if occluded == wp.int32(1):
                        continue

                    cid = cy * wp.int32(GRID_W) + cx
                    tid = cell_has_target[e, cid]
                    draw = wp.randf(rng)
                    detected = (draw < wp.float32(0.82)) if tid >= wp.int32(0) else (draw < wp.float32(0.08))

                    # Update team-merged DS map (CTDE global marginal uncertainty reward + 256D Critic)
                    if detected:
                        ds_pos_cnt[e, cid] = ds_pos_cnt[e, cid] + wp.int32(1)
                        local_ds_pos[e, u, cid] = local_ds_pos[e, u, cid] + wp.int32(1)
                    else:
                        ds_neg_cnt[e, cid] = ds_neg_cnt[e, cid] + wp.int32(1)
                        local_ds_neg[e, u, cid] = local_ds_neg[e, u, cid] + wp.int32(1)

                    old_u = ds_m_u[e, cid]
                    team_bel = ds_update_cell_fp32(
                        ds_pos_cnt[e, cid], ds_neg_cnt[e, cid], wp.float32(0.55), wp.float32(0.55)
                    )
                    ds_m_f[e, cid] = team_bel[0]
                    ds_m_u[e, cid] = team_bel[2]
                    if old_u > team_bel[2]:
                        delta_u_sum = delta_u_sum + (old_u - team_bel[2])

                    # Update private local DS cache for UAV u
                    loc_bel = ds_update_cell_fp32(
                        local_ds_pos[e, u, cid], local_ds_neg[e, u, cid], wp.float32(0.55), wp.float32(0.55)
                    )
                    local_ds_m_f[e, u, cid] = loc_bel[0]
                    local_ds_m_u[e, u, cid] = loc_bel[2]

                    # Confirmation gating: strictly NoOracle, uses local belief when local_decpomdp==1
                    use_mf = loc_bel[0] if belief_sharing_mode == wp.int32(1) else team_bel[0]
                    use_mu = loc_bel[2] if belief_sharing_mode == wp.int32(1) else team_bel[2]
                    already_rep = (
                        local_cell_reported[e, u, cid]
                        if belief_sharing_mode == wp.int32(1)
                        else cell_reported[e, cid]
                    )

                    if detected and already_rep == wp.int32(0) and use_mf >= wp.float32(0.90) and use_mu <= wp.float32(0.15):
                        cnt = q_count[e, u]
                        if cnt < wp.int32(32):
                            local_cell_reported[e, u, cid] = wp.int32(1)
                            cell_reported[e, cid] = wp.int32(1)
                            q_cell_id[e, u, cnt] = cid
                            q_timestamp[e, u, cnt] = now
                            q_retries[e, u, cnt] = wp.int32(0)
                            q_hops[e, u, cnt] = wp.int32(0)
                            q_visited_mask[e, u, cnt] = wp.int32(1) << u
                            q_count[e, u] = cnt + wp.int32(1)

        step_delta_u_global[e] = delta_u_sum
        step_propulsion_kj[e] = prop_j_sum * wp.float32(0.001)
        rng_states[e] = rng


class WarpSlottedUavEnvGraph:
    """CUDA Graph capture wrapper for zero-copy PyTorch <-> NVIDIA Warp stepping on T4."""

    def __init__(self, step_fn, torch_stream) -> None:
        if wp is None:
            raise RuntimeError("nvidia-warp is required to capture CUDA graphs.")
        self._step_fn = step_fn
        self._torch_stream = torch_stream
        self._wp_stream = wp.stream_from_torch(torch_stream)
        self._graph = None

    def capture(self) -> None:
        # Capture and warmup run on a dedicated Warp/Torch CUDA stream.
        # An explicit event dependency is required before subsequent Torch
        # reset() operations on its ordinary/current stream.
        import torch
        current = torch.cuda.current_stream(device=self._torch_stream.device)
        self._torch_stream.wait_stream(current)
        with wp.ScopedStream(self._wp_stream):
            self._step_fn()
            self._step_fn()
            with wp.ScopedCapture(stream=self._wp_stream) as cap:
                self._step_fn()
            self._graph = cap.graph
        current.wait_stream(self._torch_stream)

    def launch(self) -> None:
        if self._graph is None:
            raise RuntimeError("Call capture() before launch().")
        import torch
        current = torch.cuda.current_stream(device=self._torch_stream.device)
        # Input actions are written by PyTorch on `current`. This prevents
        # reading stale actions. The reverse edge ensures subsequent Torch
        # replay/logging consumes completed reward, observations and dones.
        self._torch_stream.wait_stream(current)
        with wp.ScopedStream(self._wp_stream):
            wp.capture_launch(self._graph)
        current.wait_stream(self._torch_stream)
