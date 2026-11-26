"""V80 Warp GPU Feasible Kinematic Safety Shield (66-candidate disk projection + 3-sweep CD).

Resolves V79 Vast.ai early safety deaths (63 MATD3 / 117 MASAC in 300 steps) WITHOUT relaxing
`safe_d=50.0 m`, `a_max=5.0 m/s^2`, or `safety_failure_terminate=True`:
- Replaces 1D straight-line braking fallback with zero-allocation 66-candidate acceleration-disk
  projection (exact online argmin equivalent to CPU `kinematic_safety_shield._project_uav`).
- Replaces single-pass `cand[peer]` check with Pass 1 (geometry) + Pass 2 (up to 3 coordinate-
  descent sweeps using `v_safe` only), eliminating peer-order domino cascades.
- Aligns Phase 2 termination detector with CPU `trajectory_safety_violations` (actual swept
  physical collision: boundary=1, building=2, uav_pair=4) while recording braking-margin
  tightness separately on bit 8 (`braking_margin_warning`), with optional `strict_braking_terminate`
  flag for exact A/B ablation against V79.
- Replaces hardcoded `_candidate_v78` (`15.0`/`5.0`) with dynamic `_candidate_v80(dt_s, v_max, a_max)`.
"""
from __future__ import annotations

from zhang_xue.gpu.kernels.mission_v79_kernels import (
    N_UAV,
    _pair_safe_v79,
    _safe_xy_v79,
    _segment_dist2_v79,
)
from zhang_xue.gpu.state_v67 import HAS_WARP, NetworkBuffersV67, wp

if HAS_WARP:
    # Non-differentiable environment transition, not a policy gradient layer.
    # Avoid unused backward codegen and large 4x16 nested loop unrolling.
    wp.set_module_options({"enable_backward": False, "max_unroll": 2})

    @wp.func
    def _clamp_vel_v80(
        vx_prev: float, vy_prev: float, vx_cmd: float, vy_cmd: float,
        dt_s: float, v_max: float, a_max: float,
    ) -> wp.vec2:
        dvx = vx_cmd - vx_prev
        dvy = vy_cmd - vy_prev
        dv_norm = wp.sqrt(dvx * dvx + dvy * dvy)
        max_dv = a_max * dt_s
        if dv_norm > max_dv and dv_norm > 1.0e-12:
            scale = max_dv / dv_norm
            dvx = dvx * scale
            dvy = dvy * scale
        vx = vx_prev + dvx
        vy = vy_prev + dvy
        v_norm = wp.sqrt(vx * vx + vy * vy)
        if v_norm > v_max and v_norm > 1.0e-12:
            scale_v = v_max / v_norm
            vx = vx * scale_v
            vy = vy * scale_v
        return wp.vec2(vx, vy)

    @wp.kernel
    def _candidate_v80(
        act: wp.array3d(dtype=wp.float32),
        vel: wp.array3d(dtype=wp.float32),
        cand: wp.array3d(dtype=wp.float32),
        dt_s: float, v_max: float, a_max: float,
    ):
        e, u = wp.tid()
        cx = wp.clamp(act[e, u, 0], -1.0, 1.0) * v_max
        cy = wp.clamp(act[e, u, 1], -1.0, 1.0) * v_max
        # Match CPU clamp_velocity_with_accel: cap acceleration FIRST, then speed.
        # Pre-normalizing an over-speed command here changes direction/acceleration.
        v_nom = _clamp_vel_v80(vel[e, u, 0], vel[e, u, 1], cx, cy, dt_s, v_max, a_max)
        cand[e, u, 0] = v_nom[0]
        cand[e, u, 1] = v_nom[1]

    @wp.func
    def _geom_physical_flags_v80(
        x: float, y: float, vx: float, vy: float,
        b: wp.array2d(dtype=wp.float32), n_b: int, dt_s: float,
    ) -> int:
        """Exact swept physical geometry detector (1=boundary, 2=building)."""
        nx = x + vx * dt_s
        ny = y + vy * dt_s
        flags = int(0)
        if (x < 10.0 or x > 990.0 or y < 10.0 or y > 990.0
                or nx < 10.0 or nx > 990.0 or ny < 10.0 or ny > 990.0):
            flags = flags | 1
        for k in range(n_b):
            radius = b[k, 2] + 2.0 - 1.0e-6
            if _segment_dist2_v79(x, y, nx, ny, b[k, 0], b[k, 1]) < radius * radius:
                flags = flags | 2
                break
        return flags

    @wp.func
    def _pair_min_sep2_v80(
        px: float, py: float, vx: float, vy: float,
        qx: float, qy: float, ux: float, uy: float, dt_s: float,
    ) -> float:
        rx = px - qx
        ry = py - qy
        dx = (vx - ux) * dt_s
        dy = (vy - uy) * dt_s
        denom = wp.max(1.0e-12, dx * dx + dy * dy)
        tau = wp.clamp(-(rx * dx + ry * dy) / denom, 0.0, 1.0)
        mx = rx + tau * dx
        my = ry + tau * dy
        return mx * mx + my * my

    @wp.func
    def _pair_physical_safe_v80(
        px: float, py: float, vx: float, vy: float,
        qx: float, qy: float, ux: float, uy: float,
        safe_d: float, dt_s: float,
    ) -> int:
        """Exact swept-segment pairwise separation detector (CPU trajectory_safety_violations)."""
        thr = safe_d - 1.0e-6
        return 0 if (_pair_min_sep2_v80(px, py, vx, vy, qx, qy, ux, uy, dt_s) < thr * thr) else 1

    @wp.func
    def _min_peer_sep2_v80(
        e: int, u: int, vx: float, vy: float,
        pos: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), dt_s: float,
    ) -> float:
        px = pos[e, u, 0]
        py = pos[e, u, 1]
        m2 = float(1.0e30)
        for j in range(N_UAV):
            if j == u:
                continue
            ux = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 0]
            uy = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 1]
            d2 = _pair_min_sep2_v80(px, py, vx, vy, pos[e, j, 0], pos[e, j, 1], ux, uy, dt_s)
            if d2 < m2:
                m2 = d2
        return m2

    @wp.func
    def _cand_phys_peers_v90(
        e: int, u: int, vx: float, vy: float,
        pos: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), safe_d: float, dt_s: float,
    ) -> int:
        px = pos[e, u, 0]
        py = pos[e, u, 1]
        for j in range(N_UAV):
            if j == u:
                continue
            ux = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 0]
            uy = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 1]
            if _pair_physical_safe_v80(px, py, vx, vy, pos[e, j, 0], pos[e, j, 1], ux, uy, safe_d, dt_s) == 0:
                return 0
        return 1

    @wp.func
    def _cand_feasible_v80(
        e: int, u: int, vx: float, vy: float, check_peers: int,
        pos: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), n_b: int,
        safe_d: float, a_max: float,
    ) -> int:
        px = pos[e, u, 0]
        py = pos[e, u, 1]
        if _safe_xy_v79(px, py, vx, vy, b, n_b, a_max) == 0:
            return 0
        if check_peers != 0:
            for j in range(N_UAV):
                if j == u:
                    continue
                ux = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 0]
                uy = 0.0 if battery[e, j] <= 0.0 else v_safe[e, j, 1]
                if _pair_safe_v79(px, py, vx, vy, pos[e, j, 0], pos[e, j, 1], ux, uy, safe_d, a_max) == 0:
                    return 0
        return 1

    @wp.func
    def _project_uav_v80(
        e: int, u: int, check_peers: int,
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        cand: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), n_b: int,
        dt_s: float, v_max: float, a_max: float, safe_d: float,
    ) -> wp.vec2:
        """Zero-allocation 66-candidate acceleration-disk projection (CPU shield parity)."""
        vpx = vel[e, u, 0]
        vpy = vel[e, u, 1]
        vnx = cand[e, u, 0]
        vny = cand[e, u, 1]
        v_brk = _clamp_vel_v80(vpx, vpy, 0.0, 0.0, dt_s, v_max, a_max)
        best_cost = float(1.0e30)
        best_vx = v_brk[0]
        best_vy = v_brk[1]
        pfb_cost = float(1.0e30)
        pfb_vx = v_brk[0]
        pfb_vy = v_brk[1]
        gfb_cost = float(1.0e30)
        gfb_vx = v_brk[0]
        gfb_vy = v_brk[1]
        px = pos[e, u, 0]
        py = pos[e, u, 1]
        # Candidate 0: current v_safe[e, u]
        c0x = v_safe[e, u, 0]
        c0y = v_safe[e, u, 1]
        if check_peers != 0 and _safe_xy_v79(px, py, c0x, c0y, b, n_b, a_max) != 0:
            gbx = c0x - v_brk[0]
            gby = c0y - v_brk[1]
            gfb_cost = gbx * gbx + gby * gby
            gfb_vx = c0x
            gfb_vy = c0y
            if _cand_phys_peers_v90(e, u, c0x, c0y, pos, v_safe, battery, safe_d, dt_s) != 0:
                pfb_cost = gfb_cost
                pfb_vx = c0x
                pfb_vy = c0y
        if _cand_feasible_v80(e, u, c0x, c0y, check_peers, pos, v_safe, battery, b, n_b, safe_d, a_max) != 0:
            dx = c0x - vnx
            dy = c0y - vny
            best_cost = dx * dx + dy * dy
            best_vx = c0x
            best_vy = c0y
        # Candidate 1: maximum kinematic braking
        if check_peers != 0 and _safe_xy_v79(px, py, v_brk[0], v_brk[1], b, n_b, a_max) != 0:
            gfb_cost = 0.0
            gfb_vx = v_brk[0]
            gfb_vy = v_brk[1]
            if _cand_phys_peers_v90(e, u, v_brk[0], v_brk[1], pos, v_safe, battery, safe_d, dt_s) != 0:
                pfb_cost = 0.0
                pfb_vx = v_brk[0]
                pfb_vy = v_brk[1]
        if _cand_feasible_v80(e, u, v_brk[0], v_brk[1], check_peers, pos, v_safe, battery, b, n_b, safe_d, a_max) != 0:
            dx = v_brk[0] - vnx
            dy = v_brk[1] - vny
            cost = dx * dx + dy * dy
            if cost < best_cost:
                best_cost = cost
                best_vx = v_brk[0]
                best_vy = v_brk[1]
        # Candidates 2..65: 4 radii x 16 angles inside kinematic acceleration disk
        max_dv = a_max * dt_s
        for ri in range(4):
            r = max_dv * (1.0 - 0.25 * float(ri))
            for ai in range(16):
                ang = float(ai) * (6.283185307179586 / 16.0)
                vc = _clamp_vel_v80(vpx, vpy, vpx + r * wp.cos(ang), vpy + r * wp.sin(ang), dt_s, v_max, a_max)
                if check_peers != 0 and _safe_xy_v79(px, py, vc[0], vc[1], b, n_b, a_max) != 0:
                    gbx = vc[0] - v_brk[0]
                    gby = vc[1] - v_brk[1]
                    gc = gbx * gbx + gby * gby
                    if gc < gfb_cost:
                        gfb_cost = gc
                        gfb_vx = vc[0]
                        gfb_vy = vc[1]
                    if gc < pfb_cost and _cand_phys_peers_v90(e, u, vc[0], vc[1], pos, v_safe, battery, safe_d, dt_s) != 0:
                        pfb_cost = gc
                        pfb_vx = vc[0]
                        pfb_vy = vc[1]
                if _cand_feasible_v80(e, u, vc[0], vc[1], check_peers, pos, v_safe, battery, b, n_b, safe_d, a_max) != 0:
                    dx = vc[0] - vnx
                    dy = vc[1] - vny
                    cost = dx * dx + dy * dy
                    if cost < best_cost:
                        best_cost = cost
                        best_vx = vc[0]
                        best_vy = vc[1]
        if best_cost >= 1.0e29 and check_peers != 0:
            if pfb_cost < 1.0e29:
                return wp.vec2(pfb_vx, pfb_vy)
            if gfb_cost < 1.0e29:
                return wp.vec2(gfb_vx, gfb_vy)
        return wp.vec2(best_vx, best_vy)

    @wp.kernel
    def _move_v80(
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        cand: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        pos_xy: wp.array3d(dtype=wp.float32), battery: wp.array2d(dtype=wp.float32),
        b: wp.array2d(dtype=wp.float32), n_b: int, net: NetworkBuffersV67,
        failed: wp.array(dtype=wp.uint8), safety_reason: wp.array(dtype=wp.uint8),
        unsafe_mask: wp.array2d(dtype=wp.uint8), intervention: wp.array(dtype=wp.float32),
        dt_s: float, v_max: float, a_max: float, safe_d: float, pa_dc_w: float,
        strict_braking_terminate: int,
    ):
        """2-pass 66-candidate feasible shield + 3-sweep coordinate descent + physical detector."""
        e = wp.tid()
        # Init v_safe from nominal candidates; dead UAVs stay stationary obstacles
        for u in range(N_UAV):
            unsafe_mask[e, u] = wp.uint8(0)
            if battery[e, u] <= 0.0:
                v_safe[e, u, 0] = 0.0
                v_safe[e, u, 1] = 0.0
                net.radio_alive[e, u] = wp.uint8(0)
            else:
                v_safe[e, u, 0] = cand[e, u, 0]
                v_safe[e, u, 1] = cand[e, u, 1]
        # Pass 1: independent geometry safety projection
        for u in range(N_UAV):
            if battery[e, u] > 0.0 and _safe_xy_v79(pos[e, u, 0], pos[e, u, 1], v_safe[e, u, 0], v_safe[e, u, 1], b, n_b, a_max) == 0:
                vp = _project_uav_v80(e, u, 0, pos, vel, cand, v_safe, battery, b, n_b, dt_s, v_max, a_max, safe_d)
                v_safe[e, u, 0] = vp[0]
                v_safe[e, u, 1] = vp[1]
        # Pass 2: up to 3 coordinate-descent sweeps using v_safe only (eliminates peer cascade)
        for _ in range(3):
            any_conflict = int(0)
            for i in range(N_UAV):
                for j in range(i + 1, N_UAV):
                    if _pair_safe_v79(pos[e, i, 0], pos[e, i, 1], v_safe[e, i, 0], v_safe[e, i, 1],
                                      pos[e, j, 0], pos[e, j, 1], v_safe[e, j, 0], v_safe[e, j, 1], safe_d, a_max) == 0:
                        any_conflict = 1
                        if battery[e, i] > 0.0:
                            vi = _project_uav_v80(e, i, 1, pos, vel, cand, v_safe, battery, b, n_b, dt_s, v_max, a_max, safe_d)
                            v_safe[e, i, 0] = vi[0]
                            v_safe[e, i, 1] = vi[1]
                        if (battery[e, j] > 0.0 and _pair_safe_v79(
                                pos[e, i, 0], pos[e, i, 1], v_safe[e, i, 0], v_safe[e, i, 1],
                                pos[e, j, 0], pos[e, j, 1], v_safe[e, j, 0], v_safe[e, j, 1], safe_d, a_max) == 0):
                            vj = _project_uav_v80(e, j, 1, pos, vel, cand, v_safe, battery, b, n_b, dt_s, v_max, a_max, safe_d)
                            v_safe[e, j, 0] = vj[0]
                            v_safe[e, j, 1] = vj[1]
            if any_conflict == 0:
                break
        # Phase 2: post-shield safety detector (bit1=boundary, bit2=building, bit4=pair, bit8=braking_warn)
        s_flags = int(0)
        phys_fail = int(0)
        for u in range(N_UAV):
            if battery[e, u] <= 0.0:
                continue
            gf = _geom_physical_flags_v80(pos[e, u, 0], pos[e, u, 1], v_safe[e, u, 0], v_safe[e, u, 1], b, n_b, dt_s)
            if gf != 0:
                s_flags = s_flags | gf
                phys_fail = 1
                unsafe_mask[e, u] = wp.uint8(1)
            elif _safe_xy_v79(pos[e, u, 0], pos[e, u, 1], v_safe[e, u, 0], v_safe[e, u, 1], b, n_b, a_max) == 0:
                s_flags = s_flags | 8
                if strict_braking_terminate != 0:
                    phys_fail = 1
                    unsafe_mask[e, u] = wp.uint8(1)
        # V81 fix: check all (u, peer) pairs symmetrically even when u is battery-depleted
        for u in range(N_UAV):
            for peer in range(u + 1, N_UAV):
                if battery[e, u] <= 0.0 and battery[e, peer] <= 0.0:
                    continue
                vx_u = 0.0 if battery[e, u] <= 0.0 else v_safe[e, u, 0]
                vy_u = 0.0 if battery[e, u] <= 0.0 else v_safe[e, u, 1]
                ux = 0.0 if battery[e, peer] <= 0.0 else v_safe[e, peer, 0]
                uy = 0.0 if battery[e, peer] <= 0.0 else v_safe[e, peer, 1]
                if _pair_physical_safe_v80(pos[e, u, 0], pos[e, u, 1], vx_u, vy_u,
                                           pos[e, peer, 0], pos[e, peer, 1], ux, uy, safe_d, dt_s) == 0:
                    s_flags = s_flags | 4
                    phys_fail = 1
                    unsafe_mask[e, u] = wp.uint8(1)
                    unsafe_mask[e, peer] = wp.uint8(1)
                elif _pair_safe_v79(pos[e, u, 0], pos[e, u, 1], vx_u, vy_u,
                                    pos[e, peer, 0], pos[e, peer, 1], ux, uy, safe_d, a_max) == 0:
                    s_flags = s_flags | 8
                    if strict_braking_terminate != 0:
                        phys_fail = 1
                        unsafe_mask[e, u] = wp.uint8(1)
                        unsafe_mask[e, peer] = wp.uint8(1)
        safety_reason[e] = wp.uint8(s_flags)
        if phys_fail != 0:
            failed[e] = wp.uint8(1)
            # Fail-closed transaction: an unsafe candidate MUST NOT be written
            # to the physics state. Terminate through _reward_v79 and let the
            # caller reset the world. Preserve pre-step velocities as well
            # (instant zeroing would itself break the acceleration contract).
            intervention[e] = 0.0
            return
        # Phase 3: commit positions, velocities, continuous intervention norm, and Zeng 2019 energy
        interventions = float(0.0)
        for u in range(N_UAV):
            if battery[e, u] <= 0.0:
                vel[e, u, 0] = 0.0
                vel[e, u, 1] = 0.0
                continue
            vx = v_safe[e, u, 0]
            vy = v_safe[e, u, 1]
            dvx = cand[e, u, 0] - vx
            dvy = cand[e, u, 1] - vy
            interventions += wp.sqrt(dvx * dvx + dvy * dvy) / wp.max(1.0e-6, v_max)
            vel[e, u, 0] = vx
            vel[e, u, 1] = vy
            pos[e, u, 0] = pos[e, u, 0] + vx * dt_s
            pos[e, u, 1] = pos[e, u, 1] + vy * dt_s
            pos_xy[e, u, 0] = pos[e, u, 0]
            pos_xy[e, u, 1] = pos[e, u, 1]
            v = wp.sqrt(vx * vx + vy * vy)
            x = v * v / (2.0 * 4.03 * 4.03)
            p_blade = 79.86 * (1.0 + 3.0 * v * v / 14400.0)
            p_ind = 88.63 * wp.sqrt(1.0 / (wp.sqrt(1.0 + x * x) + x))
            p_par = 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * v * v * v
            air_s = wp.float32(net.tx_airtime[e, u]) / 1.0e6
            battery[e, u] = wp.max(0.0, battery[e, u] - (p_blade + p_ind + p_par + 15.0) * dt_s - pa_dc_w * air_s)
            if battery[e, u] <= 0.0:
                net.radio_alive[e, u] = wp.uint8(0)
        intervention[e] = interventions
