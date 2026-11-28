"""V96 finite-horizon simultaneous-braking safety helpers (register-resident).

The three-step simulated backup is a *conditional* candidate certificate under the
model and input bounds. It does not establish global forward invariance when no
certified candidate can be found; the physical post-detector remains fail-closed.
See backup-set control literature for stronger inductive claims and assumptions.
"""
from __future__ import annotations

from zhang_xue.gpu.kernels.mission_v79_kernels import N_UAV, _segment_dist2_v79
from zhang_xue.gpu.state_v67 import HAS_WARP, wp

if HAS_WARP:
    wp.set_module_options({"enable_backward": False, "max_unroll": 2})
    mat62f = wp.types.matrix(shape=(6, 2), dtype=wp.float32)

    @wp.func
    def _clamp_vel_v96(
        vpx: float, vpy: float, vcx: float, vcy: float,
        dt_s: float, v_max: float, a_max: float,
    ) -> wp.vec2:
        dvx = vcx - vpx
        dvy = vcy - vpy
        dn = wp.sqrt(dvx * dvx + dvy * dvy)
        mdv = a_max * dt_s
        if dn > mdv and dn > 1.0e-12:
            sc = mdv / dn
            dvx = dvx * sc
            dvy = dvy * sc
        vx = vpx + dvx
        vy = vpy + dvy
        vn = wp.sqrt(vx * vx + vy * vy)
        if vn > v_max and vn > 1.0e-12:
            sv = v_max / vn
            vx = vx * sv
            vy = vy * sv
        return wp.vec2(vx, vy)

    @wp.func
    def _pair_min_sep2_pts_v96(
        px: float, py: float, vx: float, vy: float,
        qx: float, qy: float, ux: float, uy: float, dt_s: float,
    ) -> float:
        rx = px - qx
        ry = py - qy
        dx = (vx - ux) * dt_s
        dy = (vy - uy) * dt_s
        den = wp.max(1.0e-12, dx * dx + dy * dy)
        tau = wp.clamp(-(rx * dx + ry * dy) / den, 0.0, 1.0)
        mx = rx + tau * dx
        my = ry + tau * dy
        return mx * mx + my * my

    @wp.func
    def _vc_max_disc_v96(s: float, a_rel: float, dt_s: float) -> float:
        if s <= 0.0 or dt_s <= 0.0:
            return 0.0
        sdv = wp.max(1.0e-6, a_rel * dt_s)
        d1 = sdv * dt_s
        if s <= d1:
            return s / dt_s
        if s <= 3.0 * d1:
            return (s / dt_s + sdv) * 0.5
        if s <= 6.0 * d1:
            return (s / dt_s + 3.0 * sdv) / 3.0
        return (s / dt_s + 6.0 * sdv) * 0.25

    @wp.func
    def _rollout_safe_v96(
        e: int, v_try: mat62f, n_steps: int,
        pos: wp.array3d(dtype=wp.float32), battery: wp.array2d(dtype=wp.float32),
        b: wp.array2d(dtype=wp.float32), n_b: int,
        dt_s: float, v_max: float, a_max: float, safe_d: float,
    ) -> int:
        """Swept-safe multi-step simultaneous-braking certificate + step-0 bit32 check."""
        p_sim = mat62f()
        v_sim = mat62f()
        for u in range(N_UAV):
            p_sim[u, 0] = pos[e, u, 0]
            p_sim[u, 1] = pos[e, u, 1]
            if battery[e, u] > 0.0:
                v_sim[u, 0] = v_try[u, 0]
                v_sim[u, 1] = v_try[u, 1]
            else:
                v_sim[u, 0] = 0.0
                v_sim[u, 1] = 0.0
        for i in range(N_UAV):
            for j in range(i + 1, N_UAV):
                if battery[e, i] <= 0.0 and battery[e, j] <= 0.0:
                    continue
                ax_b = (0.0 if battery[e, i] <= 0.0 else a_max) + (0.0 if battery[e, j] <= 0.0 else a_max)
                fx = (p_sim[i, 0] - p_sim[j, 0]) + 2.0 * dt_s * (v_sim[i, 0] - v_sim[j, 0])
                fy = (p_sim[i, 1] - p_sim[j, 1]) + 2.0 * dt_s * (v_sim[i, 1] - v_sim[j, 1])
                if wp.sqrt(fx * fx + fy * fy) + ax_b * dt_s * dt_s + 1.0e-3 < safe_d:
                    return 0
        thr_p = safe_d + 1.0e-3
        thr_p2 = thr_p * thr_p
        for _ in range(n_steps):
            for u in range(N_UAV):
                if battery[e, u] <= 0.0:
                    continue
                x0 = p_sim[u, 0]
                y0 = p_sim[u, 1]
                x1 = x0 + v_sim[u, 0] * dt_s
                y1 = y0 + v_sim[u, 1] * dt_s
                if x1 < 10.001 or x1 > 989.999 or y1 < 10.001 or y1 > 989.999:
                    return 0
                for k in range(n_b):
                    rad = b[k, 2] + 2.001
                    if _segment_dist2_v79(x0, y0, x1, y1, b[k, 0], b[k, 1]) < rad * rad:
                        return 0
            for i in range(N_UAV):
                for j in range(i + 1, N_UAV):
                    if battery[e, i] <= 0.0 and battery[e, j] <= 0.0:
                        continue
                    if _pair_min_sep2_pts_v96(
                        p_sim[i, 0], p_sim[i, 1], v_sim[i, 0], v_sim[i, 1],
                        p_sim[j, 0], p_sim[j, 1], v_sim[j, 0], v_sim[j, 1], dt_s,
                    ) < thr_p2:
                        return 0
            for u in range(N_UAV):
                p_sim[u, 0] = p_sim[u, 0] + v_sim[u, 0] * dt_s
                p_sim[u, 1] = p_sim[u, 1] + v_sim[u, 1] * dt_s
                if battery[e, u] > 0.0:
                    vb = _clamp_vel_v96(v_sim[u, 0], v_sim[u, 1], 0.0, 0.0, dt_s, v_max, a_max)
                    v_sim[u, 0] = vb[0]
                    v_sim[u, 1] = vb[1]
        return 1

    @wp.func
    def _pass3_pocs_v96(
        e: int, v_init: mat62f,
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32), n_b: int,
        dt_s: float, v_max: float, a_max: float, safe_d: float,
    ) -> mat62f:
        """24-iteration coupled half-plane POCS on separate register matrix v_rec."""
        v_rec = v_init
        for _ in range(24):
            for i in range(N_UAV):
                for j in range(i + 1, N_UAV):
                    wi = 0.0 if battery[e, i] <= 0.0 else 1.0
                    wj = 0.0 if battery[e, j] <= 0.0 else 1.0
                    wsum = wi + wj
                    if wsum <= 0.0:
                        continue
                    rx = pos[e, i, 0] - pos[e, j, 0]
                    ry = pos[e, i, 1] - pos[e, j, 1]
                    d = wp.sqrt(rx * rx + ry * ry)
                    if d <= 1.0e-6:
                        continue
                    nx = rx / d
                    ny = ry / d
                    vc_lim = _vc_max_disc_v96(d - (safe_d + 0.02), 0.5 * wsum * a_max, dt_s)
                    viol = -vc_lim - (nx * (v_rec[i, 0] - v_rec[j, 0]) + ny * (v_rec[i, 1] - v_rec[j, 1]))
                    if viol > 0.0:
                        inv_w = 1.0 / wsum
                        if wi > 0.0:
                            v_rec[i, 0] = v_rec[i, 0] + (wi * inv_w) * viol * nx
                            v_rec[i, 1] = v_rec[i, 1] + (wi * inv_w) * viol * ny
                        if wj > 0.0:
                            v_rec[j, 0] = v_rec[j, 0] - (wj * inv_w) * viol * nx
                            v_rec[j, 1] = v_rec[j, 1] - (wj * inv_w) * viol * ny
            for u in range(N_UAV):
                if battery[e, u] <= 0.0:
                    continue
                px = pos[e, u, 0]
                py = pos[e, u, 1]
                vx = wp.clamp(v_rec[u, 0], -_vc_max_disc_v96(px - 10.01, a_max, dt_s), _vc_max_disc_v96(989.99 - px, a_max, dt_s))
                vy = wp.clamp(v_rec[u, 1], -_vc_max_disc_v96(py - 10.01, a_max, dt_s), _vc_max_disc_v96(989.99 - py, a_max, dt_s))
                for k in range(n_b):
                    rbx = px - b[k, 0]
                    rby = py - b[k, 1]
                    db = wp.sqrt(rbx * rbx + rby * rby)
                    if db > 1.0e-6:
                        nbx = rbx / db
                        nby = rby / db
                        vbl = _vc_max_disc_v96(db - (b[k, 2] + 2.01), a_max, dt_s)
                        vbv = -vbl - (nbx * vx + nby * vy)
                        if vbv > 0.0:
                            vx = vx + vbv * nbx
                            vy = vy + vbv * nby
                vc = _clamp_vel_v96(vel[e, u, 0], vel[e, u, 1], vx, vy, dt_s, v_max, a_max)
                v_rec[u, 0] = vc[0]
                v_rec[u, 1] = vc[1]
        return v_rec

    @wp.func
    def _bisect_to_brk_v96(
        e: int, v_brk: mat62f, v_target: mat62f, n_steps: int,
        pos: wp.array3d(dtype=wp.float32), battery: wp.array2d(dtype=wp.float32),
        b: wp.array2d(dtype=wp.float32), n_b: int,
        dt_s: float, v_max: float, a_max: float, safe_d: float,
    ) -> mat62f:
        """8-step joint convex bisection from certified-safe v_brk toward v_target."""
        lo = float(0.0)
        hi = float(1.0)
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            v_mid = v_brk + mid * (v_target - v_brk)
            if _rollout_safe_v96(e, v_mid, n_steps, pos, battery, b, n_b, dt_s, v_max, a_max, safe_d) != 0:
                lo = mid
            else:
                hi = mid
        return v_brk + lo * (v_target - v_brk)
