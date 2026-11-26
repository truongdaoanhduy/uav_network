"""V79 Warp GPU physics, 3D-occluded sensing, DS-AoI fusion, and reward kernels."""
from __future__ import annotations

from zhang_xue.gpu.state_v67 import HAS_WARP, NetworkBuffersV67, wp
from zhang_xue.gpu.kernels.mission_v91_bayesian import _bayes_from_counts_v91

N_UAV = 6
GRID = 20
NUM_CELLS = 400
N_TARGETS = 15

if HAS_WARP:
    @wp.func
    def _neg_expm1_v79(x: float) -> float:
        """Stable -(exp(x) - 1) for x <= 0 without FP32 cancellation near 0."""
        if wp.abs(x) < 1.0e-3:
            return -(x + 0.5 * x * x + (1.0 / 6.0) * x * x * x)
        return 1.0 - wp.exp(x)

    @wp.func
    def _ds_v79(p: float, n: float, logmf: float, logme: float) -> wp.vec3:
        if p + n <= 0.0:
            return wp.vec3(0.0, 0.0, 1.0)
        a = p * logmf
        b = n * logme
        shift = wp.max(a, b)
        pa = wp.exp(a - shift)
        pb = wp.exp(b - shift)
        pu = wp.exp(wp.min(a, b))
        mt = pb * _neg_expm1_v79(a) if p > 0.0 else 0.0
        me = pa * _neg_expm1_v79(b) if n > 0.0 else 0.0
        norm = wp.max(1.0e-20, mt + me + pu)
        return wp.vec3(mt / norm, me / norm, pu / norm)

    @wp.func
    def _segment_dist2_v79(x0: float, y0: float, x1: float, y1: float, cx: float, cy: float) -> float:
        dx = x1 - x0
        dy = y1 - y0
        den = wp.max(1.0e-12, dx * dx + dy * dy)
        t = wp.clamp(((cx - x0) * dx + (cy - y0) * dy) / den, 0.0, 1.0)
        ax = x0 + t * dx - cx
        ay = y0 + t * dy - cy
        return ax * ax + ay * ay

    @wp.func
    def _segment_hits_cylinder_3d_v79(
        x0: float, y0: float, z0: float, x1: float, y1: float, z1: float,
        bx: float, by: float, br: float, bh: float,
    ) -> int:
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        fx = x0 - bx
        fy = y0 - by
        a = dx * dx + dy * dy
        if a < 1.0e-9:
            if fx * fx + fy * fy <= br * br and wp.min(z0, z1) <= bh and wp.max(z0, z1) >= 0.0:
                return 1
            return 0
        bb = 2.0 * (fx * dx + fy * dy)
        cc = fx * fx + fy * fy - br * br
        disc = bb * bb - 4.0 * a * cc
        if disc < 0.0:
            return 0
        sd = wp.sqrt(disc)
        t_lo = wp.max(0.0, wp.min((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
        t_hi = wp.min(1.0, wp.max((-bb - sd) / (2.0 * a), (-bb + sd) / (2.0 * a)))
        if t_lo <= t_hi:
            z_lo = z0 + t_lo * dz
            z_hi = z0 + t_hi * dz
            if wp.min(z_lo, z_hi) <= bh and wp.max(z_lo, z_hi) >= 0.0:
                return 1
        return 0

    @wp.func
    def _safe_xy_v79(
        x: float, y: float, vx: float, vy: float,
        b: wp.array2d(dtype=wp.float32), n_b: int, a_max: float,
    ) -> int:
        nx = x + vx
        ny = y + vy
        inv2a = 0.5 / wp.max(1.0e-6, a_max)
        v_norm = wp.sqrt(vx * vx + vy * vy)
        if nx - wp.max(0.0, -vx) * v_norm * inv2a < 10.0:
            return 0
        if nx + wp.max(0.0, vx) * v_norm * inv2a > 990.0:
            return 0
        if ny - wp.max(0.0, -vy) * v_norm * inv2a < 10.0:
            return 0
        if ny + wp.max(0.0, vy) * v_norm * inv2a > 990.0:
            return 0
        for k in range(n_b):
            bx = b[k, 0]
            by = b[k, 1]
            radius = b[k, 2] + 2.0
            if _segment_dist2_v79(x, y, nx, ny, bx, by) <= radius * radius:
                return 0
            dx_b = bx - nx
            dy_b = by - ny
            dist_b = wp.sqrt(dx_b * dx_b + dy_b * dy_b)
            if dist_b > 1.0e-6:
                v_toward = wp.max(0.0, (vx * dx_b + vy * dy_b) / dist_b)
                if dist_b - radius < v_toward * v_toward * inv2a:
                    return 0
        return 1

    @wp.func
    def _pair_safe_v79(
        px: float, py: float, vx: float, vy: float,
        qx: float, qy: float, ux: float, uy: float,
        safe_d: float, a_max: float,
    ) -> int:
        rx = px - qx
        ry = py - qy
        dx = vx - ux
        dy = vy - uy
        denom = wp.max(1.0e-9, dx * dx + dy * dy)
        alpha = wp.clamp(-(rx * dx + ry * dy) / denom, 0.0, 1.0)
        mx = rx + alpha * dx
        my = ry + alpha * dy
        if mx * mx + my * my < safe_d * safe_d:
            return 0
        end_x = rx + dx
        end_y = ry + dy
        dist_end = wp.sqrt(end_x * end_x + end_y * end_y)
        if dist_end > 1.0e-6:
            v_close = wp.max(0.0, -(end_x * dx + end_y * dy) / dist_end)
            if dist_end - safe_d < (v_close * v_close) / wp.max(1.0e-6, 2.0 * a_max):
                return 0
        return 1

    @wp.kernel
    def _move_v79(
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        cand: wp.array3d(dtype=wp.float32), v_safe: wp.array3d(dtype=wp.float32),
        pos_xy: wp.array3d(dtype=wp.float32), battery: wp.array2d(dtype=wp.float32),
        b: wp.array2d(dtype=wp.float32), n_b: int, net: NetworkBuffersV67,
        failed: wp.array(dtype=wp.uint8), safety_reason: wp.array(dtype=wp.uint8),
        unsafe_mask: wp.array2d(dtype=wp.uint8),
        intervention: wp.array(dtype=wp.float32), v_max: float, a_max: float,
        safe_d: float, pa_dc_w: float,
    ):
        """Two-phase collision-safe motion (no in-place pos overwrite during pair checks)."""
        e = wp.tid()
        interventions = float(0.0)
        geometry_bad = int(0)
        pair_bad = int(0)
        # Phase 1: determine v_safe for every UAV using frozen old pos[e, :]
        for u in range(N_UAV):
            unsafe_mask[e, u] = wp.uint8(0)
            if battery[e, u] <= 0.0:
                v_safe[e, u, 0] = 0.0
                v_safe[e, u, 1] = 0.0
                net.radio_alive[e, u] = wp.uint8(0)
                continue
            px = pos[e, u, 0]
            py = pos[e, u, 1]
            vx = cand[e, u, 0]
            vy = cand[e, u, 1]
            safe = _safe_xy_v79(px, py, vx, vy, b, n_b, a_max)
            for peer in range(N_UAV):
                if peer == u:
                    continue
                ux = 0.0 if battery[e, peer] <= 0.0 else (v_safe[e, peer, 0] if peer < u else cand[e, peer, 0])
                uy = 0.0 if battery[e, peer] <= 0.0 else (v_safe[e, peer, 1] if peer < u else cand[e, peer, 1])
                if _pair_safe_v79(px, py, vx, vy, pos[e, peer, 0], pos[e, peer, 1], ux, uy, safe_d, a_max) == 0:
                    safe = 0
            if safe == 0:
                ox = vel[e, u, 0]
                oy = vel[e, u, 1]
                vv = wp.sqrt(ox * ox + oy * oy)
                ratio = wp.max(0.0, vv - a_max) / wp.max(1.0e-10, vv)
                vx = ox * ratio
                vy = oy * ratio
                dvx = cand[e, u, 0] - vx
                dvy = cand[e, u, 1] - vy
                interventions += wp.sqrt(dvx * dvx + dvy * dvy) / wp.max(1.0e-6, v_max)
            v_safe[e, u, 0] = vx
            v_safe[e, u, 1] = vy
        # Phase 2: post-shield trajectory safety verification (buildings + pairwise)
        for u in range(N_UAV):
            if battery[e, u] <= 0.0:
                continue
            if _safe_xy_v79(pos[e, u, 0], pos[e, u, 1], v_safe[e, u, 0], v_safe[e, u, 1], b, n_b, a_max) == 0:
                failed[e] = wp.uint8(1)
                geometry_bad = 1
                unsafe_mask[e, u] = wp.uint8(1)
            for peer in range(u + 1, N_UAV):
                ux = 0.0 if battery[e, peer] <= 0.0 else v_safe[e, peer, 0]
                uy = 0.0 if battery[e, peer] <= 0.0 else v_safe[e, peer, 1]
                if _pair_safe_v79(pos[e, u, 0], pos[e, u, 1], v_safe[e, u, 0], v_safe[e, u, 1],
                                  pos[e, peer, 0], pos[e, peer, 1], ux, uy, safe_d, a_max) == 0:
                    failed[e] = wp.uint8(1)
                    pair_bad = 1
                    unsafe_mask[e, u] = wp.uint8(1)
                    unsafe_mask[e, peer] = wp.uint8(1)
        # Bitmask: 1=building/boundary unsafe, 2=UAV-to-UAV unsafe.
        safety_reason[e] = wp.uint8(geometry_bad + 2 * pair_bad)
        # Fail-closed terminal rollback: the post-shield check is NOT a formal
        # viable-set projection. Never apply a trajectory that we already flagged
        # as colliding. This emergency state rollback is deliberately NOT counted
        # as a physically valid deceleration manoeuvre (can break a_max).
        # A real safety-guaranteed controller must solve a feasible joint trajectory.
        if failed[e] != wp.uint8(0):
            for u in range(N_UAV):
                vel[e, u, 0] = 0.0
                vel[e, u, 1] = 0.0
            intervention[e] = interventions
            return
        # Phase 3: commit positions, velocities, and Zeng 2019 energy + exact PA power
        for u in range(N_UAV):
            if battery[e, u] <= 0.0:
                vel[e, u, 0] = 0.0
                vel[e, u, 1] = 0.0
                continue
            vx = v_safe[e, u, 0]
            vy = v_safe[e, u, 1]
            vel[e, u, 0] = vx
            vel[e, u, 1] = vy
            pos[e, u, 0] = pos[e, u, 0] + vx
            pos[e, u, 1] = pos[e, u, 1] + vy
            pos_xy[e, u, 0] = pos[e, u, 0]
            pos_xy[e, u, 1] = pos[e, u, 1]
            v = wp.sqrt(vx * vx + vy * vy)
            x = v * v / (2.0 * 4.03 * 4.03)
            p_blade = 79.86 * (1.0 + 3.0 * v * v / 14400.0)
            p_ind = 88.63 * wp.sqrt(1.0 / (wp.sqrt(1.0 + x * x) + x))
            p_par = 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * v * v * v
            air_s = wp.float32(net.tx_airtime[e, u]) / 1.0e6
            battery[e, u] = wp.max(0.0, battery[e, u] - (p_blade + p_ind + p_par + 15.0) - pa_dc_w * air_s)
            if battery[e, u] <= 0.0:
                net.radio_alive[e, u] = wp.uint8(0)
        intervention[e] = interventions

    @wp.kernel
    def _sense_visible_v79(
        net: NetworkBuffersV67, pos: wp.array3d(dtype=wp.float32),
        truth: wp.array2d(dtype=wp.uint8), ep_step: wp.array(dtype=wp.int32),
        ep_id: wp.array(dtype=wp.int32), battery: wp.array2d(dtype=wp.float32),
        unsafe_mask: wp.array2d(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8),
        b: wp.array2d(dtype=wp.float32), n_b: int, seed: int, pd: float, pf: float, eta: float,
        static_raw_count: int,
    ):
        e, u = wp.tid()
        if (failed[e] != wp.uint8(0) or battery[e, u] <= 0.0
                or unsafe_mask[e, u] != wp.uint8(0) or net.overflow[e] != wp.uint8(0)):
            return
        x = pos[e, u, 0]
        y = pos[e, u, 1]
        z = pos[e, u, 2]
        cx = int(wp.floor(x / 50.0))
        cy = int(wp.floor(y / 50.0))
        t = int(ep_step[e]) + 1
        rng = wp.rand_init(seed + e * 4093 + ep_id[e] * 11939 + u * 131 + t * 1013)
        for i in range(3):
            for j in range(3):
                ix = cx + i - 1
                iy = cy + j - 1
                if ix < 0 or iy < 0 or ix >= GRID or iy >= GRID:
                    continue
                cell_x = (float(ix) + 0.5) * 50.0
                cell_y = (float(iy) + 0.5) * 50.0
                ddx = cell_x - x
                ddy = cell_y - y
                if ddx * ddx + ddy * ddy > 2500.0:
                    continue
                blocked = int(0)
                for k in range(n_b):
                    bx = b[k, 0]
                    by = b[k, 1]
                    br = b[k, 2]
                    bh = b[k, 3]
                    if (cell_x - bx) * (cell_x - bx) + (cell_y - by) * (cell_y - by) <= br * br:
                        blocked = 1
                        break
                    if _segment_hits_cylinder_3d_v79(x, y, z, cell_x, cell_y, 0.0, bx, by, br, bh) != 0:
                        blocked = 1
                        break
                if blocked != 0:
                    continue
                cid = iy * GRID + ix
                q = wp.float32(pd) if truth[e, cid] != wp.uint8(0) else wp.float32(pf)
                detected = wp.randf(rng) < q
                if static_raw_count != 0:
                    # One sample per visible cell per 1s step. 900 < uint16 capacity.
                    # Do not forget static-target evidence merely due to blackout.
                    np_v = int(net.ledger_pos[e, u, u, cid])
                    nn_v = int(net.ledger_neg[e, u, u, cid])
                    if detected:
                        np_v = wp.min(65535, np_v + 1)
                    else:
                        nn_v = wp.min(65535, nn_v + 1)
                    net.ledger_pos[e, u, u, cid] = wp.uint16(np_v)
                    net.ledger_neg[e, u, u, cid] = wp.uint16(nn_v)
                    net.ledger_gen[e, u, u, cid] = wp.int16(t)
                    continue
                old_t = int(net.ledger_gen[e, u, u, cid])
                p = float(net.ledger_pos[e, u, u, cid]) / 256.0
                n = float(net.ledger_neg[e, u, u, cid]) / 256.0
                if old_t >= 0 and t > old_t:
                    decay = wp.exp(-eta * float(t - old_t))
                    p = p * decay
                    n = n * decay
                if detected:
                    p = p + 1.0
                else:
                    n = n + 1.0
                net.ledger_pos[e, u, u, cid] = wp.uint16(wp.clamp(int(wp.round(p * 256.0)), 0, 65535))
                net.ledger_neg[e, u, u, cid] = wp.uint16(wp.clamp(int(wp.round(n * 256.0)), 0, 65535))
                net.ledger_gen[e, u, u, cid] = wp.int16(t)

    @wp.func
    def _fuse_uq88_v79(
        net: NetworkBuffersV67, e: int, receiver: int, cid: int, t: int,
        logmf: float, logme: float, eta: float, static_raw_count: int,
    ) -> wp.vec3:
        p = float(0.0)
        n = float(0.0)
        for org in range(N_UAV):
            if net.ledger_gen[e, receiver, org, cid] < 0:
                continue
            age = wp.max(0.0, float(t - int(net.ledger_gen[e, receiver, org, cid])))
            weight = 1.0 if static_raw_count != 0 else wp.exp(-eta * age) * (1.0 / 256.0)
            p += float(net.ledger_pos[e, receiver, org, cid]) * weight
            n += float(net.ledger_neg[e, receiver, org, cid]) * weight
        return _ds_v79(p, n, logmf, logme)

    @wp.func
    def _fuse_belief_v91(
        net: NetworkBuffersV67, e: int, receiver: int, cid: int, t: int,
        logmf: float, logme: float, eta: float, static_raw_count: int,
        bayesian: int, log_prior: float, llr_pos: float, llr_neg: float,
    ) -> wp.vec3:
        if bayesian == 0:
            return _fuse_uq88_v79(net, e, receiver, cid, t, logmf, logme, eta, static_raw_count)
        p = float(0.0)
        n = float(0.0)
        for org in range(N_UAV):
            if net.ledger_gen[e, receiver, org, cid] < 0:
                continue
            p += float(net.ledger_pos[e, receiver, org, cid])
            n += float(net.ledger_neg[e, receiver, org, cid])
        return _bayes_from_counts_v91(p, n, log_prior, llr_pos, llr_neg)

    @wp.kernel
    def _reward_v79(
        net: NetworkBuffersV67, truth: wp.array2d(dtype=wp.uint8),
        prev_u: wp.array2d(dtype=wp.float32), tp: wp.array2d(dtype=wp.uint8),
        fp: wp.array2d(dtype=wp.uint8), fp_ever: wp.array2d(dtype=wp.uint8),
        battery: wp.array2d(dtype=wp.float32), ep_step: wp.array(dtype=wp.int32),
        failed: wp.array(dtype=wp.uint8), intervention: wp.array(dtype=wp.float32),
        reward: wp.array(dtype=wp.float32), components: wp.array2d(dtype=wp.float32),
        terminated: wp.array(dtype=wp.uint8),
        truncated: wp.array(dtype=wp.uint8), done: wp.array(dtype=wp.uint8),
        terminal_reason: wp.array(dtype=wp.uint8),
        logmf: float, logme: float, eta: float, static_raw_count: int,
        bayesian: int, log_prior: float, llr_pos: float, llr_neg: float,
        confirm: float, uncert_ceiling: float,
        r_tp: float, r_fp: float, fp_hold: float, enable_fp_unlatch: int,
        safety_pen: float, w_aoi: float, w_loc: float, w_block: float,
        lambda_b: float, aoi_penalty: float, ref_age: float, horizon: int,
    ):
        e = wp.tid()
        t = int(ep_step[e]) + 1
        loc = float(0.0)
        delivered = float(0.0)
        fp_pen = float(0.0)
        fp_active = int(0)
        aoi = float(0.0)
        confirmed = int(0)
        for cid in range(NUM_CELLS):
            p = float(0.0)
            n = float(0.0)
            for u in range(N_UAV):
                if net.ledger_gen[e, u, u, cid] < 0:
                    continue
                age = wp.max(0.0, float(t - int(net.ledger_gen[e, u, u, cid])))
                discount = 1.0 if static_raw_count != 0 else wp.exp(-eta * age) * (1.0 / 256.0)
                p += float(net.ledger_pos[e, u, u, cid]) * discount
                n += float(net.ledger_neg[e, u, u, cid]) * discount
            if p + n > 0.0:
                ds = _bayes_from_counts_v91(p, n, log_prior, llr_pos, llr_neg) if bayesian != 0 else _ds_v79(p, n, logmf, logme)
                if ds[2] < prev_u[e, cid]:
                    loc += w_loc * (prev_u[e, cid] - ds[2])
                    prev_u[e, cid] = ds[2]
            m = _fuse_belief_v91(net, e, N_UAV, cid, t, logmf, logme, eta, static_raw_count, bayesian, log_prior, llr_pos, llr_neg)
            if m[0] >= confirm and (bayesian != 0 or m[2] <= uncert_ceiling):
                if truth[e, cid] != wp.uint8(0) and tp[e, cid] == wp.uint8(0):
                    tp[e, cid] = wp.uint8(1)
                    rx_t = float(net.gcs_last_rx_us[e, cid] - net.mission_epoch_us[e]) / 1.0e6
                    gen_t = float(net.gcs_latest[e, cid])
                    delay = wp.max(0.0, rx_t - gen_t)
                    delivered += r_tp + w_aoi * wp.exp(-delay / ref_age)
                elif truth[e, cid] == wp.uint8(0) and fp[e, cid] == wp.uint8(0):
                    fp[e, cid] = wp.uint8(1)
                    if fp_ever[e, cid] == wp.uint8(0):
                        fp_ever[e, cid] = wp.uint8(1)
                        fp_pen += r_fp
            elif enable_fp_unlatch != 0 and truth[e, cid] == wp.uint8(0) and fp[e, cid] == wp.uint8(1):
                fp[e, cid] = wp.uint8(0)
            if fp[e, cid] != wp.uint8(0):
                fp_active += 1
            if truth[e, cid] != wp.uint8(0) and tp[e, cid] != wp.uint8(0):
                confirmed += 1
            last = int(net.gcs_latest[e, cid])
            aoi += wp.max(0.0, float(t - last)) if last >= 0 else float(t)
        energy_alive = int(0)
        air = float(0.0)
        for u in range(N_UAV):
            if battery[e, u] > 0.0:
                energy_alive += 1
            air += float(net.tx_airtime[e, u]) / 1.0e6
        fail_pen = safety_pen if failed[e] != wp.uint8(0) else 0.0
        # V89 evidence ledger: preserve exactly the preexisting scalar reward
        # while logging each term on-device (no CPU synchronize per step).
        components[e, 0] = loc
        components[e, 1] = delivered
        components[e, 2] = -fp_pen
        components[e, 3] = -fp_hold * float(fp_active)
        components[e, 4] = -fail_pen
        components[e, 5] = -w_block * intervention[e]
        components[e, 6] = -lambda_b * air
        components[e, 7] = -aoi_penalty * aoi / (float(NUM_CELLS) * ref_age)
        reward[e] = (
            loc + delivered - fp_pen - fp_hold * float(fp_active) - fail_pen
            - w_block * intervention[e] - lambda_b * air
            - aoi_penalty * aoi / (float(NUM_CELLS) * ref_age)
        )
        ep_step[e] = t
        is_term = wp.uint8(1) if (confirmed >= N_TARGETS or energy_alive == 0 or failed[e] != wp.uint8(0)) else wp.uint8(0)
        is_trunc = wp.uint8(1) if (is_term == wp.uint8(0) and t >= horizon) else wp.uint8(0)
        terminated[e] = is_term
        truncated[e] = is_trunc
        done[e] = wp.uint8(1) if (is_term != wp.uint8(0) or is_trunc != wp.uint8(0)) else wp.uint8(0)
        # Bitmask; reasons may coexist. Stable audit schema:
        # 1=safety_failed, 2=all_uav_batteries_empty,
        # 4=all_targets_confirmed_at_GCS, 8=time_limit_truncated.
        reason = 0
        if failed[e] != wp.uint8(0):
            reason = reason + 1
        if energy_alive == 0:
            reason = reason + 2
        if confirmed >= N_TARGETS:
            reason = reason + 4
        if is_trunc != wp.uint8(0):
            reason = reason + 8
        terminal_reason[e] = wp.uint8(reason)
