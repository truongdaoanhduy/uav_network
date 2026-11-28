"""v5.7 Numerically Stable Dempster-Shafer (FP32/FP64), Exact EWMA AoI Aging,
Zeng 2019 Battery (Steady-Level + Acceleration Thrust), and Kinematic Safety Shield.

Upgrades over v5.4/v5.6:
  - REF-DS-01 + FP32 int-diff refinement: exact up to n=65535 in FP32 & FP64, raises on total conflict.
  - Exponential effective-count aging is implemented in the belief module.
    Count-power DS fusion differs from Xue-style per-sample mass discounting.
  - V57-CAL-BBA: optional reliability cap `max_mass` in `sensor_likelihood_to_bba` so a single false alarm
    cannot immediately exceed `confirm_th=0.90`.
  - V57-ENERGY-ACCEL: `zeng_propulsion_power_with_accel_w` adds thrust-tilt factor kappa=(1+(||a||/g)^2)^0.5
    on induced power and positive kinetic energy delta (m=2.0 kg, eta_prop=0.70).
  - V57-SHIELD: `kinematic_safety_shield` replaces hard-zero velocity veto (which violated a_max=5 m/s^2 by 5x)
    with braking-aware kinematic safe action projection strictly enforcing ||v_new - v_prev||/dt <= a_max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
@dataclass(frozen=True)
class ZengBatteryConfig:
    """Rotary-wing propulsion model (Zeng, Xu & Zhang, IEEE TWC 2019, Eq. 12).

    v5.7 adds optional acceleration thrust-tilt and kinetic energy terms (m=2.0 kg, eta_prop=0.70).
    """
    p0_w: float = 79.86
    pi_w: float = 88.63
    u_tip_mps: float = 120.0
    v0_mps: float = 4.03
    d0: float = 0.6
    rho: float = 1.225
    s: float = 0.05
    area_m2: float = 0.503
    avionics_w: float = 15.0
    rf_tx_pa_w: float = 3.0
    e_max_j: float = 180_000.0  # 50 Wh usable (legacy v5.4 default; v5.7 mission-matched uses 270 kJ)
    uav_mass_kg: float = 2.0
    prop_efficiency: float = 0.70
    gravity_mps2: float = 9.81


def zeng_propulsion_power_w(v_mps: float, cfg: ZengBatteryConfig = ZengBatteryConfig()) -> float:
    """Steady-level propulsion power P(V) in Watts with stable reciprocal induced term."""
    v = max(0.0, float(v_mps))
    p_blade = cfg.p0_w * (1.0 + 3.0 * (v * v) / (cfg.u_tip_mps * cfg.u_tip_mps))
    x = (v * v) / (2.0 * cfg.v0_mps * cfg.v0_mps)
    # Reciprocal identity: sqrt(1 + x^2) - x = 1 / (sqrt(1 + x^2) + x) avoids cancellation at high V
    ind_inner = 1.0 / (math.hypot(1.0, x) + x)
    p_ind = cfg.pi_w * math.sqrt(ind_inner)
    p_par = 0.5 * cfg.d0 * cfg.rho * cfg.s * cfg.area_m2 * (v ** 3)
    return p_blade + p_ind + p_par


def zeng_propulsion_power_with_accel_w(
    v_prev_xy: Sequence[float],
    v_new_xy: Sequence[float],
    dt_s: float = 1.0,
    cfg: ZengBatteryConfig = ZengBatteryConfig(),
) -> float:
    """Zeng 2019 propulsion power augmented with horizontal acceleration thrust-tilt & kinetic delta."""
    vp = np.asarray(v_prev_xy, dtype=np.float64)
    vn = np.asarray(v_new_xy, dtype=np.float64)
    dt = max(1e-6, float(dt_s))
    v_prev = float(np.linalg.norm(vp))
    v_new = float(np.linalg.norm(vn))
    v_avg = 0.5 * (v_prev + v_new)
    a_norm = float(np.linalg.norm(vn - vp)) / dt

    p_blade = cfg.p0_w * (1.0 + 3.0 * (v_avg * v_avg) / (cfg.u_tip_mps * cfg.u_tip_mps))
    x = (v_avg * v_avg) / (2.0 * cfg.v0_mps * cfg.v0_mps)
    # V122 C83: exact Zeng et al. (IEEE TWC 2019, Appendix A Eq. 69) induced power:
    # P_ind = P_i * kappa * sqrt(sqrt(kappa^2 + x^2) - x) = P_i * kappa * sqrt(kappa^2 / (hypot(kappa, x) + x)).
    kappa = math.hypot(1.0, a_norm / cfg.gravity_mps2)
    ind_inner = (kappa * kappa) / (math.hypot(kappa, x) + x)
    p_ind = cfg.pi_w * kappa * math.sqrt(ind_inner)
    p_par = 0.5 * cfg.d0 * cfg.rho * cfg.s * cfg.area_m2 * (v_avg ** 3)
    delta_ke = max(0.0, 0.5 * cfg.uav_mass_kg * (v_new * v_new - v_prev * v_prev))
    p_acc = delta_ke / (max(1e-3, cfg.prop_efficiency) * dt)
    return p_blade + p_ind + p_par + p_acc


def clamp_velocity_with_accel(
    v_prev: Sequence[float],
    v_cmd: Sequence[float],
    dt_s: float = 1.0,
    v_max_mps: float = 25.0,
    a_max_mps2: float = 5.0,
) -> np.ndarray:
    """Enforce both ||v_new - v_prev|| / dt <= a_max and ||v_new|| <= v_max."""
    vp = np.asarray(v_prev, dtype=np.float64)
    vc = np.asarray(v_cmd, dtype=np.float64)
    dv = vc - vp
    dv_norm = float(np.linalg.norm(dv))
    max_dv = a_max_mps2 * dt_s
    if dv_norm > max_dv and dv_norm > 1e-12:
        dv = dv * (max_dv / dv_norm)
    v_new = vp + dv
    v_norm = float(np.linalg.norm(v_new))
    if v_norm > v_max_mps and v_norm > 1e-12:
        v_new = v_new * (v_max_mps / v_norm)
    return v_new


def _segment_point_distance(p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> float:
    delta = p1 - p0
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        return float(np.linalg.norm(p0 - c))
    t = float(np.clip(np.dot(c - p0, delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm((p0 + t * delta) - c))


def _shield_cost_improves_v121(cost: float, best_cost: float) -> bool:
    # Fixed 64*binary32-epsilon cost tie band; never a geometry/safety tolerance.
    # Near ties keep the earlier candidate in the shared 0..65 enumeration.
    tolerance = 7.62939453125e-6 * max(1.0, abs(cost), abs(best_cost))
    return cost < best_cost - tolerance


def kinematic_safety_shield(
    positions_xy: np.ndarray,
    velocities_xy: np.ndarray,
    v_cmd_xy: np.ndarray,
    alive_mask: np.ndarray,
    buildings: Sequence[Tuple[float, float, float, float]],
    map_size_m: float = 1000.0,
    dt_s: float = 1.0,
    v_max_mps: float = 25.0,
    a_max_mps2: float = 5.0,
    uav_safe_distance_m: float = 50.0,
    wall_margin_m: float = 10.0,
    building_margin_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Acceleration-limited motion projection; NOT an unconditional collision guarantee.

    The fallback to kinematic braking may still collide when the current state
    is outside the viable set. Call `trajectory_safety_violations` on the returned
    velocities and terminate/penalize infeasible transitions instead of silently
    treating them as safe. No CBF is used.

    Returns (v_safe_all, intervention_norm).
    """
    n_uav = positions_xy.shape[0]
    v_nom = np.zeros((n_uav, 2), dtype=np.float64)
    v_safe = np.zeros((n_uav, 2), dtype=np.float64)
    intervention = np.zeros(n_uav, dtype=np.float64)

    for u in range(n_uav):
        if not alive_mask[u]:
            v_safe[u] = 0.0
            continue
        v_nom[u] = clamp_velocity_with_accel(
            velocities_xy[u], v_cmd_xy[u], dt_s, v_max_mps, a_max_mps2
        )
        v_safe[u] = v_nom[u].copy()

    def _is_geom_safe(u: int, v_cand: np.ndarray) -> bool:
        p0 = positions_xy[u]
        p1 = p0 + v_cand * dt_s
        # v83: L2-coupled collinear braking envelope (|v_x|*||v||/(2*a_max)) for all 4 walls.
        # Under ||a||_2 <= a_max, braking along -v/||v|| yields a_x = a_max*|v_x|/||v||,
        # eliminating corner inevitable-collision states caused by decoupled L_inf braking.
        v_norm = float(math.hypot(v_cand[0], v_cand[1]))
        brk_x = (abs(v_cand[0]) * v_norm) / (2.0 * a_max_mps2) if a_max_mps2 > 0 else 0.0
        brk_y = (abs(v_cand[1]) * v_norm) / (2.0 * a_max_mps2) if a_max_mps2 > 0 else 0.0
        if p1[0] - (brk_x if v_cand[0] < 0.0 else 0.0) < wall_margin_m - 1e-6:
            return False
        if p1[0] + (brk_x if v_cand[0] > 0.0 else 0.0) > map_size_m - wall_margin_m + 1e-6:
            return False
        if p1[1] - (brk_y if v_cand[1] < 0.0 else 0.0) < wall_margin_m - 1e-6:
            return False
        if p1[1] + (brk_y if v_cand[1] > 0.0 else 0.0) > map_size_m - wall_margin_m + 1e-6:
            return False
        # Building segment clearance + FULL physical braking clearance (removes *0.5 bug)
        for bx, by, br, _ in buildings:
            c = np.array([bx, by], dtype=np.float64)
            if _segment_point_distance(p0, p1, c) <= br + building_margin_m:
                return False
            rel = p1 - c
            dist = float(np.linalg.norm(rel))
            if dist > 1e-6 and a_max_mps2 > 0.0:
                n_hat = rel / dist
                v_toward = max(0.0, -float(np.dot(v_cand, n_hat)))
                if v_toward > 1e-6 and dist - (br + building_margin_m) < (v_toward * v_toward) / (2.0 * a_max_mps2):
                    return False
        return True

    def _pair_min_sep(i: int, j: int, vi: np.ndarray, vj: np.ndarray) -> float:
        rel0 = positions_xy[i] - positions_xy[j]
        reld = (vi - vj) * dt_s
        denom = float(np.dot(reld, reld))
        tau = float(np.clip(-np.dot(rel0, reld) / max(1e-12, denom), 0.0, 1.0))
        return float(np.linalg.norm(rel0 + tau * reld))

    def _pair_safe_with_braking(i: int, j: int, vi: np.ndarray, vj: np.ndarray) -> bool:
        if _pair_min_sep(i, j, vi, vj) < uav_safe_distance_m - 1e-6:
            return False
        rel1 = (positions_xy[i] + vi * dt_s) - (positions_xy[j] + vj * dt_s)
        d1 = float(np.linalg.norm(rel1))
        if d1 > 1e-6 and a_max_mps2 > 0.0:
            dv = vi - vj
            v_close = max(0.0, -float(np.dot(dv, rel1 / d1)))
            a_pair = a_max_mps2
            if d1 - uav_safe_distance_m < (v_close * v_close) / (2.0 * a_pair) - 1e-6:
                return False
        return True

    def _pair_next_step_viable(i: int, j: int, vi: np.ndarray, vj: np.ndarray) -> bool:
        ax_b = (a_max_mps2 if alive_mask[i] else 0.0) + (a_max_mps2 if alive_mask[j] else 0.0)
        fx = (positions_xy[i] - positions_xy[j]) + 2.0 * dt_s * (vi - vj)
        return float(np.linalg.norm(fx)) + ax_b * dt_s * dt_s + 1e-3 >= uav_safe_distance_m

    def _vc_max_disc(s: float, a_rel: float) -> float:
        if s <= 0.0 or dt_s <= 0.0:
            return 0.0
        step_dv = max(1e-6, a_rel * dt_s)
        d1 = step_dv * dt_s
        if s <= d1:
            return s / dt_s
        if s <= 3.0 * d1:
            return (s / dt_s + step_dv) * 0.5
        if s <= 6.0 * d1:
            return (s / dt_s + 3.0 * step_dv) / 3.0
        return (s / dt_s + 6.0 * step_dv) * 0.25

    # Build deterministic kinematic sample grid around v_prev for projection fallback
    angles = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    radii_frac = (1.0, 0.75, 0.5, 0.25)

    def _project_uav(u: int, check_peers: bool) -> np.ndarray:
        vp = velocities_xy[u]
        # Candidate 0: nominal; Candidate 1: maximum kinematic braking toward 0
        v_brake = clamp_velocity_with_accel(vp, np.zeros(2), dt_s, v_max_mps, a_max_mps2)
        cands = [v_safe[u], v_brake]
        # Add samples inside the kinematic acceleration disk ||v - vp|| <= a_max * dt
        max_dv = a_max_mps2 * dt_s
        for rf in radii_frac:
            r = max_dv * rf
            for ang in angles:
                vc = vp + np.array([r * math.cos(ang), r * math.sin(ang)], dtype=np.float64)
                vc = clamp_velocity_with_accel(vp, vc, dt_s, v_max_mps, a_max_mps2)
                cands.append(vc)
        # Shared Warp order: current candidate, brake, then 4 radii x 16 angles.
        best = None
        best_cost = float("inf")
        phys_fallback = None
        phys_fb_dist2 = float("inf")
        geom_fallback = None
        geom_fb_dist2 = float("inf")
        for vc in cands:
            if not _is_geom_safe(u, vc):
                continue
            if check_peers:
                d_brk = float(np.dot(vc - v_brake, vc - v_brake))
                if geom_fallback is None or _shield_cost_improves_v121(d_brk, geom_fb_dist2):
                    geom_fb_dist2 = d_brk
                    geom_fallback = vc
                ok_phys = True
                ok_peers = True
                for j in range(n_uav):
                    if j == u:
                        continue
                    if _pair_min_sep(u, j, vc, v_safe[j]) < uav_safe_distance_m - 1e-6:
                        ok_phys = False
                        ok_peers = False
                        break
                    if not _pair_safe_with_braking(u, j, vc, v_safe[j]):
                        ok_peers = False
                if ok_phys and (phys_fallback is None or _shield_cost_improves_v121(d_brk, phys_fb_dist2)):
                    phys_fb_dist2 = d_brk
                    phys_fallback = vc
                if not ok_peers:
                    continue
            cost = float(np.dot(vc - v_nom[u], vc - v_nom[u]))
            if best is None or _shield_cost_improves_v121(cost, best_cost):
                best, best_cost = vc, cost
        if best is not None:
            return best
        # V90 Tier-2A: geom-safe + physically pair-safe closest to v_brake before Tier-2B.
        if phys_fallback is not None:
            return phys_fallback
        return geom_fallback if geom_fallback is not None else v_brake

    # Pass 1: Geometry safety per UAV
    for u in range(n_uav):
        if alive_mask[u] and not _is_geom_safe(u, v_safe[u]):
            v_safe[u] = _project_uav(u, check_peers=False)

    # Pass 2: Pairwise separation projection (up to 3 coordinate-descent sweeps)
    for _ in range(3):
        any_conflict = False
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                if not _pair_safe_with_braking(i, j, v_safe[i], v_safe[j]):
                    any_conflict = True
                    if alive_mask[i]:
                        v_safe[i] = _project_uav(i, check_peers=True)
                    if alive_mask[j] and not _pair_safe_with_braking(i, j, v_safe[i], v_safe[j]):
                        v_safe[j] = _project_uav(j, check_peers=True)
        if not any_conflict:
            break

    # Pass 3 (V96): Proactive 3-step simultaneous-braking forward-invariance enforcement
    # Bit-consistent with GPU mission_v96_backup.py (24-iter POCS + 8-step convex bisection).
    def _rollout_safe_cpu(v_try: np.ndarray, n_steps: int) -> bool:
        p_sim = positions_xy.copy()
        v_sim = np.where(alive_mask[:, None], v_try, 0.0)
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                if not (alive_mask[i] or alive_mask[j]):
                    continue
                ax_b = (a_max_mps2 if alive_mask[i] else 0.0) + (a_max_mps2 if alive_mask[j] else 0.0)
                fx = (p_sim[i] - p_sim[j]) + 2.0 * dt_s * (v_sim[i] - v_sim[j])
                if float(np.linalg.norm(fx)) + ax_b * dt_s * dt_s + 1e-3 < uav_safe_distance_m:
                    return False
        thr_p = uav_safe_distance_m + 1e-3
        for _ in range(n_steps):
            for u in range(n_uav):
                if not alive_mask[u]:
                    continue
                p0, p1 = p_sim[u], p_sim[u] + v_sim[u] * dt_s
                if (p1 < wall_margin_m + 1e-3).any() or (p1 > map_size_m - wall_margin_m - 1e-3).any():
                    return False
                for bx, by, br, _ in buildings:
                    if _segment_point_distance(p0, p1, np.array([bx, by], dtype=np.float64)) < br + building_margin_m + 1e-3:
                        return False
            for i in range(n_uav):
                for j in range(i + 1, n_uav):
                    if not (alive_mask[i] or alive_mask[j]):
                        continue
                    r = p_sim[i] - p_sim[j]
                    dv = (v_sim[i] - v_sim[j]) * dt_s
                    tau = float(np.clip(-np.dot(r, dv) / max(1e-12, float(np.dot(dv, dv))), 0.0, 1.0))
                    if float(np.linalg.norm(r + tau * dv)) < thr_p:
                        return False
            p_sim = p_sim + v_sim * dt_s
            for u in range(n_uav):
                if alive_mask[u]:
                    v_sim[u] = clamp_velocity_with_accel(v_sim[u], np.zeros(2), dt_s, v_max_mps, a_max_mps2)
        return True

    n_brk = max(3, int(math.ceil(v_max_mps / max(1e-6, a_max_mps2 * dt_s))))
    if not _rollout_safe_cpu(v_safe, n_brk):
        v_rec = v_safe.copy()
        for _ in range(24):
            for i in range(n_uav):
                for j in range(i + 1, n_uav):
                    wi = 1.0 if alive_mask[i] else 0.0
                    wj = 1.0 if alive_mask[j] else 0.0
                    wsum = wi + wj
                    if wsum <= 0.0:
                        continue
                    r = positions_xy[i] - positions_xy[j]
                    d = float(np.linalg.norm(r))
                    if d <= 1e-6:
                        continue
                    n_hat = r / d
                    vc_lim = _vc_max_disc(d - (uav_safe_distance_m + 0.02), 0.5 * wsum * a_max_mps2)
                    viol = -vc_lim - float(np.dot(n_hat, v_rec[i] - v_rec[j]))
                    if viol > 0.0:
                        inv_w = 1.0 / wsum
                        if wi > 0.0:
                            v_rec[i] = v_rec[i] + (wi * inv_w) * viol * n_hat
                        if wj > 0.0:
                            v_rec[j] = v_rec[j] - (wj * inv_w) * viol * n_hat
            for u in range(n_uav):
                if not alive_mask[u]:
                    continue
                px, py = float(positions_xy[u, 0]), float(positions_xy[u, 1])
                vx = float(np.clip(v_rec[u, 0], -_vc_max_disc(px - (wall_margin_m + 0.01), a_max_mps2), _vc_max_disc(map_size_m - (wall_margin_m + 0.01) - px, a_max_mps2)))
                vy = float(np.clip(v_rec[u, 1], -_vc_max_disc(py - (wall_margin_m + 0.01), a_max_mps2), _vc_max_disc(map_size_m - (wall_margin_m + 0.01) - py, a_max_mps2)))
                vu = np.array([vx, vy], dtype=np.float64)
                for bx, by, br, _ in buildings:
                    rb = positions_xy[u] - np.array([bx, by], dtype=np.float64)
                    db = float(np.linalg.norm(rb))
                    if db > 1e-6:
                        nb = rb / db
                        vb_viol = -_vc_max_disc(db - (br + building_margin_m + 0.01), a_max_mps2) - float(np.dot(nb, vu))
                        if vb_viol > 0.0:
                            vu = vu + vb_viol * nb
                v_rec[u] = clamp_velocity_with_accel(velocities_xy[u], vu, dt_s, v_max_mps, a_max_mps2)
        v_brk_all = np.array([clamp_velocity_with_accel(velocities_xy[u], np.zeros(2), dt_s, v_max_mps, a_max_mps2) if alive_mask[u] else np.zeros(2) for u in range(n_uav)], dtype=np.float64)
        if _rollout_safe_cpu(v_rec, n_brk):
            v_safe = v_rec
        elif _rollout_safe_cpu(v_brk_all, n_brk):
            def _bisect(v_tgt: np.ndarray) -> np.ndarray:
                lo, hi = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    if _rollout_safe_cpu(v_brk_all + mid * (v_tgt - v_brk_all), n_brk):
                        lo = mid
                    else:
                        hi = mid
                return v_brk_all + lo * (v_tgt - v_brk_all)
            vc1, vc2 = _bisect(v_rec), _bisect(v_safe)
            c1 = float(np.sum((vc1[alive_mask] - v_nom[alive_mask]) ** 2))
            c2 = float(np.sum((vc2[alive_mask] - v_nom[alive_mask]) ** 2))
            v_safe = vc1 if c1 <= c2 else vc2
        elif _rollout_safe_cpu(v_rec, 1) and not _rollout_safe_cpu(v_safe, 1):
            v_safe = v_rec

    for u in range(n_uav):
        if alive_mask[u]:
            intervention[u] = float(np.linalg.norm(v_nom[u] - v_safe[u])) / max(1e-6, v_max_mps)
    return v_safe, intervention


def trajectory_safety_violations(
    positions_xy: np.ndarray,
    proposed_velocities_xy: np.ndarray,
    alive_mask: np.ndarray,
    buildings: Sequence[Tuple[float, float, float, float]],
    *,
    map_size_m: float,
    dt_s: float,
    uav_safe_distance_m: float,
    wall_margin_m: float = 10.0,
    building_margin_m: float = 2.0,
) -> dict[str, list]:
    """Detect failures of swept 2-D motion, including an infeasible shield fallback.

    This is a *detector*, not an avoidance controller. It checks continuous
    pairwise separation over the step, swept cylinder footprint contact, and
    boundary crossings. It deliberately includes stationary depleted UAVs as
    obstacles; those nodes cannot simply disappear from collision checks.
    """
    pos = np.asarray(positions_xy, dtype=np.float64)
    vel = np.asarray(proposed_velocities_xy, dtype=np.float64)
    alive = np.asarray(alive_mask, dtype=bool)
    if pos.ndim != 2 or pos.shape[1] != 2 or vel.shape != pos.shape or alive.shape != (len(pos),):
        raise ValueError("Expected positions/velocities (N,2) and alive_mask (N,)")
    if dt_s <= 0 or uav_safe_distance_m <= 0:
        raise ValueError("dt_s and uav_safe_distance_m must be positive")
    future = pos + vel * dt_s
    violations: dict[str, list] = {"boundary": [], "building": [], "pairwise": []}
    for i in range(len(pos)):
        if not alive[i]:
            continue
        p0, p1 = pos[i], future[i]
        if ((p1 < wall_margin_m).any() or (p1 > map_size_m - wall_margin_m).any()
                or (p0 < wall_margin_m).any() or (p0 > map_size_m - wall_margin_m).any()):
            violations["boundary"].append(i)
        for bidx, (bx, by, br, _height) in enumerate(buildings):
            if _segment_point_distance(p0, p1, np.array([bx, by], dtype=np.float64)) < br + building_margin_m - 1e-6:
                violations["building"].append((i, bidx))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if not (alive[i] or alive[j]):
                continue
            rel0 = pos[i] - pos[j]
            vi = vel[i] if alive[i] else np.zeros(2)
            vj = vel[j] if alive[j] else np.zeros(2)
            rel_displacement = (vi - vj) * dt_s
            tau = float(np.clip(-np.dot(rel0, rel_displacement) / max(1e-12, np.dot(rel_displacement, rel_displacement)), 0.0, 1.0))
            if np.linalg.norm(rel0 + tau * rel_displacement) < uav_safe_distance_m - 1e-6:
                violations["pairwise"].append((i, j))
    return violations


def uav_step_energy_j(
    actual_speed_mps: float,
    own_tx_airtime_s: float,
    dt_s: float = 1.0,
    cfg: ZengBatteryConfig = ZengBatteryConfig(),
    v_prev_xy: Sequence[float] | None = None,
    v_new_xy: Sequence[float] | None = None,
    tx_power_w: float | None = None,
) -> float:
    """Per-UAV energy drain from ACTUAL accepted speed (or v_prev->v_new with accel) + own TX airtime."""
    if v_prev_xy is not None and v_new_xy is not None:
        p_prop = zeng_propulsion_power_with_accel_w(v_prev_xy, v_new_xy, dt_s, cfg)
    else:
        p_prop = zeng_propulsion_power_w(actual_speed_mps, cfg)
    p_flight = p_prop + cfg.avionics_w
    pa_w = cfg.rf_tx_pa_w if tx_power_w is None else rf_pa_dc_power_w(tx_power_w)
    return p_flight * dt_s + pa_w * max(0.0, float(own_tx_airtime_s))


def rf_pa_dc_power_w(tx_power_w: float, p_circuit_w: float = 0.5, eta_pa: float = 0.20) -> float:
    """Linear PA DC power model: 3.0 W at P_tx=0.50 W (v6.1 baseline), 0.75 W at P_tx=0.05 W."""
    return float(p_circuit_w) + max(0.0, float(tx_power_w)) / max(1e-3, float(eta_pa))
