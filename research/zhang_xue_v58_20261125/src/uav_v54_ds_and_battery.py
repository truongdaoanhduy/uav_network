"""v5.7 Numerically Stable Dempster-Shafer (FP32/FP64), Exact EWMA AoI Aging,
Zeng 2019 Battery (Steady-Level + Acceleration Thrust), and Kinematic Safety Shield.

Upgrades over v5.4/v5.6:
  - REF-DS-01 + FP32 int-diff refinement: exact up to n=65535 in FP32 & FP64, raises on total conflict.
  - V57-AOI-EXACT: O(1) EWMA effective counts (`ewma_effective_counts_update` + `counts_to_canonical_ds_effective`)
    eliminating artificial rejuvenation of stale evidence (matches Xue 2026 Eq. 9-10 per-sample exponential decay).
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


def counts_to_canonical_ds_stable(
    n_pos: int, n_neg: int, mf: float = 0.80, me: float = 0.80
) -> Tuple[float, float, float]:
    """Exact overflow/underflow-free canonical Dempster-Shafer on Theta = {T, F}.

    Handles n_pos, n_neg in [0, 65535] without cancellation in FP32 or FP64.
    Raises ValueError on undefined total conflict (K = 1, i.e., mf=1, me=1, n_pos>0, n_neg>0).
    """
    if n_pos < 0 or n_neg < 0:
        raise ValueError(f"Counts must be non-negative: ({n_pos}, {n_neg})")
    if not (0.0 <= mf <= 1.0 and 0.0 <= me <= 1.0):
        raise ValueError(f"BBAs must lie in [0, 1]: ({mf}, {me})")
    if n_pos == 0 and n_neg == 0:
        return 0.0, 0.0, 1.0
    if mf == 1.0 and me == 1.0 and n_pos > 0 and n_neg > 0:
        raise ValueError("Total Dempster-Shafer conflict (K = 1) is undefined")
    if mf == 1.0 and n_pos > 0:
        return 1.0, 0.0, 0.0
    if me == 1.0 and n_neg > 0:
        return 0.0, 1.0, 0.0

    lmf = math.log1p(-mf)
    lme = math.log1p(-me)
    a = 0.0 if n_pos == 0 else n_pos * lmf
    b = 0.0 if n_neg == 0 else n_neg * lme

    if mf == me:
        # Integer-difference path prevents FP32 subtraction cancellation at large n
        d = (n_pos - n_neg) * lmf  # equals a - b
        if n_pos <= n_neg:  # a >= b because lmf <= 0
            pa, pb, pu = 1.0, math.exp(-d), math.exp(b)
        else:
            pa, pb, pu = math.exp(d), 1.0, math.exp(a)
    else:
        s = max(a, b)
        pa = math.exp(a - s)
        pb = math.exp(b - s)
        pu = math.exp(min(a, b))

    t_num = 0.0 if n_pos == 0 else pb * (-math.expm1(a))
    f_num = 0.0 if n_neg == 0 else pa * (-math.expm1(b))
    z = t_num + f_num + pu
    return t_num / z, f_num / z, pu / z


def ewma_effective_counts_update(
    eff_pos: float,
    eff_neg: float,
    last_ts_s: float,
    now_s: float,
    det: bool,
    eta: float = 0.02,
) -> Tuple[float, float]:
    """O(1) exact exponential aging of per-sample evidence counts (Xue 2026 Eq. 9-10).

    Because exp(-eta * (t - t1)) == exp(-eta * (t - t2)) * exp(-eta * (t2 - t1)),
    decaying accumulated effective counts before adding the new observation eliminates
    the 'artificial rejuvenation' defect of frozen cumulative-snapshot discounting.
    """
    if eta < 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and non-negative")
    dt = max(0.0, float(now_s) - float(last_ts_s)) if last_ts_s >= 0.0 else 0.0
    decay = math.exp(-eta * dt) if (eta > 0.0 and dt > 0.0) else 1.0
    new_pos = float(eff_pos) * decay + (1.0 if det else 0.0)
    new_neg = float(eff_neg) * decay + (0.0 if det else 1.0)
    return new_pos, new_neg


def counts_to_canonical_ds_effective(
    eff_pos: float, eff_neg: float, mf: float = 0.80, me: float = 0.80
) -> Tuple[float, float, float]:
    """Numerically stable canonical DS for continuous/discounted effective counts >= 0."""
    ep = float(eff_pos)
    en = float(eff_neg)
    if ep < 0.0 or en < 0.0 or not (math.isfinite(ep) and math.isfinite(en)):
        raise ValueError(f"Effective counts must be finite non-negative: ({ep}, {en})")
    if not (0.0 <= mf <= 1.0 and 0.0 <= me <= 1.0):
        raise ValueError(f"BBAs must lie in [0, 1]: ({mf}, {me})")
    if ep == 0.0 and en == 0.0:
        return 0.0, 0.0, 1.0
    if mf == 1.0 and me == 1.0 and ep > 0.0 and en > 0.0:
        raise ValueError("Total Dempster-Shafer conflict (K = 1) is undefined")
    if mf == 1.0 and ep > 0.0:
        return 1.0, 0.0, 0.0
    if me == 1.0 and en > 0.0:
        return 0.0, 1.0, 0.0

    lmf = math.log1p(-mf)
    lme = math.log1p(-me)
    a = 0.0 if ep == 0.0 else ep * lmf
    b = 0.0 if en == 0.0 else en * lme
    s = max(a, b)
    pa = math.exp(a - s)
    pb = math.exp(b - s)
    pu = math.exp(min(a, b))
    t_num = 0.0 if ep == 0.0 else pb * (-math.expm1(a))
    f_num = 0.0 if en == 0.0 else pa * (-math.expm1(b))
    z = t_num + f_num + pu
    return t_num / z, f_num / z, pu / z


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
    ind_inner = 1.0 / (math.hypot(1.0, x) + x)
    kappa = math.hypot(1.0, a_norm / cfg.gravity_mps2)
    p_ind = cfg.pi_w * (kappa ** 1.5) * math.sqrt(ind_inner)
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
        # Map boundaries with 1-step braking awareness
        sp_x = abs(v_cand[0])
        sp_y = abs(v_cand[1])
        brk_x = (sp_x * sp_x) / (2.0 * a_max_mps2) if a_max_mps2 > 0 else 0.0
        brk_y = (sp_y * sp_y) / (2.0 * a_max_mps2) if a_max_mps2 > 0 else 0.0
        if (p1[0] < wall_margin_m and v_cand[0] < 0) or (p1[0] - (brk_x if v_cand[0] < 0 else 0.0) < wall_margin_m - 1e-6 and v_cand[0] < 0 and p0[0] >= wall_margin_m):
            if p1[0] < wall_margin_m:
                return False
        if p1[0] < wall_margin_m or p1[0] > map_size_m - wall_margin_m:
            return False
        if p1[1] < wall_margin_m or p1[1] > map_size_m - wall_margin_m:
            return False
        # Building segment clearance + braking clearance along radial approach
        for bx, by, br, _ in buildings:
            c = np.array([bx, by], dtype=np.float64)
            if _segment_point_distance(p0, p1, c) <= br + building_margin_m:
                return False
            # Braking distance check if heading toward building
            rel = p1 - c
            dist = float(np.linalg.norm(rel))
            if dist > 1e-6:
                n_hat = rel / dist
                v_toward = max(0.0, -float(np.dot(v_cand, n_hat)))
                if dist - (br + building_margin_m) < (v_toward * v_toward) / (2.0 * a_max_mps2) * 0.5:
                    # Only enforce braking margin if already outside building clearance
                    if float(np.linalg.norm(p0 - c)) > br + building_margin_m + 1.0:
                        return False
        return True

    def _pair_min_sep(i: int, j: int, vi: np.ndarray, vj: np.ndarray) -> float:
        rel0 = positions_xy[i] - positions_xy[j]
        reld = (vi - vj) * dt_s
        denom = float(np.dot(reld, reld))
        tau = float(np.clip(-np.dot(rel0, reld) / max(1e-12, denom), 0.0, 1.0))
        return float(np.linalg.norm(rel0 + tau * reld))

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
        # Sort by distance to desired v_nom[u]
        cands.sort(key=lambda vc: float(np.dot(vc - v_nom[u], vc - v_nom[u])))
        for vc in cands:
            if not _is_geom_safe(u, vc):
                continue
            if check_peers:
                ok_peers = True
                for j in range(n_uav):
                    if j == u:
                        continue
                    if _pair_min_sep(u, j, vc, v_safe[j]) < uav_safe_distance_m - 1e-6:
                        ok_peers = False
                        break
                if not ok_peers:
                    continue
            return vc
        # Worst-case fallback: maximum kinematic braking (strictly respects a_max!)
        return v_brake

    # Pass 1: Geometry safety per UAV
    for u in range(n_uav):
        if alive_mask[u] and not _is_geom_safe(u, v_safe[u]):
            v_safe[u] = _project_uav(u, check_peers=False)

    # Pass 2: Pairwise separation projection (up to 3 coordinate-descent sweeps)
    for _ in range(3):
        any_conflict = False
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                if _pair_min_sep(i, j, v_safe[i], v_safe[j]) < uav_safe_distance_m - 1e-6:
                    any_conflict = True
                    if alive_mask[i]:
                        v_safe[i] = _project_uav(i, check_peers=True)
                    if alive_mask[j] and _pair_min_sep(i, j, v_safe[i], v_safe[j]) < uav_safe_distance_m - 1e-6:
                        v_safe[j] = _project_uav(j, check_peers=True)
        if not any_conflict:
            break

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
) -> float:
    """Per-UAV energy drain from ACTUAL accepted speed (or v_prev->v_new with accel) + own TX airtime."""
    if v_prev_xy is not None and v_new_xy is not None:
        p_prop = zeng_propulsion_power_with_accel_w(v_prev_xy, v_new_xy, dt_s, cfg)
    else:
        p_prop = zeng_propulsion_power_w(actual_speed_mps, cfg)
    p_flight = p_prop + cfg.avionics_w
    return p_flight * dt_s + cfg.rf_tx_pa_w * max(0.0, float(own_tx_airtime_s))


def discounted_mass(mass: Sequence[float], freshness_weight: float) -> Tuple[float, float, float]:
    """Xue-style evidence reliability discount: m'(T/E)=psi*m(T/E), m'(U)=1-psi*(1-m(U))."""
    mf, me, mu = (float(x) for x in mass)
    w = float(freshness_weight)
    if not (math.isfinite(w) and 0.0 <= w <= 1.0) or not all(math.isfinite(x) and x >= 0.0 for x in (mf, me, mu)):
        raise ValueError("Invalid mass or freshness weight")
    if abs(mf + me + mu - 1.0) > 1e-7:
        raise ValueError("Mass must sum to 1")
    if w == 1.0:
        return mf, me, mu
    if w == 0.0:
        return 0.0, 0.0, 1.0
    return w * mf, w * me, 1.0 - w * (mf + me)


def combine_ds_masses(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Orthogonal sum of two binary-frame DS BPAs; never hide total conflict."""
    at, af, au = (float(v) for v in a)
    bt, bf, bu = (float(v) for v in b)
    for m in ((at, af, au), (bt, bf, bu)):
        if not all(math.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in m) or abs(sum(m) - 1.0) > 1e-7:
            raise ValueError("Invalid BPA")
    k = at * bf + af * bt
    if k >= 1.0 - 1e-12:
        raise ValueError("Dempster total conflict, K=1")
    denom = 1.0 - k
    tt = (at * bt + at * bu + au * bt) / denom
    ff = (af * bf + af * bu + au * bf) / denom
    uu = (au * bu) / denom
    z = tt + ff + uu
    return tt / z, ff / z, uu / z


def sensor_likelihood_to_bba(
    pd: float, pf: float, max_mass: float | None = None
) -> Tuple[float, float]:
    """Likelihood-ratio-to-BBA mapping with optional v5.7 reliability cap `max_mass`.

    Positive evidence support = 1 - P(+|empty)/P(+|target).
    Negative evidence support = 1 - P(-|target)/P(-|empty).
    When `max_mass` (e.g. 0.80) is provided, caps single-observation support so a single
    false alarm cannot immediately trigger `confirm_th=0.90`.
    """
    pd, pf = float(pd), float(pf)
    if not (0.0 < pf < pd < 1.0):
        raise ValueError("require 0 < Pf < Pd < 1")
    mf = 1.0 - pf / pd
    me = 1.0 - (1.0 - pd) / (1.0 - pf)
    if max_mass is not None:
        cap = float(max_mass)
        if not (0.0 < cap <= 1.0):
            raise ValueError("max_mass must lie in (0, 1]")
        mf = min(mf, cap)
        me = min(me, cap)
    return mf, me


def fuse_provenance_counts(
    n_pos: Sequence[int], n_neg: Sequence[int], gen_ts_s: Sequence[int],
    rx_age_s: Sequence[float], *, mf: float = 0.8, me: float = 0.8, eta: float = 0.0,
    mode: str = "sum",
) -> Tuple[float, float, float]:
    """Fuse uniquely origin-tagged cumulative evidence (not repeated re-transmissions).

    `sum`: combine each origin's canonical DS mass once, with delivery-age discount per origin.
    `zhang_max`: exact Eq (16) MAX of counts before canonical DS, no AoI discount (baseline).
    """
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite nonnegative")
    if mode not in ("sum", "zhang_max"):
        raise ValueError("unknown fusion mode")
    p = [int(x) for x in n_pos]
    n = [int(x) for x in n_neg]
    ts = [int(x) for x in gen_ts_s]
    delays = [float(x) for x in rx_age_s]
    if not (len(p) == len(n) == len(ts) == len(delays)):
        raise ValueError("origin arrays must match")
    if any(x < 0 for x in p + n):
        raise ValueError("negative counts")
    if mode == "zhang_max":
        return counts_to_canonical_ds_stable(max(p, default=0), max(n, default=0), mf, me)
    if eta == 0.0:
        return counts_to_canonical_ds_stable(sum(p), sum(n), mf, me)
    result = (0.0, 0.0, 1.0)
    for pos, neg, t, delay in zip(p, n, ts, delays):
        if pos == 0 and neg == 0:
            continue
        if t < 0 or not math.isfinite(delay) or delay < 0:
            raise ValueError("non-empty origin evidence requires valid generation timestamp and receive age")
        single = counts_to_canonical_ds_stable(pos, neg, mf, me)
        psi = math.exp(-eta * delay)
        result = combine_ds_masses(result, discounted_mass(single, psi))
    return result
