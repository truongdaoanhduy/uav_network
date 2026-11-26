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
        geom_fallback = None
        geom_fb_dist2 = float("inf")
        for vc in cands:
            if not _is_geom_safe(u, vc):
                continue
            if check_peers:
                d_brk = float(np.dot(vc - v_brake, vc - v_brake))
                if d_brk < geom_fb_dist2:
                    geom_fb_dist2 = d_brk
                    geom_fallback = vc
                ok_peers = True
                for j in range(n_uav):
                    if j == u:
                        continue
                    if not _pair_safe_with_braking(u, j, vc, v_safe[j]):
                        ok_peers = False
                        break
                if not ok_peers:
                    continue
            return vc
        # V83: preserve geometry-safe braking fallback (closest to v_brake) before raw v_brake.
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
