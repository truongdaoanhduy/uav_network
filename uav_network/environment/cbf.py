"""Acceleration-space CBF safety filtering for XYZ motion.

The filter keeps the MARL acceleration when it already satisfies every safety
half-space and otherwise projects it onto the most violated constraint.  The
constraint form follows second-order acceleration CBF/lookahead filters used
for UAV collision avoidance, while staying dependency-free for a later Warp
port.
"""
from __future__ import annotations

import numpy as np


def _horizon(closing_speed: float, decel: float, dt: float) -> float:
    return max(float(dt), float(closing_speed) / max(float(decel), 1e-12))


def _bound(gap: float, closing_speed: float, horizon: float, dt: float) -> float:
    # move_xyz uses semi-implicit Euler: v[k+1]=v[k]+a*dt, then
    # p[k+1]=p[k]+v[k+1]*dt.  Over horizon H this gives a larger
    # acceleration displacement than the continuous 0.5*a*H^2 term.
    coefficient = 0.5 * horizon * (horizon + dt)
    return (float(gap) - float(closing_speed) * horizon) / coefficient


def filter_xyz_acceleration(
    positions,
    velocities,
    nominal_accel,
    obstacles,
    heights,
    world,
    disabled=None,
    *,
    max_iterations: int = 32,
    tolerance: float = 1e-9,
):
    """Project nominal physical accelerations onto CBF safety half-spaces.

    Inputs and output accelerations are in m/s^2.  Obstacle rows use the
    finite-cylinder geometry already used by the XYZ environment: below the
    cylinder top, lateral clearance is protected; above the top while inside
    the footprint, vertical clearance is protected.  When both routes are
    currently available, the nominal command is left alone unless it would
    violate both, in which case the smaller correction is selected.
    """
    p = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    a0 = np.asarray(nominal_accel, dtype=float)
    obs = np.asarray(obstacles, dtype=float).reshape(-1, 3)
    hgt = np.asarray(heights, dtype=float)
    n = len(p)
    disabled = np.zeros(n, dtype=bool) if disabled is None else np.asarray(disabled, dtype=bool)
    amax = float(world.max_accel_mps2)
    dt = float(world.dt_s)

    nominal = np.clip(a0, -amax, amax).copy()
    nominal[disabled] = 0.0
    rows, bounds, kinds, owners = [], [], [], []

    def add(row, bound, kind, owner):
        rows.append(np.asarray(row, dtype=float))
        bounds.append(float(bound))
        kinds.append(kind)
        owners.append(tuple(owner))

    def single_row(i, direction):
        row = np.zeros((n, 3), dtype=float)
        row[i] = direction
        return row.reshape(-1)

    def constraint_values(i, direction, gap, closing, decel):
        H = _horizon(closing, decel, dt)
        row = single_row(i, direction)
        b = _bound(gap, closing, H, dt)
        return row, b

    # Map/altitude planes.  Direction always points toward the unsafe side.
    lower = np.array([0.0, 0.0, float(world.altitude_min_m)])
    upper = np.array([float(world.map_size_m), float(world.map_size_m), float(world.altitude_max_m)])
    axes = np.eye(3)
    for i in range(n):
        if disabled[i]:
            continue
        for axis in range(3):
            gap_lo = p[i, axis] - lower[axis]
            closing_lo = max(0.0, -v[i, axis])
            row, b = constraint_values(i, -axes[axis], gap_lo, closing_lo, amax)
            add(row, b, "boundary", (i,))

            gap_hi = upper[axis] - p[i, axis]
            closing_hi = max(0.0, v[i, axis])
            row, b = constraint_values(i, axes[axis], gap_hi, closing_hi, amax)
            add(row, b, "boundary", (i,))

    # Static finite cylinders.  The safe set is outside the lateral wall OR
    # above the top.  If both routes are currently available, intervene only
    # when the nominal acceleration violates both and choose the cheaper one.
    for i in range(n):
        if disabled[i]:
            continue
        for obstacle, height in zip(obs, hgt):
            delta = obstacle[:2] - p[i, :2]
            radial = float(np.linalg.norm(delta))
            radius = float(obstacle[2] + world.obstacle_clearance_m)
            top = float(height + world.obstacle_clearance_m)
            if radial > 1e-12:
                toward_xy = np.array([delta[0] / radial, delta[1] / radial, 0.0])
            else:
                toward_xy = np.array([1.0, 0.0, 0.0])
            side_gap = radial - radius
            side_closing = max(0.0, float(np.dot(v[i], toward_xy)))
            side = constraint_values(i, toward_xy, side_gap, side_closing, amax)

            top_gap = p[i, 2] - top
            top_closing = max(0.0, -float(v[i, 2]))
            top_c = constraint_values(i, np.array([0.0, 0.0, -1.0]), top_gap, top_closing, amax)

            if p[i, 2] <= top and radial >= radius:
                add(*side, "obstacle", (i,))
            elif radial <= radius and p[i, 2] >= top:
                add(*top_c, "obstacle", (i,))
            elif p[i, 2] <= top and radial <= radius:
                # Already inside the expanded cylinder: expose both escape
                # directions and let the projector make a best-effort recovery.
                add(*side, "obstacle", (i,))
                add(*top_c, "obstacle", (i,))
            else:
                flat = nominal.reshape(-1)
                side_violation = float(np.dot(side[0], flat) - side[1])
                top_violation = float(np.dot(top_c[0], flat) - top_c[1])
                if side_violation > tolerance and top_violation > tolerance:
                    side_cost = side_violation / max(float(np.dot(side[0], side[0])), 1e-12)
                    top_cost = top_violation / max(float(np.dot(top_c[0], top_c[0])), 1e-12)
                    add(*(side if side_cost <= top_cost else top_c), "obstacle", (i,))

    # Pairwise peer constraints couple both accelerations.  Because both UAVs
    # can contribute, relative stopping authority is 2*a_max.
    for i in range(n):
        for j in range(i):
            delta = p[j] - p[i]
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = delta / distance
            closing = max(0.0, float(np.dot(v[i] - v[j], direction)))
            H = _horizon(closing, 2.0 * amax, dt)
            b = _bound(distance - float(world.safety_distance_m), closing, H, dt)
            row = np.zeros((n, 3), dtype=float)
            row[i] = direction
            row[j] = -direction
            add(row.reshape(-1), b, "peer", (i, j))

    if not rows:
        return nominal, {
            "cbf_interventions": 0,
            "cbf_obstacle_interventions": 0,
            "cbf_peer_interventions": 0,
            "cbf_boundary_interventions": 0,
            "cbf_infeasible": 0,
            "cbf_iterations": 0,
            "cbf_max_violation": 0.0,
        }

    G = np.vstack(rows)
    b = np.asarray(bounds)
    u_nom = nominal.reshape(-1)
    initial_violation = G @ u_nom - b
    active_owners = {kind: set() for kind in ("obstacle", "peer", "boundary")}
    for violation, kind, owner in zip(initial_violation, kinds, owners):
        if violation > tolerance:
            active_owners[kind].update(owner)

    u = u_nom.copy()
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        violation = G @ u - b
        idx = int(np.argmax(violation))
        worst = float(violation[idx])
        if worst <= tolerance:
            iterations -= 1
            break
        row = G[idx]
        denom = float(np.dot(row, row))
        if denom <= 1e-24:
            break
        u -= (worst / denom) * row
        u = np.clip(u, -amax, amax)
        if disabled.any():
            u.reshape(n, 3)[disabled] = 0.0

    final_violation = G @ u - b
    max_violation = max(0.0, float(np.max(final_violation)))
    safe = u.reshape(n, 3)
    changed = np.linalg.norm(safe - nominal, axis=1) > 1e-9
    changed[disabled] = False
    return safe, {
        "cbf_interventions": int(changed.sum()),
        "cbf_obstacle_interventions": len(active_owners["obstacle"]),
        "cbf_peer_interventions": len(active_owners["peer"]),
        "cbf_boundary_interventions": len(active_owners["boundary"]),
        "cbf_infeasible": int(max_violation > 1e-6),
        "cbf_iterations": int(iterations),
        "cbf_max_violation": max_violation,
    }
