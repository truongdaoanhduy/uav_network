"""Deterministic XY dynamic-window filter with a swept braking certificate.

The filter sees obstacle geometry and peer beacons, never targets or reports.
It passes safe policy velocities unchanged. Unsafe headings get tangential
alternatives, not a distance-dependent repulsive force. This is a local safety
filter inspired by DWA, not the original unicycle DWA or a global path planner.
"""
import numpy as np


def _cap(velocity, maximum):
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return velocity * np.minimum(1, maximum / np.maximum(speed, 1e-12))


def reachable_velocities(desired, previous, world):
    """Project onto the per-axis acceleration box intersected with the speed disk."""
    step = world.max_accel_mps2 * world.dt_s
    lower, upper = previous - step, previous + step
    result = np.clip(desired, lower, upper)
    over = np.linalg.norm(result, axis=-1) > world.max_speed_mps
    if not over.any():
        return result
    # Euclidean projection: clip(desired / (1 + lambda), lower, upper).
    low = np.zeros(result.shape[:-1])
    high = np.ones_like(low)
    feasible = np.linalg.norm(np.clip(np.zeros_like(result), lower, upper), axis=-1) <= world.max_speed_mps
    pending = over & feasible
    while True:
        candidate = np.clip(desired / (1 + high[..., None]), lower, upper)
        expand = pending & (np.linalg.norm(candidate, axis=-1) > world.max_speed_mps)
        if not expand.any():
            break
        high[expand] *= 2
    for _ in range(40):
        middle = (low + high) / 2
        candidate = np.clip(desired / (1 + middle[..., None]), lower, upper)
        outside = np.linalg.norm(candidate, axis=-1) > world.max_speed_mps
        low = np.where(outside, middle, low)
        high = np.where(outside, high, middle)
    projected = np.clip(desired / (1 + high[..., None]), lower, upper)
    # An externally injected state may have no reachable point inside the disk.
    return np.where(pending[..., None], projected, _cap(result, world.max_speed_mps))


def _brake(velocity, step):
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    return velocity * np.maximum(0, 1 - step / np.maximum(speed, 1e-12))


def braking_paths(positions, velocities, world):
    """One candidate step, then bounded radial braking all the way to rest."""
    step = world.max_accel_mps2 * world.dt_s
    horizon = int(np.ceil(world.max_speed_mps / step)) + 1
    points = [np.asarray(positions, dtype=float)]
    velocity = np.asarray(velocities, dtype=float).copy()
    for _ in range(horizon):
        points.append(points[-1] + velocity * world.dt_s)
        velocity = _brake(velocity, step)
    return np.stack(points, axis=1)


def _distances(paths, centers):
    start, delta = paths[:, :-1, None], np.diff(paths, axis=1)[:, :, None]
    offset = np.asarray(centers)[None, None] - start
    norm_sq = np.square(delta).sum(-1)
    fraction = np.clip((offset * delta).sum(-1) / np.maximum(norm_sq, 1e-24), 0, 1)
    distance = np.linalg.norm(offset - fraction[..., None] * delta, axis=-1)
    return distance.min(axis=1)


def _static_valid(paths, obstacles, world):
    valid = ((paths >= 0) & (paths <= world.map_size_m)).all(axis=(1, 2))
    if len(obstacles):
        valid &= (_distances(paths, obstacles[:, :2]) >=
                  obstacles[None, :, 2] + world.obstacle_clearance_m + 1e-9).all(axis=1)
    return valid


def _peer_valid(paths, others, world):
    valid = np.ones(len(paths), dtype=bool)
    for other in others:
        valid &= _distances(paths - other[None], [[0., 0.]])[:, 0] >= world.safety_distance_m - 1e-9
    return valid


def _tangent(normal, previous, nominal):
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    tangent = np.array([-normal[1], normal[0]])
    # Keep the established side; an exactly symmetric encounter uses right-hand passing.
    component = previous @ tangent
    if abs(component) < 1e-8:
        component = nominal @ tangent
    return tangent * (-1 if component < -1e-8 else 1) * np.linalg.norm(nominal)


def _obstacle_tangent(position, previous, nominal, obstacle, obstacles, world):
    guide = _tangent(position - obstacle[:2], previous, nominal)
    axis = int(np.argmax(np.abs(guide)))
    direction = 1 if guide[axis] >= 0 else -1
    radius = obstacle[2] + world.obstacle_clearance_m + 1e-6
    points = np.repeat(obstacle[None, :2], 2, axis=0)
    points[0, axis] += direction * radius
    points[1, axis] -= direction * radius
    # Prefer the other side if the chosen side is sealed by a boundary/neighbor.
    open_sides = _static_valid(np.repeat(points[:, None], 2, axis=1), obstacles, world)
    return -guide if not open_sides[0] and open_sides[1] else guide


def _guidance(position, previous, nominal, path, obstacles, other_paths, world):
    if len(obstacles):
        gaps = _distances(path[None], obstacles[:, :2])[0] - obstacles[:, 2] - world.obstacle_clearance_m
        threatening = np.flatnonzero(gaps < 1e-9)
        if len(threatening):
            index = min(threatening, key=lambda k: np.linalg.norm(position - obstacles[k, :2]))
            return _obstacle_tangent(position, previous, nominal, obstacles[index],
                                     obstacles, world), "obstacle"
    outside = ((path < 0) | (path > world.map_size_m)).any(axis=0)
    if outside.any():
        guide = nominal.copy()
        guide[outside] = 0  # Brake the outward component, keep existing motion along the edge.
        if np.linalg.norm(guide) <= 1e-8 or guide @ previous < -1e-8:
            # Keep the established side when an outward command erases/reverses
            # edge motion; otherwise turns can settle into a small corner loop.
            # Both wall tangents still need the same braking certificates.
            step = world.max_accel_mps2 * world.dt_s
            stop_guide = None
            for axis in range(2):
                if not outside[axis]:
                    continue
                normal = np.zeros(2)
                normal[axis] = -1 if (path[:, axis] < 0).any() else 1
                tangent = _tangent(normal, previous, nominal)
                for alternative in (tangent, -tangent):
                    aimed = reachable_velocities(
                        (previous + np.clip(alternative - previous, -step, step))[None],
                        previous[None], world)[0]
                    certificate = braking_paths(position[None], aimed[None], world)
                    if (_static_valid(certificate, obstacles, world)[0]
                            and _peer_valid(certificate, other_paths, world)[0]):
                        if np.linalg.norm(aimed) > 1e-8:
                            return alternative, "boundary"
                        if stop_guide is None:
                            stop_guide = alternative
            if stop_guide is not None:
                # A bounded reversal can require one stopped step before turning.
                return stop_guide, "boundary"
        return guide, "boundary"
    for other in other_paths:
        if not _peer_valid(path[None], [other], world)[0]:
            return _tangent(position - other[0], previous, nominal), "peer"
    return nominal, None


def filter_velocities(positions, velocities, nominal, obstacles, world, disabled):
    """Choose reachable velocities whose joint candidate/braking paths are safe.

    Sequential reservations are deterministic in UAV ID order. This favors
    earlier IDs in dense conflicts; no global progress guarantee is claimed.
    The exact next-step guard in motion.py handles an infeasible initial state
    or a peer suddenly immobilized by its battery.
    """
    nominal_paths = braking_paths(positions, nominal, world)
    step = world.max_accel_mps2 * world.dt_s
    static = (_static_valid(nominal_paths, obstacles, world)
              & (np.abs(nominal - velocities) <= step + 1e-9).all(axis=1))
    pair_safe = all(_peer_valid(nominal_paths[i:i + 1], nominal_paths[:i], world)[0]
                    for i in range(len(positions)))
    counts = {f"shield_{name}": 0 for name in (
        "interventions", "obstacle_interventions", "peer_interventions",
        "boundary_interventions", "infeasible")}
    if static.all() and pair_safe:
        return nominal, counts
    step = world.max_accel_mps2 * world.dt_s
    chosen = _brake(velocities, step)
    chosen[disabled] = 0
    paths = braking_paths(positions, chosen, world)
    # ponytail: 25 acceleration samples per conflict; refine only if measured
    # dense-scene failures require a continuous optimizer.
    grid = np.array([(x, y) for x in (-1., -.5, 0., .5, 1.)
                     for y in (-1., -.5, 0., .5, 1.)]) * step
    for i, position in enumerate(positions):
        if disabled[i]:
            continue
        others = np.concatenate((paths[:i], paths[i + 1:]))
        path = nominal_paths[i]
        if static[i] and _peer_valid(path[None], others, world)[0]:
            chosen[i], paths[i] = nominal[i], path
            continue
        guide, kind = _guidance(position, velocities[i], nominal[i], path,
                                obstacles, others, world)
        aimed = velocities[i] + np.clip(guide - velocities[i], -step, step)
        candidates = reachable_velocities(
            np.vstack((aimed, nominal[i], chosen[i], velocities[i] + grid)), velocities[i], world)
        candidate_paths = braking_paths(np.repeat(position[None], len(candidates), axis=0),
                                        candidates, world)
        valid = (_static_valid(candidate_paths, obstacles, world)
                 & _peer_valid(candidate_paths, others, world)
                 & (np.abs(candidates - velocities[i]) <= step + 1e-9).all(axis=1))
        if valid.any():
            scores = np.square(candidates - guide).sum(1) + .01 * np.square(candidates - nominal[i]).sum(1)
            scores[~valid] = np.inf
            index = int(np.argmin(scores))
            chosen[i], paths[i] = candidates[index], candidate_paths[index]
        else:
            counts["shield_infeasible"] += 1
        if not np.allclose(chosen[i], nominal[i], rtol=0, atol=1e-9):
            counts["shield_interventions"] += 1
            if kind:
                counts[f"shield_{kind}_interventions"] += 1
    return chosen, counts
