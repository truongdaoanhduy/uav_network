"""Synchronous XY motion with continuous obstacle and peer safety tests."""
import numpy as np

from .avoidance import filter_velocities, reachable_velocities
from .geometry import segment_distance, segment_hits_circle


def move_uavs(positions, velocities, acceleration, obstacles, config, disabled=None, diagnostics=None):
    w = config.world
    disabled = np.zeros(len(positions), dtype=bool) if disabled is None else disabled
    proposed_velocity = velocities + acceleration * w.max_accel_mps2 * w.dt_s
    if w.motion_safety == "dynamic_window":
        proposed_velocity = reachable_velocities(proposed_velocity, velocities, w)
    else:
        speed = np.linalg.norm(proposed_velocity, axis=1, keepdims=True)
        proposed_velocity *= np.minimum(1, w.max_speed_mps / np.maximum(speed, 1e-12))
    proposed_velocity[disabled] = 0
    nominal_velocity = proposed_velocity.copy()
    shield = {f"shield_{name}": 0 for name in ("interventions", "obstacle_interventions",
              "peer_interventions", "boundary_interventions", "infeasible")}
    if w.motion_safety == "dynamic_window":
        proposed_velocity, shield = filter_velocities(positions, velocities, proposed_velocity,
                                                      obstacles, w, disabled)
    active = ~disabled
    shield_correction = (float(np.linalg.norm(proposed_velocity[active] - nominal_velocity[active], axis=1).mean()
                               / w.max_speed_mps) if active.any() else 0.0)
    unbounded = positions + proposed_velocity * w.dt_s
    proposed = np.clip(unbounded, 0, w.map_size_m)
    proposed[disabled] = positions[disabled]
    clipped = np.any(proposed != unbounded, axis=1)
    blocked = np.zeros(len(positions), dtype=bool)
    for i, (start, end) in enumerate(zip(positions, proposed)):
        if any(segment_hits_circle(start, end, o[:2], o[2] + w.obstacle_clearance_m)
               for o in obstacles):
            blocked[i] = True
    obstacle_blocked = blocked.copy()
    # Re-check after rejection: a moving peer can hit a UAV which just stopped.
    for _ in range(len(positions) + 1):
        resolved = np.where(blocked[:, None], positions, proposed)
        previous = blocked.copy()
        for i in range(len(positions)):
            for j in range(i):
                if segment_distance(positions[i] - positions[j], resolved[i] - resolved[j],
                                    [0, 0]) < w.safety_distance_m - 1e-9:
                    blocked[i] = blocked[j] = True
        if np.array_equal(previous, blocked):
            break
    resolved = np.where(blocked[:, None], positions, proposed)
    actual_velocity = (resolved - positions) / w.dt_s
    if diagnostics is not None:
        diagnostics.update(shield)
        diagnostics["shield_correction"] = shield_correction
        diagnostics.update(obstacle_blocks=int((obstacle_blocked & ~disabled).sum()),
                           peer_blocks=int((blocked & ~obstacle_blocked & ~disabled).sum()),
                           commanded_stalls=int(((np.linalg.norm(acceleration, axis=1) > .1)
                               & (np.linalg.norm(actual_velocity, axis=1) < 1e-8) & ~disabled).sum()))
    return resolved, actual_velocity, int((blocked & ~disabled).sum()), int((clipped & ~disabled).sum())
