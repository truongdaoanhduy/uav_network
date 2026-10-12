"""Synchronous XY motion with continuous obstacle and peer safety tests."""
import numpy as np

from .geometry import segment_distance, segment_hits_circle


def move_uavs(positions, velocities, acceleration, obstacles, config):
    w = config.world
    proposed_velocity = velocities + acceleration * w.max_accel_mps2 * w.dt_s
    speed = np.linalg.norm(proposed_velocity, axis=1, keepdims=True)
    proposed_velocity *= np.minimum(1, w.max_speed_mps / np.maximum(speed, 1e-12))
    unbounded = positions + proposed_velocity * w.dt_s
    proposed = np.clip(unbounded, 0, w.map_size_m)
    clipped = np.any(proposed != unbounded, axis=1)
    blocked = np.zeros(len(positions), dtype=bool)
    for i, (start, end) in enumerate(zip(positions, proposed)):
        if any(segment_hits_circle(start, end, o[:2], o[2] + w.obstacle_clearance_m)
               for o in obstacles):
            blocked[i] = True
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
    return resolved, actual_velocity, int(blocked.sum()), int(clipped.sum())
