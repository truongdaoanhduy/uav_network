"""Local FOV observations; privileged geometry is confined to critic state."""
import numpy as np

from .sensing import visible_target_ids


def observation_dim(config):
    w, s = config.world, config.sensor
    return (10 + 2 * w.num_targets + 4 * (s.visible_target_slots + s.visible_obstacle_slots)
            + 6 * (w.num_uavs - 1) + w.num_uavs + 1)


def state_dim(config):
    w = config.world
    return 7 * w.num_uavs + 3 * w.num_targets + 3 * w.num_obstacles + 2 * w.num_uavs * w.num_targets + 1


def destination_mask(env):
    """Ideal control-plane beacons: range/liveness; no hidden target geometry."""
    n, count = env.config.network, env.config.world.num_uavs
    masks = np.zeros((count, count + 1), dtype=bool)
    masks[:, 0] = True
    for i in range(count):
        if not n.enabled or not env.active[i] or not env.reports.buffers[i]:
            continue
        for slot, j in enumerate(k for k in range(count) if k != i):
            masks[i, slot + 1] = (env.active[j]
                                  and np.linalg.norm(env.positions[j] - env.positions[i]) <= n.peer_range_m)
        masks[i, count] = np.linalg.norm(env.gcs - env.positions[i]) <= n.gcs_range_m
    return masks


def local_observations(env):
    c, w, n, s = env.config, env.config.world, env.config.network, env.config.sensor
    rows = []
    masks = destination_mask(env)
    for i, position in enumerate(env.positions):
        own = [*(position / w.map_size_m), *(env.velocities[i] / w.max_speed_mps),
               *((env.gcs - position) / w.map_size_m), env.steps / w.max_steps,
               env.reports.bytes_used(i) / n.buffer_bytes, env.battery_fraction[i], env.active[i]]
        age = np.zeros(w.num_targets)
        for target_id, report in env.reports.buffers[i].items():
            age[target_id] = max(0, 1 - (env.time_s - report.created_s) / n.report_ttl_s)
        targets = np.zeros((s.visible_target_slots, 4))
        ids = visible_target_ids(position, env.targets, env.obstacles, s) if env.active[i] else []
        for slot, target_id in enumerate(ids[:s.visible_target_slots]):
            targets[slot] = [*((env.targets[target_id] - position) / w.map_size_m),
                             env.reports.known[i, target_id], 1]
        obstacles = np.zeros((s.visible_obstacle_slots, 4))
        order = sorted(range(len(env.obstacles)),
                       key=lambda j: np.linalg.norm(env.obstacles[j, :2] - position))
        visible = [j for j in order if np.linalg.norm(env.obstacles[j, :2] - position)
                   <= s.footprint_radius_m + env.obstacles[j, 2]]
        for slot, j in enumerate(visible[:s.visible_obstacle_slots]):
            o = env.obstacles[j]
            obstacles[slot] = [*((o[:2] - position) / w.map_size_m), o[2] / w.map_size_m, 1]
        peers = np.zeros((w.num_uavs - 1, 6))
        for slot, j in enumerate(k for k in range(w.num_uavs) if k != i):
            if np.linalg.norm(env.positions[j] - position) <= n.peer_range_m:
                peers[slot] = [*((env.positions[j] - position) / w.map_size_m),
                               *(env.velocities[j] / w.max_speed_mps), env.battery_fraction[j], env.active[j]]
        rows.append(np.r_[own, env.reports.known[i], age, targets.ravel(),
                          obstacles.ravel(), peers.ravel(), masks[i]])
    return np.asarray(rows, dtype=np.float32)


def centralized_state(env):
    w, n = env.config.world, env.config.network
    ages = np.zeros((w.num_uavs, w.num_targets))
    for i, buffer in enumerate(env.reports.buffers):
        for target_id, report in buffer.items():
            ages[i, target_id] = max(0, 1 - (env.time_s - report.created_s) / n.report_ttl_s)
    inflight = env.network.inflight_counts() if env.network else np.zeros(w.num_uavs)
    return np.r_[env.positions.ravel() / w.map_size_m,
                 env.velocities.ravel() / w.max_speed_mps,
                 env.targets.ravel() / w.map_size_m,
                 env.obstacles.ravel() / w.map_size_m,
                 env.reports.known.ravel(), ages.ravel(), env.reports.delivered,
                 inflight / n.max_queue_packets, env.battery_fraction, env.active,
                 env.steps / w.max_steps].astype(np.float32)
