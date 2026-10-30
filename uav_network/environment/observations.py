"""Local FOV observations; privileged geometry is confined to critic state."""
import numpy as np

from .sensing import visible_target_ids
from ..communication.link_budget import nominal_required_power_w, power_fraction

# Local-radio link metadata and report sensing without remote GCS receipts.
OBSERVATION_VERSION = 3

def observation_version(config):
    return 4 if config.motion_dim == 3 else OBSERVATION_VERSION


def observation_dim(config):
    w, s, n = config.world, config.sensor, config.network
    link = 3 * w.num_uavs if n.link_state_observation else 0
    extra = (2 + s.visible_obstacle_slots + 2*(w.num_uavs-1) + s.belief_patch_cells**2) if config.motion_dim==3 else 0
    return (extra + 10 + 2 * w.num_targets + 4 * (s.visible_target_slots + s.visible_obstacle_slots)
            + 6 * (w.num_uavs - 1) + link + w.num_uavs + 1
            + s.coverage_grid_cells ** 2)


def state_dim(config):
    w = config.world
    extra = (2*w.num_uavs+w.num_obstacles+w.num_uavs*config.sensor.belief_patch_cells**2) if config.motion_dim==3 else 0
    return (extra + 7 * w.num_uavs + 3 * w.num_targets + 3 * w.num_obstacles
            + 2 * w.num_uavs * w.num_targets + 1
            + w.num_uavs * config.sensor.coverage_grid_cells ** 2)


def _nominal_required(env, source, destination):
    count = env.config.world.num_uavs
    target = env.gcs if destination == count else env.positions[destination]
    return nominal_required_power_w(env.positions[source], target, env.obstacles, env.config.network,
                                    env.config.sensor.altitude_m, destination == count)


def _power_feasible(env, source, destination):
    return (not env.config.network.power_limited_admission
            or _nominal_required(env, source, destination) <= env.config.network.tx_power_max_w)


def link_state_features(env, source):
    w, n = env.config.world, env.config.network
    count = w.num_uavs
    source_gcs = float(np.linalg.norm(env.positions[source] - env.gcs))
    diagonal = np.sqrt(2.0) * w.map_size_m
    values = []
    for destination in [j for j in range(count) if j != source] + [count]:
        target = env.gcs if destination == count else env.positions[destination]
        reach = n.gcs_range_m if destination == count else n.peer_range_m
        observable = (env.active[source]
                      and (destination == count or env.active[destination])
                      and np.linalg.norm(target - env.positions[source]) <= reach)
        if not observable:
            values.extend((0.0, 0.0, 0.0))
            continue
        required = _nominal_required(env, source, destination)
        destination_gcs = 0.0 if destination == count else float(np.linalg.norm(target - env.gcs))
        progress = np.clip((source_gcs - destination_gcs) / diagonal, -1.0, 1.0)
        free = 1.0 if destination == count else np.clip(
            (n.buffer_bytes - env.reports.bytes_used(destination)) / n.buffer_bytes, 0.0, 1.0)
        values.extend((power_fraction(required, n), progress, free))
    return np.asarray(values, dtype=float)


def destination_mask(env):
    """Ideal beacons/admission handshake; no SINR or hidden target coordinates.

    The receiver advertises whether it can accept any report held by the sender.
    This metadata handshake is currently reliable and has no separate airtime cost.
    """
    n, count = env.config.network, env.config.world.num_uavs
    masks = np.zeros((count, count + 1), dtype=bool)
    masks[:, 0] = True
    for i in range(count):
        if not n.enabled or not env.active[i] or not env.reports.buffers[i]:
            continue
        for slot, j in enumerate(k for k in range(count) if k != i):
            masks[i, slot + 1] = (env.active[j]
                                  and np.linalg.norm(env.positions[j] - env.positions[i]) <= n.peer_range_m
                                  and _power_feasible(env, i, j)
                                  and (not n.receiver_admission_mask or any(
                                      env.reports.can_forward(i, j, report, env.time_s)
                                      for report in env.reports.buffers[i].values())))
        masks[i, count] = (np.linalg.norm(env.gcs - env.positions[i]) <= n.gcs_range_m
                           and _power_feasible(env, i, count)
                           and (not n.receiver_admission_mask or any(
                               env.reports.can_forward(i, count, report, env.time_s)
                               for report in env.reports.buffers[i].values())))
    return masks


def report_status(env):
    """Remaining TTL, zero for no custody, -1 for local loss needing a fresh view."""
    n = env.config.network
    status = np.zeros_like(env.reports.known, dtype=float)
    if n.rediscovery_recovery:
        status[env.reports.recovery_pending] = -1
    for i, buffer in enumerate(env.reports.buffers):
        for target_id, report in buffer.items():
            status[i, target_id] = max(0, 1 - (env.time_s - report.created_s) / n.report_ttl_s)
    return status


def local_observations(env):
    if env.config.motion_dim == 3:
        from .xyz_observations import xyz_features
        return xyz_features(env)
    c, w, n, s = env.config, env.config.world, env.config.network, env.config.sensor
    rows = []
    masks = destination_mask(env)
    reports = report_status(env)
    for i, position in enumerate(env.positions):
        own = [*(position / w.map_size_m), *(env.velocities[i] / w.max_speed_mps),
               *((env.gcs - position) / w.map_size_m), env.steps / w.max_steps,
               env.reports.bytes_used(i) / n.buffer_bytes, env.battery_fraction[i], env.active[i]]
        age = reports[i]
        targets = np.zeros((s.visible_target_slots, 4))
        ids = visible_target_ids(position, env.targets, env.obstacles, s) if env.active[i] else []
        for slot, target_id in enumerate(ids[:s.visible_target_slots]):
            targets[slot] = [*((env.targets[target_id] - position) / w.map_size_m),
                             env.reports.known[i, target_id], 1]
        obstacles = np.zeros((s.visible_obstacle_slots, 4))
        order = sorted(range(len(env.obstacles)),
                       key=lambda j: np.linalg.norm(env.obstacles[j, :2] - position))
        visible = [j for j in order if np.linalg.norm(env.obstacles[j, :2] - position)
                   <= s.footprint_radius_m + env.obstacles[j, 2] and env.active[i]]
        for slot, j in enumerate(visible[:s.visible_obstacle_slots]):
            o = env.obstacles[j]
            obstacles[slot] = [*((o[:2] - position) / w.map_size_m), o[2] / w.map_size_m, 1]
        peers = np.zeros((w.num_uavs - 1, 6))
        for slot, j in enumerate(k for k in range(w.num_uavs) if k != i):
            if env.active[i] and np.linalg.norm(env.positions[j] - position) <= n.peer_range_m:
                peers[slot] = [*((env.positions[j] - position) / w.map_size_m),
                               *(env.velocities[j] / w.max_speed_mps), env.battery_fraction[j], env.active[j]]
        link = link_state_features(env, i) if n.link_state_observation else np.empty(0)
        rows.append(np.r_[own, env.reports.known[i], age, targets.ravel(),
                          obstacles.ravel(), peers.ravel(), link,
                          env.coverage[i] if s.coverage_grid_cells else np.empty(0), masks[i]])
    return np.asarray(rows, dtype=np.float32)


def centralized_state(env):
    w, n = env.config.world, env.config.network
    ages = report_status(env)
    inflight = env.network.inflight_counts() if env.network else np.zeros(w.num_uavs)
    extra = np.empty(0)
    if env.config.motion_dim == 3:
        from .bayesian import belief_patch
        extra = np.r_[env.positions[:,2]/w.altitude_max_m, env.velocities[:,2]/w.max_speed_mps,
                      env.obstacle_heights/w.altitude_max_m,
                      np.concatenate([belief_patch(env,i) for i in range(w.num_uavs)])]
    return np.r_[env.positions[:,:2].ravel() / w.map_size_m,
                 env.velocities[:,:2].ravel() / w.max_speed_mps,
                 env.targets.ravel() / w.map_size_m,
                 env.obstacles.ravel() / w.map_size_m,
                 env.reports.known.ravel(), ages.ravel(), env.reports.delivered,
                 inflight / n.max_queue_packets, env.battery_fraction, env.active,
                 env.steps / w.max_steps,
                 env.coverage.ravel() if env.config.sensor.coverage_grid_cells else np.empty(0), extra].astype(np.float32)
