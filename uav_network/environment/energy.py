"""A discrete-step propulsion budget based on realized, collision-safe motion."""
import numpy as np

from .motion import move_uavs


def propulsion_cost(previous_velocity, velocity, config):
    e, dt = config.energy, config.world.dt_s
    acceleration = (velocity - previous_velocity) / dt
    power = (e.idle_power_w + e.hover_power_w
             + e.speed_squared_coefficient * np.square(velocity).sum(axis=1)
             + e.acceleration_squared_coefficient * np.square(acceleration).sum(axis=1))
    return power * dt


def funded_motion(positions, velocities, acceleration, obstacles, battery, config, diagnostics=None, obstacle_heights=None):
    """Stop unfunded moves, then recheck safety against every stationary peer.

    An unfunded full step consumes the remaining charge without displacement.
    This deliberately conservative discretization avoids flying on energy debt.
    """
    alive = battery > 0 if config.energy.enabled else np.ones(len(battery), dtype=bool)
    disabled = ~alive
    for _ in range(len(battery) + 1):
        if config.motion_dim == 3:
            from .xyz_motion import move_xyz
            result = move_xyz(positions, velocities, acceleration, obstacles, obstacle_heights,
                              config, disabled, diagnostics)
        else:
            result = move_uavs(positions, velocities, acceleration, obstacles, config, disabled, diagnostics)
        cost = propulsion_cost(velocities, result[1], config)
        unaffordable = alive & ~disabled & (cost > battery) if config.energy.enabled else disabled
        if not unaffordable.any():
            break
        disabled |= unaffordable
    cost[~alive] = 0
    exhausted = disabled & alive
    if config.energy.enabled:
        cost[exhausted] = battery[exhausted]
        battery[:] = np.maximum(0, battery - cost)
    return (*result, cost)
