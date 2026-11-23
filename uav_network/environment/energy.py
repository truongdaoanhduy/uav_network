"""A discrete-step propulsion budget based on realized, collision-safe motion."""
import numpy as np

from .motion import move_uavs


def propulsion_cost(previous_velocity, velocity, config):
    e, dt = config.energy, config.world.dt_s
    acceleration = (velocity - previous_velocity) / dt
    if e.propulsion_model == 'zeng2019':
        rotor_p0=(e.rotor_profile_drag_coefficient/8 * e.rotor_air_density_kg_m3 *
                  e.rotor_solidity * e.rotor_disc_area_m2 *
                  e.rotor_angular_velocity_rad_s**3 * e.rotor_radius_m**3)
        rotor_pi=((1+e.rotor_induced_correction_factor)*e.rotor_weight_n**1.5 /
                  np.sqrt(2*e.rotor_air_density_kg_m3*e.rotor_disc_area_m2))
        u_tip=e.rotor_angular_velocity_rad_s*e.rotor_radius_m
        v0=np.sqrt(e.rotor_weight_n/(2*e.rotor_air_density_kg_m3*e.rotor_disc_area_m2))
        parasite=.5*e.rotor_fuselage_drag_ratio*e.rotor_air_density_kg_m3 * e.rotor_solidity*e.rotor_disc_area_m2
        xy=np.asarray(velocity)[:,:2]
        speed=np.linalg.norm(xy,axis=1)
        ratio=np.square(speed)/(2*v0*v0)
        induced=np.sqrt(np.maximum(np.sqrt(1+np.square(ratio))-ratio,0))
        power=(rotor_p0*(1+3*np.square(speed)/(u_tip*u_tip)) +
               rotor_pi*induced + parasite*speed**3 +
               e.rotor_weight_n*np.maximum(np.asarray(velocity)[:,2],0))
    else:
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
