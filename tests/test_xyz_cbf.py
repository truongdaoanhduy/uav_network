import numpy as np

from uav_network.configuration import Config
from uav_network.environment.xyz_motion import move_xyz


def _config():
    c = Config()
    c.world.motion_dimensions = 3
    c.world.motion_safety = "cbf"
    c.world.max_speed_mps = 20.0
    c.world.max_accel_mps2 = 2.0
    c.world.safety_distance_m = 15.0
    c.world.obstacle_clearance_m = 10.0
    c.world.altitude_min_m = 0.0
    c.world.altitude_max_m = 150.0
    c.world.map_size_m = 1000.0
    c.world.dt_s = 1.0
    return c


def _move(positions, velocities, actions, obstacles=(), heights=()):
    diagnostics = {}
    result = move_xyz(
        np.asarray(positions, dtype=float),
        np.asarray(velocities, dtype=float),
        np.asarray(actions, dtype=float),
        np.asarray(obstacles, dtype=float).reshape(-1, 3),
        np.asarray(heights, dtype=float),
        _config(),
        diagnostics=diagnostics,
    )
    return result, diagnostics


def test_cbf_brakes_before_static_cylinder_collision():
    (position, velocity, blocked, _), diagnostics = _move(
        [[350.0, 500.0, 25.0]],
        [[20.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[500.0, 500.0, 10.0]],
        [40.0],
    )
    assert blocked == 0
    assert velocity[0, 0] < 20.0
    assert position[0, 0] < 370.0
    assert diagnostics["shield_interventions"] == 1


def test_cbf_splits_head_on_peer_avoidance_across_both_uavs():
    (position, velocity, blocked, _), diagnostics = _move(
        [[350.0, 500.0, 80.0], [650.0, 500.0, 80.0]],
        [[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert blocked == 0
    assert velocity[0, 0] < 20.0
    assert velocity[1, 0] > -20.0
    assert np.isclose(abs(velocity[0, 0]), abs(velocity[1, 0]), atol=1e-9)
    assert diagnostics["shield_interventions"] == 2


def test_cbf_corrects_outward_boundary_motion_before_clipping():
    (position, velocity, blocked, clipped), diagnostics = _move(
        [[120.0, 500.0, 80.0]],
        [[-20.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    )
    assert blocked == 0
    assert clipped == 0
    assert velocity[0, 0] > -20.0
    assert diagnostics["shield_interventions"] == 1


def test_cbf_passes_through_safe_parallel_peer_motion():
    (_, velocity, blocked, clipped), diagnostics = _move(
        [[300.0, 500.0, 80.0], [320.0, 500.0, 80.0]],
        [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert blocked == 0
    assert clipped == 0
    np.testing.assert_allclose(velocity[:, 0], [10.0, 10.0], atol=1e-12)
    assert diagnostics["shield_interventions"] == 0


def test_cbf_outputs_remain_finite_near_constraint_boundaries():
    (_, velocity, _, _), diagnostics = _move(
        [[20.0, 20.0, 2.0], [40.0, 20.0, 2.0]],
        [[-5.0, -5.0, -1.0], [-5.0, 5.0, -1.0]],
        [[-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]],
        [[30.0, 30.0, 5.0]],
        [10.0],
    )
    assert np.isfinite(velocity).all()
    assert np.isfinite(diagnostics["shield_correction"])


def test_cbf_uses_simulator_integration_when_protecting_near_boundary():
    (_, velocity, blocked, clipped), diagnostics = _move(
        [[1.5, 500.0, 80.0]],
        [[0.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0]],
    )
    assert blocked == 0
    assert clipped == 0
    assert velocity[0, 0] >= -1.5 - 1e-9
    assert diagnostics["shield_interventions"] == 1
