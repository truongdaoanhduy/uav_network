"""Deterministic multi-step stress tests for the production XYZ CBF filter."""
import numpy as np

from uav_network.configuration import Config
from uav_network.environment.geometry import segment_distance
from uav_network.environment.xyz_motion import move_xyz


def _config():
    c = Config()
    w = c.world
    w.motion_dimensions = 3
    w.motion_safety = "cbf"
    w.max_speed_mps = 20.0
    w.max_accel_mps2 = 2.0
    w.safety_distance_m = 15.0
    w.obstacle_clearance_m = 10.0
    w.altitude_min_m = 0.0
    w.altitude_max_m = 150.0
    w.map_size_m = 1000.0
    w.dt_s = 1.0
    return c


def _roll(positions, velocities, actions, steps, obstacles=(), heights=()):
    c = _config()
    p = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    a = np.asarray(actions, dtype=float)
    obs = np.asarray(obstacles, dtype=float).reshape(-1, 3)
    heights = np.asarray(heights, dtype=float)
    records = []
    for _ in range(steps):
        before = p.copy()
        diagnostics = {}
        p, v, blocked, clipped = move_xyz(
            p, v, a, obs, heights, c, diagnostics=diagnostics)
        assert np.isfinite(p).all() and np.isfinite(v).all()
        assert np.isfinite(diagnostics["shield_correction"])
        records.append((before.copy(), p.copy(), v.copy(), blocked, clipped, diagnostics))
    return c, p, v, records


def test_hostile_policy_never_crosses_protected_cylinder_over_many_steps():
    c, p, v, records = _roll(
        [[350, 500, 25]], [[20, 0, 0]], [[1, 0, 0]], 40,
        [[500, 500, 10]], [40])
    required = 10 + c.world.obstacle_clearance_m
    clearances = [segment_distance(before[0, :2], after[0, :2], [500, 500]) - required
                  for before, after, *_ in records]
    assert min(clearances) >= -1e-9
    assert sum(r[5]["shield_interventions"] for r in records) == 40
    assert sum(r[4] for r in records) == 0
    assert p[0, 0] < 500 - required + 1e-9
    assert np.linalg.norm(v[0]) == 0


def test_hostile_head_on_peers_never_violate_separation_over_many_steps():
    c, _, v, records = _roll(
        [[350, 500, 80], [650, 500, 80]],
        [[20, 0, 0], [-20, 0, 0]],
        [[1, 0, 0], [-1, 0, 0]], 30)
    separations = [segment_distance(before[0] - before[1], after[0] - after[1], [0, 0, 0])
                   for before, after, *_ in records]
    assert min(separations) >= c.world.safety_distance_m - 1e-9
    assert sum(r[3] for r in records) == 0
    assert np.linalg.norm(v, axis=1).max() < 1e-9


def test_hostile_boundary_command_stays_inside_without_clipping():
    _, p, v, records = _roll(
        [[120, 500, 80]], [[-20, 0, 0]], [[-1, 0, 0]], 25)
    assert min(after[0, 0] for _, after, *_ in records) >= 0
    assert sum(r[4] for r in records) == 0
    assert p[0, 0] == 0
    assert np.linalg.norm(v[0]) == 0


def test_safe_parallel_motion_has_zero_false_interventions_over_many_steps():
    _, _, v, records = _roll(
        [[300, 500, 80], [320, 500, 80]],
        [[10, 0, 0], [10, 0, 0]],
        [[0, 0, 0], [0, 0, 0]], 20)
    assert sum(r[5]["shield_interventions"] for r in records) == 0
    assert sum(r[3] for r in records) == 0
    assert sum(r[4] for r in records) == 0
    np.testing.assert_allclose(v[:, 0], [10, 10], atol=1e-12)
