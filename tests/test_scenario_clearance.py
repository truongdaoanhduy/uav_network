"""Placement must not admit points forbidden by motion or camera occlusion."""
import importlib.util

import numpy as np
import pytest
import torch

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.geometry import segment_hits_circle


def boundary_scene(kind, gap):
    c = Config()
    c.world.num_uavs = c.world.num_targets = c.world.num_obstacles = 1
    c.network.enabled = False
    scene = dict(positions=[[100, 100]], targets=[[800, 800]],
                 obstacles=[[500, 500, 50]])
    radius = 50 + (c.world.obstacle_clearance_m if kind == "uav" else 0)
    point = [500 - radius - gap, 500]
    if kind == "uav":
        scene["positions"] = [point]
    elif kind == "target":
        scene["targets"] = [point]
    else:
        c.world.gcs_position_m = point
    return c, scene


@pytest.mark.parametrize("kind", ["uav", "target", "gcs"])
@pytest.mark.parametrize("gap", [0.0, 0.5e-9])
def test_explicit_scenario_rejects_forbidden_obstacle_boundary(kind, gap):
    c, scene = boundary_scene(kind, gap)
    env = SwarmSearchEnv(c)
    try:
        with pytest.raises(ValueError, match="obstacle"):
            env.reset(seed=44, options=scene)
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["uav", "target", "gcs"])
def test_explicit_scenario_accepts_point_clear_of_geometry_tolerance(kind):
    c, scene = boundary_scene(kind, 2e-9)
    env = SwarmSearchEnv(c)
    try:
        env.reset(seed=44, options=scene)
    finally:
        env.close()


if importlib.util.find_spec("warp"):
    import warp as wp
    from uav_network.environment.gpu.state import WorldState
    from uav_network.environment.gpu.world import _outside

    @wp.kernel
    def placement_allowed(s: WorldState, points: wp.array(dtype=wp.vec2d),
                          clearance: wp.float64, result: wp.array(dtype=wp.int32)):
        i = wp.tid()
        result[i] = wp.int32(_outside(s, 0, points[i], 1, clearance))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("clearance", [0.0, 10.0])
def test_warp_placement_agrees_with_stationary_segment_geometry(device, clearance):
    wp = pytest.importorskip("warp")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    from uav_network.environment.gpu.state import allocate_state

    c = Config()
    c.world.num_obstacles = 1
    s, _ = allocate_state(c, num_envs=1, device=device)
    s.obstacles.assign(np.array([[[500, 500, 50]]], dtype=float))
    points = np.array([[450 - clearance - gap, 500] for gap in
                       (-1e-9, 0.0, 0.5e-9, 2e-9)], dtype=float)
    expected = [not segment_hits_circle(point, point, [500, 500], 50 + clearance)
                for point in points]
    assert expected == [False, False, False, True]
    result = wp.zeros(4, dtype=wp.int32, device=device)
    wp.launch(placement_allowed, dim=4,
              inputs=[s, wp.array(points, dtype=wp.vec2d, device=device),
                      clearance, result], device=device)
    assert result.numpy().astype(bool).tolist() == expected
