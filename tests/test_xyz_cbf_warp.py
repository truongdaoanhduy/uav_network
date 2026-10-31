import numpy as np
import pytest
import torch
import warp as wp

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv, Action
from uav_network.environment.gpu.vector import WarpVectorEnv


def _config(n=1, obstacles=0):
    c = Config()
    c.world.motion_dimensions = 3
    c.world.motion_safety = "cbf"
    c.world.num_uavs = n
    c.world.num_targets = 1
    c.world.num_obstacles = obstacles
    c.sensor.model = "bayesian"
    c.network.enabled = False
    c.energy.enabled = False
    c.runtime.num_envs = 1
    c.runtime.device = "cpu"
    c.runtime.environment_backend = "warp_packet"
    return c


def _run_case(positions, velocities, actions, obstacles=()):
    c = _config(len(positions), len(obstacles))
    native = SwarmSearchEnv(c)
    warp = WarpVectorEnv(c, device="cpu")
    try:
        native.reset(seed=44, options={
            "positions": positions,
            "targets": [[950.0, 950.0]],
            "obstacles": obstacles,
        })
        native.velocities[:] = np.asarray(velocities, dtype=float)

        warp.reset(seed=44)
        wp.to_torch(warp.s.positions)[0] = torch.tensor(np.asarray(positions)[:, :2], dtype=torch.float64)
        wp.to_torch(warp.s.altitudes)[0] = torch.tensor(np.asarray(positions)[:, 2], dtype=torch.float64)
        wp.to_torch(warp.s.velocities)[0] = torch.tensor(np.asarray(velocities)[:, :2], dtype=torch.float64)
        wp.to_torch(warp.s.vertical_velocity)[0] = torch.tensor(np.asarray(velocities)[:, 2], dtype=torch.float64)
        wp.to_torch(warp.s.targets)[0] = torch.tensor([[950.0, 950.0]], dtype=torch.float64)
        if obstacles:
            obs = np.asarray(obstacles, dtype=float)
            wp.to_torch(warp.s.obstacles)[0] = torch.tensor(obs[:, :3], dtype=torch.float64)
            wp.to_torch(warp.s.obstacle_heights)[0] = torch.tensor(obs[:, 3], dtype=torch.float64)

        action = Action(np.asarray(actions, dtype=np.float32), np.zeros(len(positions), dtype=np.int64))
        _, native_reward, _, _, native_info = native.step(action)
        warp.step({"continuous": action.continuous[None], "destination": action.destination[None]})
        warp_positions = np.column_stack((warp.s.positions.numpy()[0], warp.s.altitudes.numpy()[0]))
        warp_velocities = np.column_stack((warp.s.velocities.numpy()[0], warp.s.vertical_velocity.numpy()[0]))

        np.testing.assert_allclose(warp_positions, native.positions, rtol=0, atol=1e-8)
        np.testing.assert_allclose(warp_velocities, native.velocities, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(warp.s.motion_counts.numpy()[0, :8],
                                      list(native.motion_counts.values())[:8])
        assert float(warp.s.shield_correction.numpy()[0]) == pytest.approx(
            native.motion_counts["shield_correction"], abs=1e-10)
        return native_info
    finally:
        native.close()
        warp.close()


def test_warp_cbf_safe_parallel_motion_matches_native_without_intervention():
    info = _run_case(
        [[300, 500, 80], [320, 500, 80]],
        [[10, 0, 0], [10, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 0


def test_warp_cbf_static_cylinder_braking_matches_native():
    info = _run_case(
        [[350, 500, 25]],
        [[20, 0, 0]],
        [[0, 0, 0, 0]],
        [[500, 500, 10, 40]],
    )
    assert info["motion_counts"]["shield_interventions"] == 1


def test_warp_cbf_head_on_peer_projection_matches_native():
    info = _run_case(
        [[350, 500, 80], [650, 500, 80]],
        [[20, 0, 0], [-20, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 2


def test_warp_cbf_boundary_projection_matches_native():
    info = _run_case(
        [[120, 500, 80]],
        [[-20, 0, 0]],
        [[0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 1
