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


def _run_case(positions, velocities, actions, obstacles=(), *, unsafe_action=0.0,
              shield_correction=0.0, shield_intervention=0.0):
    c = _config(len(positions), len(obstacles))
    c.reward.unsafe_action = unsafe_action
    c.reward.shield_correction = shield_correction
    c.reward.shield_intervention = shield_intervention
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
        from uav_network.communication.gpu.metrics import world_info
        warp_info = world_info(warp.s, warp.p)
        warp_positions = np.column_stack((warp.s.positions.numpy()[0], warp.s.altitudes.numpy()[0]))
        warp_velocities = np.column_stack((warp.s.velocities.numpy()[0], warp.s.vertical_velocity.numpy()[0]))

        np.testing.assert_allclose(warp_positions, native.positions, rtol=0, atol=1e-8)
        np.testing.assert_allclose(warp_velocities, native.velocities, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(warp.s.motion_counts.numpy()[0, :13],
                                      [native.motion_counts[k] for k in (
                                          "obstacle_blocks", "peer_blocks", "commanded_stalls",
                                          "shield_interventions", "shield_obstacle_interventions",
                                          "shield_peer_interventions", "shield_boundary_interventions",
                                          "shield_infeasible", "raw_unsafe_attempts",
                                          "raw_boundary_attempts", "raw_obstacle_attempts",
                                          "raw_peer_attempts", "shield_rescues")])
        assert float(warp.s.shield_correction.numpy()[0]) == pytest.approx(
            native.motion_counts["shield_correction"], abs=1e-10)
        assert float(warp.s.reward.numpy()[0]) == pytest.approx(native_reward, abs=2e-5)
        for key in ("shield_correction", "unsafe_action", "shield_intervention"):
            assert warp_info["reward_components"][key] == pytest.approx(
                native_info["reward_components"][key], abs=1e-10)
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
        [[380, 500, 25]],
        [[20, 0, 0]],
        [[0, 0, 0, 0]],
        [[500, 500, 10, 40]],
    )
    assert info["motion_counts"]["shield_interventions"] == 1


def test_warp_cbf_head_on_peer_projection_matches_native():
    info = _run_case(
        [[390, 500, 80], [610, 500, 80]],
        [[20, 0, 0], [-20, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 2


def test_warp_cbf_viable_boundary_coast_matches_native_without_intervention():
    info = _run_case(
        [[110, 500, 80]],
        [[-20, 0, 0]],
        [[0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 0


def test_warp_cbf_boundary_projection_matches_native():
    info = _run_case(
        [[100, 500, 80]],
        [[-20, 0, 0]],
        [[0, 0, 0, 0]],
    )
    assert info["motion_counts"]["shield_interventions"] == 1


def test_warp_raw_boundary_rescue_penalizes_policy_before_cbf():
    info = _run_case(
        [[500.0, 500.0, 0.5]],
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0, 0.0]],
        unsafe_action=1.0,
        shield_correction=0.1,
        shield_intervention=0.5,
    )
    assert info["motion_counts"]["raw_boundary_attempts"] == 1
    assert info["motion_counts"]["shield_rescues"] == 1
    assert info["reward_components"]["unsafe_action"] == pytest.approx(-1.0)
    assert info["reward_components"]["shield_correction"] == pytest.approx(-0.075)
    assert info["reward_components"]["shield_intervention"] == pytest.approx(-0.5)
