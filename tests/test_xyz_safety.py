"""Regression tests for XYZ safety-filter intervention semantics."""
import numpy as np
import torch
import warp as wp

from test_xyz_sensing import xyz_config
from uav_network.environment.xyz_motion import move_xyz
from uav_network.environment.gpu.vector import WarpVectorEnv


def _config():
    c = xyz_config(n=2, targets=1, obstacles=0)
    c.world.motion_safety = "dynamic_window"
    c.energy.enabled = False
    return c


def test_native_safe_joint_nominal_motion_passes_through_unchanged():
    """Two UAVs with the same safe velocity must not be modified by the shield."""
    c = _config()
    positions = np.array([[300.0, 500.0, 60.0], [320.0, 500.0, 60.0]])
    velocities = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    acceleration = np.zeros((2, 3))
    diagnostics = {}

    next_positions, actual, blocked, clipped = move_xyz(
        positions,
        velocities,
        acceleration,
        np.empty((0, 3)),
        np.empty((0,)),
        c,
        diagnostics=diagnostics,
    )

    np.testing.assert_allclose(actual, velocities)
    np.testing.assert_allclose(next_positions[:, 0], [310.0, 330.0])
    assert blocked == 0
    assert clipped == 0
    assert diagnostics["shield_interventions"] == 0


def test_warp_safe_joint_nominal_motion_passes_through_unchanged():
    """Warp must use the same pass-through rule as native XYZ motion."""
    c = _config()
    c.runtime.num_envs = 1
    c.runtime.device = "cpu"
    c.runtime.environment_backend = "warp_packet"
    env = WarpVectorEnv(c, device="cpu")
    try:
        env.reset(seed=44)
        wp.to_torch(env.s.positions)[0] = torch.tensor([[300.0, 500.0], [320.0, 500.0]])
        wp.to_torch(env.s.altitudes)[0] = 60.0
        wp.to_torch(env.s.velocities)[0] = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        wp.to_torch(env.s.vertical_velocity)[0] = 0.0
        action = {
            "continuous": np.zeros((1, 2, 4), np.float32),
            "destination": np.zeros((1, 2), np.int64),
        }
        env.step(action)

        np.testing.assert_allclose(wp.to_torch(env.s.positions)[0].cpu().numpy()[:, 0], [310.0, 330.0])
        np.testing.assert_allclose(wp.to_torch(env.s.velocities)[0].cpu().numpy()[:, 0], [10.0, 10.0])
        assert int(wp.to_torch(env.s.motion_counts)[0, 3]) == 0
    finally:
        env.close()
