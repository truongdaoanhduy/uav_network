"""2.5D is opt-in, non-oracle, and keeps physics/PHY-MAC unchanged."""
import numpy as np
import pytest
import torch
from uav_network.configuration import load_config
from uav_network.environment.xyz_motion import move_xyz

BASE = "configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml"
AB = "configs/matd3-nooracle-map1000-25d-h35-130-vz4-20261123.yaml"
CBF = "configs/matd3-nooracle-map1000-25d-cbf-h35-130-vz4-20261123.yaml"


def test_baseline_is_unchanged_and_2_5d_has_required_constraints():
    original, variant, cbf = map(load_config, (BASE, AB, CBF))
    assert original.world.max_vertical_speed_mps == 0
    assert (original.world.altitude_min_m, original.world.altitude_max_m) == (0, 150)
    assert (variant.world.altitude_min_m, variant.world.altitude_max_m) == (35, 130)
    assert variant.world.max_vertical_speed_mps == 4
    assert variant.world.motion_safety == "reject"
    assert cbf.world.motion_safety == "cbf"
    assert variant.sensor.belief_prior == pytest.approx(35 / 1600)
    assert variant.sensor.confirmation_mode == "posterior_only"
    assert variant.network == original.network
    assert variant.runtime.episodes == original.runtime.episodes == 20000
    assert variant.runtime.num_envs == 4096
    assert variant.matd3.batch_size == 8192
    assert variant.matd3.updates_per_step == 4
    assert variant.runtime.warm_start_checkpoint == "none"


def test_invalid_vertical_caps_rejected():
    config = load_config(AB)
    for value in (-1, 21):
        config.world.max_vertical_speed_mps = value
        with pytest.raises(ValueError):
            config.validate()


def test_native_2_5d_vertical_cap_respects_acceleration_and_boundary():
    config = load_config(AB)
    config.world.num_obstacles = 0
    positions = np.array([[500., 500., 100.]])
    velocities = np.array([[0., 0., 3.5]])
    acceleration = np.array([[0., 0., 1.]])
    out, real_v, blocked, clipped = move_xyz(
        positions, velocities, acceleration, np.empty((0, 3)),
        np.empty((0,)), config)
    assert blocked == clipped == 0
    assert real_v[0, 2] == pytest.approx(4.0)
    assert out[0, 2] == pytest.approx(104.0)


def test_warp_cpu_2_5d_applies_vertical_cap():
    import warp as wp
    from uav_network.environment.gpu.vector import WarpVectorEnv
    config = load_config(AB)
    config.runtime.num_envs = 1
    config.runtime.device = "cpu"
    config.world.num_uavs = 2
    config.world.num_targets = 1
    config.world.num_obstacles = 0
    config.network.enabled = False
    env = WarpVectorEnv(config, device="cpu")
    try:
        env.reset(seed=44)
        wp.to_torch(env.s.positions)[0] = torch.tensor([[500.,500.],[700.,700.]], dtype=torch.float64)
        wp.to_torch(env.s.altitudes)[0] = 100.
        wp.to_torch(env.s.velocities)[0] = 0.
        wp.to_torch(env.s.vertical_velocity)[0] = torch.tensor([3.5,0.0], dtype=torch.float64)
        action = dict(continuous=np.zeros((1,2,4),dtype=np.float32),
                      destination=np.zeros((1,2),dtype=np.int64))
        action["continuous"][0,0,2] = 1.
        env.step(action)
        result = wp.to_torch(env.s.vertical_velocity).cpu().numpy()
        np.testing.assert_allclose(result[0], [4.,0.],atol=1e-8)
        assert np.all(wp.to_torch(env.s.altitudes).cpu().numpy()>=35.)
    finally:
        env.close()


def test_aoi_science_knob_only_affects_new_ablation():
    base, variant = load_config(BASE), load_config(AB)
    assert base.reward.delivery_aoi_weight == 0
    assert variant.reward.delivery_aoi_weight == .5
    assert variant.reward.communication_efficiency == .05
    assert base.reward.communication_efficiency == .2
    from uav_network.environment.gpu.state import params_from_config
    assert params_from_config(base).reward_delivery_aoi_weight == 0
    assert params_from_config(variant).reward_delivery_aoi_weight == .5
    for weight in (-.01, 1.01):
        variant.reward.delivery_aoi_weight = weight
        with pytest.raises(ValueError, match="delivery_aoi_weight"):
            variant.validate()
