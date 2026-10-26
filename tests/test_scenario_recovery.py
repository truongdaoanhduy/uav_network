"""Generation must recover layout exhaustion without weakening scenario constraints."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from uav_network.configuration import load_config
from uav_network.environment.world import generate_scenario, validate_scenario
from uav_network.environment.gpu.vector import WarpVectorEnv


def primary():
    return load_config("configs/comparison-20k-hybrid75-1kb.yaml")


def assert_scene(scene, config):
    validate_scenario(config, scene)
    w = config.world
    assert scene.obstacles.shape == (25, 3)
    assert np.all(scene.obstacles[:, 2] >= w.obstacle_radius_min_m)
    assert np.all(scene.obstacles[:, 2] <= w.obstacle_radius_max_m)
    expanded = scene.obstacles[:, 2] + w.obstacle_clearance_m
    assert np.all(scene.obstacles[:, :2] > expanded[:, None])
    assert np.all(scene.obstacles[:, :2] < w.map_size_m - expanded[:, None])
    for i, circle in enumerate(scene.obstacles):
        distance = np.linalg.norm(scene.obstacles[:i, :2] - circle[:2], axis=1)
        assert np.all(distance > expanded[:i] + expanded[i])


def test_recovery_has_its_own_generation_protocol():
    assert primary().world.generation_version == 3


@pytest.mark.parametrize("seed", [9642, 44, 302, 9641, 9643, 2000000044])
def test_complete_scene_survives_bad_and_independent_seeds(seed):
    c = primary()
    assert_scene(generate_scenario(c, np.random.default_rng(seed)), c)


def test_exhaustion_uses_repeatable_certified_scene_without_shared_mutation():
    c = primary()
    c.world.placement_attempts = 1
    a = generate_scenario(c, np.random.default_rng(9642))
    b = generate_scenario(c, np.random.default_rng(9642))
    assert a.generation_fallback and b.generation_fallback
    assert_scene(a, c)
    for field in ("positions", "targets", "obstacles", "gcs"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    a.positions[:] = -999
    fresh = generate_scenario(c, np.random.default_rng(9642))
    np.testing.assert_array_equal(fresh.positions, b.positions)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("attempts", [1, 5000])
def test_native_warp_recovery_and_masked_reset_match(device, attempts):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("Real CUDA device required")
    import warp as wp
    c = primary()
    c.runtime.num_envs = 4
    c.world.placement_attempts = attempts
    seeds = [9642, 44, 9643, 2000000044]
    env = WarpVectorEnv(c, device=device)
    try:
        env.reset(seed=seeds)
        fields = ("positions", "targets", "obstacles", "gcs")
        for i, seed in enumerate(seeds):
            generator = np.random.default_rng(seed)
            scene = generate_scenario(c, generator)
            assert_scene(scene, c)
            pcg = generator.bit_generator.state["state"]
            words = np.array([pcg["state"] & ((1 << 64) - 1), pcg["state"] >> 64,
                              pcg["inc"] & ((1 << 64) - 1), pcg["inc"] >> 64],
                             dtype=np.uint64)
            np.testing.assert_array_equal(wp.to_torch(env.s.pcg_state)[i].cpu().numpy(), words)
            assert int(wp.to_torch(env.s.generation_attempts)[i]) == scene.generation_attempts
            for field in fields:
                np.testing.assert_allclose(wp.to_torch(getattr(env.s, field))[i].cpu().numpy(),
                                           getattr(scene, field), rtol=0, atol=2e-10)
            assert int(wp.to_torch(env.s.generation_fallback)[i]) == int(scene.generation_fallback)
        before = {field: wp.to_torch(getattr(env.s, field)).clone()
                  for field in (*fields, "pcg_state", "generation_fallback")}
        env.reset(seed=[None, 9642, None, None],
                  options={"reset_mask": np.array([False, True, False, False])})
        for field, expected in before.items():
            np.testing.assert_array_equal(
                wp.to_torch(getattr(env.s, field)).cpu().numpy()[[0, 2, 3]],
                expected.cpu().numpy()[[0, 2, 3]])
        assert not bool(wp.to_torch(env.s.errors).any())
    finally:
        env.close()


def test_uncertifiable_configuration_is_rejected_at_device_startup():
    c = primary()
    c.runtime.num_envs = 1
    c.world.target_exclusion_radius_m = 2000
    with pytest.raises(ValueError, match="certif|fit|generation"):
        WarpVectorEnv(c, device="cpu")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forced_fallback_snapshot_restores_future_world_and_rng(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("Real CUDA device required")
    import warp as wp
    c = primary()
    c.runtime.num_envs = 2
    c.world.placement_attempts = 1
    reference = WarpVectorEnv(c, device=device)
    restored = WarpVectorEnv(c, device=device)
    try:
        observation, _ = reference.reset(seed=[9642, 44])
        assert bool(wp.to_torch(reference.s.generation_fallback).all())
        snapshot = reference.snapshot_state()
        action = reference.random_actions(observation["observation"])
        expected_action = {name: value.cpu().clone() for name, value in vars(action).items()}
        reference.step(vars(action))
        expected = reference.snapshot_state()
        restored.reset(seed=[9000, 9001])
        observation = restored.restore_state(snapshot)
        action = restored.random_actions(observation["observation"])
        for name, value in vars(action).items():
            assert torch.equal(value.cpu(), expected_action[name])
        restored.step(vars(action))
        actual = restored.snapshot_state()
        assert actual["arrays"].keys() == expected["arrays"].keys()
        for name, value in expected["arrays"].items():
            assert torch.equal(actual["arrays"][name], value), name
        assert torch.equal(actual["action_generator"], expected["action_generator"])
    finally:
        reference.close()
        restored.close()
