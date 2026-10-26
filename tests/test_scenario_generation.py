"""Synthetic missions must not isolate targets behind clearance barriers."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from uav_network.configuration import load_config, Config
from uav_network.environment.world import generate_scenario
from uav_network.algorithms.checkpoints import load_policy, make_learner
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.comparison_metrics import evaluation_protocol
from uav_network.experiments.evaluation_snapshots import load_evaluation_snapshot, save_evaluation_snapshot
from test_checkpoint_observation_version import settings, real_resume


def primary():
    return load_config("configs/comparison-20k-hybrid75-1kb.yaml")


def assert_free_routes(obstacles, config):
    w = config.world
    expanded = obstacles[:, 2] + w.obstacle_clearance_m
    assert np.all(obstacles[:, :2] > expanded[:, None] + 1e-9)
    assert np.all(obstacles[:, :2] < w.map_size_m - expanded[:, None] - 1e-9)
    for i, circle in enumerate(obstacles):
        distances = np.linalg.norm(obstacles[:i, :2] - circle[:2], axis=1)
        assert np.all(distances > expanded[:i] + expanded[i] + 2e-9)


@pytest.mark.parametrize("seed", [44, 302])
def test_native_obstacle_clearance_regions_preserve_free_routes(seed):
    c = primary()
    scene = generate_scenario(c, np.random.default_rng(seed))
    assert_free_routes(scene.obstacles, c)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("altitude", [1.0, 100.0])
def test_warp_obstacle_clearance_regions_preserve_free_routes(device, altitude):
    wp = pytest.importorskip("warp")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    from uav_network.environment.gpu.state import allocate_state
    from uav_network.environment.gpu.world import reset_geometry
    c = primary()
    c.sensor.altitude_m = altitude
    s, p = allocate_state(c, num_envs=4, device=device)
    mask = wp.array([1] * 4, dtype=wp.int32, device=device)
    wp.launch(reset_geometry, dim=4, inputs=[s, p, mask], device=device)
    assert not s.errors.numpy().any()
    for obstacles, targets in zip(s.obstacles.numpy(), s.targets.numpy()):
        assert_free_routes(obstacles, c)
        extra = max(0.0, c.world.obstacle_clearance_m - c.sensor.footprint_radius_m)
        distance = np.linalg.norm(targets[:, None] - obstacles[None, :, :2], axis=2)
        assert np.all(distance > obstacles[None, :, 2] + extra + 1e-9)


def test_small_camera_does_not_leave_generated_targets_in_forbidden_annulus():
    c = primary()
    c.sensor.altitude_m = 1.0
    scene = generate_scenario(c, np.random.default_rng(302))
    extra = max(0.0, c.world.obstacle_clearance_m - c.sensor.footprint_radius_m)
    distance = np.linalg.norm(scene.targets[:, None] - scene.obstacles[None, :, :2], axis=2)
    assert np.all(distance > scene.obstacles[None, :, 2] + extra + 1e-9)


def with_generation(saved, version):
    saved = deepcopy(saved)
    if version is None:
        saved["config"]["world"].pop("generation_version", None)
    else:
        saved["config"]["world"]["generation_version"] = version
    return saved


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("version", [None, 1, 2])
def test_policy_rejects_old_scene_generation(tmp_path, algorithm, version):
    c = settings(algorithm)
    learner = make_learner(c, observation_dim(c), state_dim(c))
    path = tmp_path / "policy.pt"
    learner.save(path)
    assert load_policy(path).algorithm == algorithm
    saved = torch.load(path, weights_only=True)
    with pytest.raises(ValueError, match="generation"):
        load_policy(path, payload=with_generation(saved, version))


@pytest.mark.parametrize("version", [None, 1, 2])
def test_evaluation_snapshot_rejects_old_scene_generation(tmp_path, version):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    path = save_evaluation_snapshot(c, tmp_path, learner.actors.state_dict(), 0,
                                    {"episodes": 0, "global_step": 4})
    saved = torch.load(path, weights_only=True)
    assert load_evaluation_snapshot(saved).algorithm == "masac"
    with pytest.raises(ValueError, match="generation"):
        load_evaluation_snapshot(with_generation(saved, version))


@pytest.mark.parametrize("version", [None, 1, 2])
def test_resume_rejects_old_scene_generation(real_resume, version):
    c, saved, validate = real_resume
    with pytest.raises(ValueError, match="generation"):
        validate(with_generation(saved, version), c)


def test_scenario_protocol_identifies_generator_semantics():
    protocol = evaluation_protocol(primary(), [2000000044], {})
    assert protocol["scenario"]["world"].get("generation_version") == 3


def test_unsupported_generation_version_is_rejected():
    c = Config()
    c.world.generation_version = 1
    with pytest.raises(ValueError, match="generation"):
        c.validate()
