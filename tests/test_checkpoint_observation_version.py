"""Equal tensor dimensions must not mix policies with different observations."""
from copy import deepcopy

import pytest
import torch

from uav_network.algorithms.checkpoints import load_policy, make_learner
from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.evaluation_snapshots import load_evaluation_snapshot, save_evaluation_snapshot


def settings(algorithm="masac"):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
    c.world.max_steps = 1
    c.network.enabled = False
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.algorithm = algorithm
    c.runtime.num_envs = c.runtime.episodes = 2
    c.runtime.torch_threads = 1
    c.pql.mode = "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps = 1
    for a in (c.masac,c.matd3):
        a.hidden_sizes, a.replay_capacity = [8,8], 8
        a.batch_size = a.learning_starts = 2
    return c


def old_observation_payload(saved, version):
    saved = deepcopy(saved)
    if version is None:
        saved.pop("observation_version", None)
    else:
        saved["observation_version"] = version
    return saved


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("version", [None,2])
def test_policy_rejects_same_shape_with_old_observation_semantics(tmp_path, algorithm, version):
    c = settings(algorithm)
    learner = make_learner(c, observation_dim(c), state_dim(c))
    path = tmp_path / "policy.pt"
    learner.save(path)
    assert load_policy(path).algorithm == algorithm
    saved = old_observation_payload(torch.load(path, weights_only=True), version)
    with pytest.raises(ValueError, match="observation"):
        load_policy(path, payload=saved)


@pytest.mark.parametrize("version", [None,2])
def test_evaluation_snapshot_rejects_old_observation_semantics(tmp_path, version):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    path = save_evaluation_snapshot(c, tmp_path, learner.actors.state_dict(), 0,
                                    {"episodes":0,"global_step":4})
    saved = torch.load(path, weights_only=True)
    assert load_evaluation_snapshot(saved).algorithm == "masac"
    with pytest.raises(ValueError, match="observation"):
        load_evaluation_snapshot(old_observation_payload(saved, version))


@pytest.fixture(params=[False,True], ids=["base","pql"])
def real_resume(tmp_path, request):
    from uav_network.experiments.training import train
    from uav_network.experiments.fixed_checkpoints import validate_fixed_checkpoint
    from uav_network.experiments.pql_checkpoints import validate_pql_checkpoint
    c = settings()
    c.pql.enabled = request.param
    c.runtime.output_dir = str(tmp_path / "train")
    result = train(c)
    saved = torch.load(result["resume_checkpoint"], weights_only=True)
    c.runtime.episodes = 4
    validate = validate_pql_checkpoint if c.pql.enabled else validate_fixed_checkpoint
    assert validate(saved,c)["episodes"] == 2
    return c, saved, validate


@pytest.mark.parametrize("version", [None,2])
def test_resume_preflight_rejects_old_observation_semantics(real_resume, version):
    c, saved, validate = real_resume
    with pytest.raises(ValueError, match="observation"):
        validate(old_observation_payload(saved,version),c)
