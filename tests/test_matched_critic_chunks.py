"""Chunk flags must bound real critic calls without changing a logical update."""
from copy import deepcopy

import pytest
import torch

from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3
from uav_network.configuration import Config, load_config


def make_learner(algorithm, chunk, shared=False, vectorized=False, width=16):
    c = Config()
    c.world.num_uavs = 2
    a = getattr(c, algorithm)
    a.hidden_sizes = [width, width]
    a.actor_layer_norm = True
    a.share_actor_parameters = a.share_critic_parameters = shared
    a.grad_clip_mode, a.grad_clip_norm = "global", .01
    c.runtime.critic_chunk_size = chunk
    c.runtime.vectorize_critics = vectorized
    return {"masac": MASAC, "matd3": MATD3}[algorithm](c, 12, 5)


def replay_batch(size=11):
    observation = torch.randn(size, 2, 12)
    observation[:, :, 9] = 1
    observation[:, :, -3:] = 1
    return {
        "observation": observation, "next_observation": observation.clone(),
        "state": torch.randn(size, 5), "next_state": torch.randn(size, 5),
        "continuous": torch.zeros(size, 2, 3),
        "destination": torch.zeros(size, 2, dtype=torch.long),
        "reward": torch.arange(size).float(),
        "terminated": (torch.arange(size) % 2).float(),
    }


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
def test_chunk_flag_bounds_target_regression_and_actor_critic_calls(algorithm, shared, vectorized):
    # Removing any chunk wrapper must expose an oversized REAL critic call.
    torch.set_num_threads(1)
    torch.manual_seed(44)
    model = make_learner(algorithm, 4, shared, vectorized)
    batch = replay_batch()
    seen = {"target": [], "online": []}
    handles = []
    for role, critics in (("target", model.target_critics), ("online", model.critics)):
        for critic in critics:
            handles.append(critic.register_forward_pre_hook(
                lambda module, inputs, role=role: seen[role].append(len(inputs[0]))))
    try:
        model.update_critic(batch)
        # Chunk is original replay rows; enumeration happens inside each chunk.
        expansion = (model.destinations * (model.num_uavs if shared else 1)
                     if algorithm == "masac" else 1)
        assert seen["target"] and max(seen["target"]) <= 4 * expansion
        assert seen["online"] and max(seen["online"]) <= 4
        # Uneven final partition is actually evaluated.
        assert 3 * expansion in seen["target"]
        assert 3 in seen["online"]
        seen["online"].clear()
        model.update_actor(batch)
        expansion = ((model.destinations * (model.num_uavs if shared else 1))
                     if algorithm == "masac" else (model.num_uavs if shared else 1))
        assert seen["online"] and max(seen["online"]) <= 4 * expansion
        assert 3 * expansion in seen["online"]
    finally:
        for handle in handles:
            handle.remove()


def optimizer_snapshots(model):
    optimizers = [model.critic_optimizer, *model.actor_optimizers]
    if model.algorithm == "masac":
        optimizers.append(model.alpha_optimizer)
    return [deepcopy(optimizer.state_dict()) for optimizer in optimizers]


def assert_tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_close(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("vectorized", [False, True])
def test_chunked_updates_keep_rng_gradients_clipping_and_optimizer_state(algorithm, shared, vectorized):
    # Wrong chunk weighting, resampling or per-chunk optimizer/clipping fails here.
    torch.set_num_threads(1)
    torch.manual_seed(72)
    reference = make_learner(algorithm, 0, shared, vectorized)
    actual = make_learner(algorithm, 4, shared, vectorized)
    actual.load_state_dict(reference.state_dict())
    batch = replay_batch()
    for step in range(3):
        torch.manual_seed(440 + step)
        expected = reference.update(batch)
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(440 + step)
        observed = actual.update(batch)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert_tree_close(actual.state_dict(), reference.state_dict())
        assert_tree_close(optimizer_snapshots(actual), optimizer_snapshots(reference))
        for name, value in expected.items():
            assert observed[name] == pytest.approx(value, rel=3e-5, abs=3e-5)
        assert observed["critic_gradient_clipped"] == 1.0
        assert reference.updates == actual.updates == step + 1
        actor_steps = step + 1 if algorithm == "masac" else (step + 1) // 2
        assert reference.actor_updates == actual.actor_updates == actor_steps


def test_matd3_checkpoint_reduces_saved_actor_critic_storage_and_keeps_input_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(99)
    reference = make_learner("matd3", 0, True, width=64)
    actual = make_learner("matd3", 4, True, width=64)
    actual.load_state_dict(reference.state_dict())
    state = torch.randn(73, 5)
    joints = torch.randn(2, 73, 2, 6)
    storage_totals, gradients, values = [], [], []
    for model in (reference, actual):
        model.critics.requires_grad_(False)
        actions = joints.clone().requires_grad_()
        storage = {}
        def pack(tensor):
            backing = tensor.untyped_storage()
            storage[backing.data_ptr()] = backing.nbytes()
            return tensor
        before_rng = torch.random.get_rng_state().clone()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            q1, q2 = model._shared_critic_actor_values(state, actions)
        storage_totals.append(sum(storage.values()))
        values.append((q1.detach(), q2.detach()))
        (q1.square().mean() + q2.square().mean()).backward()
        gradients.append(actions.grad)
        assert torch.equal(torch.random.get_rng_state(), before_rng)
        assert all(p.grad is None for p in model.critics.parameters())
    assert_tree_close(values[1], values[0])
    assert_tree_close(gradients[1], gradients[0])
    assert storage_totals[1] < storage_totals[0] * .8


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_current_profile_constructs_shared_models_with_bounded_execution(algorithm):
    c = load_config("configs/comparison-reference-masac-matd3-xyz-sensing.yaml")
    c.runtime.algorithm = algorithm
    model = {"masac": MASAC, "matd3": MATD3}[algorithm](c, 198, 565)
    assert len(model.actors) == len(model.critics) == 1
    assert model.share_actor_parameters and model.share_critic_parameters
    assert c.runtime.critic_chunk_size == 2048


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("pql_enabled", [False, True])
def test_current_shared_profile_trains_through_real_cpu_collector(tmp_path, algorithm, pql_enabled):
    # Validate config propagation through actual BASE/PQL learners and checkpoints.
    pytest.importorskip("warp")
    from uav_network.experiments.training import train
    c = load_config("configs/comparison-reference-masac-matd3-xyz-sensing.yaml")
    c.runtime.algorithm = algorithm
    c.pql.enabled = pql_enabled
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.runtime.output_dir = str(tmp_path / "run")
    c.runtime.num_envs = c.runtime.episodes = 4
    c.world.max_steps = 4
    c.runtime.critic_chunk_size = 2  # Small but effective chunks in this CPU smoke.
    c.tracking.enabled = False
    c.tracking.evaluation_interval = c.tracking.evaluation_interval_steps = 0
    c.runtime.checkpoint_interval = 4
    c.pql.critic_sample_ratio = 2
    c.pql.warmup_collection_steps = 1
    for a in (c.masac, c.matd3):
        a.hidden_sizes = [16, 16]
        a.batch_size = a.learning_starts = 4
        a.replay_capacity = 32
        a.updates_per_step = 2
    c.validate()
    result = train(c)
    assert result["episodes"] == 4
    assert result["updates"] >= 2
    model = {"masac": MASAC, "matd3": MATD3}[algorithm].load(result["checkpoint"])
    assert model.share_actor_parameters and model.share_critic_parameters
    assert len(model.actors) == len(model.critics) == 1
    assert model.config.runtime.critic_chunk_size == 2
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
