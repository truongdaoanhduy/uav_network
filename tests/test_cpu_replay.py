"""CPU replay placement, bounded pinning and safe replay RNG restoration."""
from copy import deepcopy

import pytest
import torch

from uav_network.algorithms.optimization import tensor_batch
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.experiments.pql_replay import MeasuredReplay
from uav_network.experiments.pql_system import PQLSystem
from test_pql_system import pql_config, transition
from test_split_updates import replay_batch, small_learner


@pytest.fixture(autouse=True)
def one_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("replay_cls", [TensorReplayBuffer, MeasuredReplay])
def test_cpu_storage_stays_pageable_when_sample_pinning_is_requested(replay_cls):
    # A 1M replay must not lock ~12 GiB; pin only the returned minibatch.
    replay = replay_cls(16, (2, 3), 4, device="cpu", pin_memory=True)
    assert replay.pin_memory
    assert all(value.device.type == "cpu" and not value.is_pinned()
               for value in replay.data.values())


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only host")
def test_pql_rejects_an_explicit_unavailable_cuda_replay():
    config = pql_config()
    config.runtime.replay_device = "cuda"
    engine = None
    try:
        with pytest.raises(RuntimeError, match="CUDA replay"):
            engine = PQLSystem(config)
    finally:
        if engine is not None:
            engine.close()


def test_pql_replay_checkpoint_records_the_rng_device():
    engine = PQLSystem(pql_config())
    try:
        assert engine.replay.state_dict().get("device_type") == "cpu"
    finally:
        engine.close()


def test_foreign_replay_rng_device_is_rejected_before_mutation():
    learner = small_learner()
    replay = MeasuredReplay(16, (learner.num_uavs, learner.observation_dim),
                            learner.state_dim)
    replay.add_batch(*transition(replay_batch(learner)))
    replay.sample(4)
    before = deepcopy(replay.state_dict())
    bad = deepcopy(before)
    bad["device_type"] = "cuda"
    bad["index"] = (before["index"] + 1) % replay.capacity
    bad["collections"] += 10
    bad["data"]["observation"].fill_(99)
    with pytest.raises(ValueError, match="replay.*device|replay.*RNG"):
        replay.load_state_dict(bad)
    after = replay.state_dict()
    assert (after["index"], after["size"], after["collections"]) == (
        before["index"], before["size"], before["collections"])
    torch.testing.assert_close(after["rng"], before["rng"], rtol=0, atol=0)
    for name in before["data"]:
        torch.testing.assert_close(after["data"][name], before["data"][name],
                                   rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_two_gpu_pql_learns_from_pageable_cpu_replay(algorithm):
    config = pql_config(algorithm, "async")
    config.runtime.device = "cuda"
    config.runtime.replay_device = "cpu"
    config.pql.collector_device = config.pql.learner_device = config.pql.policy_device = "auto"
    engine = PQLSystem(config)
    try:
        assert str(engine.collector_device) == "cuda:0"
        assert str(engine.critic_device) == str(engine.policy_device) == "cuda:1"
        assert engine.replay.device.type == "cpu"
        batch = {name: value.to("cuda:0")
                 for name, value in replay_batch(engine.critic).items()}
        for _ in range(3):
            engine.submit(transition(batch))
        engine.flush()
        assert engine.critic.updates > 0
        assert engine.policy.actor_updates > 0
        assert all(not value.is_pinned() for value in engine.replay.data.values())
        sampled = engine.replay.sample(4)
        assert all(value.is_pinned() for value in sampled.values())
        moved = tensor_batch(sampled, engine.critic_device)
        torch.cuda.synchronize(engine.critic_device)
        for name in sampled:
            assert moved[name].device == engine.critic_device
            torch.testing.assert_close(moved[name].cpu(), sampled[name], rtol=0, atol=0)
    finally:
        engine.close()

@pytest.mark.parametrize("requested,learner,expected", [
    ("auto", "cpu", "cpu"), ("auto", "cuda:1", "cuda:1"),
    ("cpu", "cuda:1", "cpu"), ("cuda", "cuda:1", "cuda:1"),
])
def test_replay_role_selection_preserves_the_learner_default(requested, learner, expected):
    from uav_network.algorithms import replay
    assert str(replay.resolve_replay_device(requested, torch.device(learner))) == expected


def test_invalid_legacy_rng_is_rejected_before_replay_mutation():
    learner = small_learner()
    replay = MeasuredReplay(16, (learner.num_uavs, learner.observation_dim),
                            learner.state_dim)
    replay.add_batch(*transition(replay_batch(learner)))
    before = deepcopy(replay.state_dict())
    bad = deepcopy(before)
    bad.pop("device_type", None)
    bad["rng"] = torch.zeros(3, dtype=torch.uint8)
    bad["index"] = (bad["index"] + 1) % replay.capacity
    bad["data"]["observation"].fill_(99)
    with pytest.raises(ValueError, match="replay.*RNG"):
        replay.load_state_dict(bad)
    assert replay.index == before["index"]
    torch.testing.assert_close(replay.data["observation"][:replay.size],
                               before["data"]["observation"], rtol=0, atol=0)

@pytest.mark.parametrize("legacy", [False, True])
def test_cpu_replay_resume_keeps_next_samples_and_measurements(legacy):
    learner = small_learner()
    values = transition(replay_batch(learner))
    original = MeasuredReplay(16, values[0].shape[1:], learner.state_dim)
    for _ in range(5):
        original.add_batch(*values)
    original.sample(4)
    saved = deepcopy(original.state_dict())
    if legacy:
        saved.pop("device_type", None)
    restored = MeasuredReplay(16, values[0].shape[1:], learner.state_dim, seed=99)
    restored.load_state_dict(saved)
    expected, actual = original.sample(4), restored.sample(4)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert restored.stats() == original.stats()
    assert restored.index == original.index


def test_full_cpu_replay_save_can_borrow_storage_without_changing_frozen_snapshots():
    from uav_network.experiments.fixed_checkpoints import replay_state
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = MeasuredReplay(4, values[0].shape[1:], learner.state_dim)
    replay.add_batch(*values)
    frozen = replay_state(replay)
    borrowed = replay_state(replay, copy=False)
    for name in replay.data:
        assert borrowed["data"][name].data_ptr() == replay.data[name].data_ptr()
        assert frozen["data"][name].data_ptr() != replay.data[name].data_ptr()
    replay.data["observation"].add_(100)
    torch.testing.assert_close(frozen["data"]["observation"], values[0],
                               rtol=0, atol=0)


def test_synchronous_fixed_save_borrows_full_cpu_storage_and_writes_immutable_file(tmp_path):
    from unittest.mock import patch
    import numpy as np
    from uav_network.algorithms.checkpoints import save_policy
    from uav_network.experiments.fixed_checkpoints import save_fixed_checkpoint

    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = MeasuredReplay(4, values[0].shape[1:], learner.state_dim)
    replay.add_batch(*values)
    borrowed = []

    def observe_real_save(*args, **kwargs):
        data = kwargs["training"]["replay"]["data"]
        borrowed.append(all(data[name].data_ptr() == replay.data[name].data_ptr()
                            for name in data))
        return save_policy(*args, **kwargs)

    path = tmp_path / "resume.pt"
    with patch("uav_network.experiments.fixed_checkpoints.save_policy",
               side_effect=observe_real_save):
        save_fixed_checkpoint(
            learner, path,
            {"episodes": 0, "global_step": 4, "collection_steps": 1,
             "next_seed": 48, "records": []},
            replay, np.random.default_rng(45), warmup_origin=0,
            random_warmup=True, next_metrics_at=0, latest={},
            pending_learning=False)
    assert borrowed == [True]
    replay.data["observation"].add_(100)
    written = torch.load(path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(written["training"]["replay"]["data"]["observation"],
                               values[0], rtol=0, atol=0)


def test_partial_cpu_pql_checkpoint_contains_only_occupied_storage(tmp_path):
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = MeasuredReplay(64, values[0].shape[1:], learner.state_dim)
    replay.add_batch(*values)
    path = tmp_path / "partial.pt"
    torch.save(replay.state_dict(), path)
    written = torch.load(path, map_location="cpu", weights_only=True)
    for name, value in written["data"].items():
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size(), name


@pytest.mark.skipif(not __import__("pathlib").Path("/proc/self/maps").is_file(),
                    reason="Linux mmap introspection required")
@pytest.mark.parametrize("trainer", ["base", "pql"])
def test_full_training_restore_maps_checkpoint_and_keeps_source_file_immutable(tmp_path, trainer):
    from hashlib import sha256
    from pathlib import Path
    if trainer == "base":
        from test_baseline_resume import small_config
        from uav_network.experiments.training import train
        from uav_network.experiments.continuation import restore_learner
        config = small_config(tmp_path / "first", "matd3")
        result = train(config)
        path = Path(result["resume_checkpoint"]).resolve()
        config.runtime.warm_start_checkpoint = str(path)
        config.runtime.episodes = 3
        before = sha256(path.read_bytes()).hexdigest()
        learner, progress = restore_learner(config, "cpu")
        assert str(path) in Path("/proc/self/maps").read_text()
        # Optimizer state is mapped privately: an actual resumed update must
        # never write through into the checkpoint file.
        learner.update(replay_batch(learner))
        assert progress["_training"]["replay"]["size"] > 0
    else:
        config = pql_config("masac")
        engine = PQLSystem(config)
        path = (tmp_path / "pql-resume.pt").resolve()
        try:
            values = transition(replay_batch(engine.critic))
            for _ in range(3):
                engine.submit(values)
            engine.save(path, {"episodes": 0, "global_step": 12,
                               "collection_steps": 3, "next_seed": 48, "records": []})
        finally:
            engine.close()
        before = sha256(path.read_bytes()).hexdigest()
        restored = PQLSystem(config, checkpoint=path)
        try:
            # PQL copies every live tensor it needs during construction, so the
            # lazy checkpoint mapping can be released before the first update.
            assert str(path) not in Path("/proc/self/maps").read_text()
            restored.submit(values)
            restored.flush()
        finally:
            restored.close()
    assert sha256(path.read_bytes()).hexdigest() == before
