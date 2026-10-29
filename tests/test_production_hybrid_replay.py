"""Production replay placement and exact sampling survive cloud launch and resume."""
from copy import deepcopy
import json

import pytest
import torch

from test_pql_system import pql_config, transition
from test_split_updates import replay_batch, small_learner
from uav_network.cloud.remote import learner_command
from uav_network.configuration import Config
from uav_network.experiments.pql_system import PQLSystem


def test_cloud_learner_preserves_explicit_cpu_replay_placement():
    spec = {"stage": "train", "config": "configs/kaggle-smoke.yaml",
            "overrides": ["runtime.replay_device=cpu", "runtime.replay_cpu_fraction=0.75"]}
    args, _ = learner_command(spec, "masac", {"id": "run", "name": "test"}, "0,1", "/tmp/train/masac")
    placement = [arg for arg in args if arg.startswith("runtime.replay_device=")]
    assert placement[-1] == "runtime.replay_device=cpu"


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_pql_constructs_hybrid_replay_and_restores_partial_rng(tmp_path, algorithm):
    torch.set_num_threads(1)
    c = pql_config(algorithm)
    c.runtime.replay_cpu_fraction = .75
    values = transition(replay_batch(small_learner(algorithm)))
    with_system = PQLSystem(c)
    try:
        with_system.submit(values)
        with_system.flush()
        assert getattr(with_system.replay, "cpu_capacity", None) == 12
        assert with_system.replay.hot_capacity == 4
        path = tmp_path / "resume.pt"
        with_system.save(path, {"episodes": 0, "global_step": 4, "collection_steps": 1,
                                "next_seed": 44, "records": []})
        expected = with_system.replay.sample(4)
    finally:
        with_system.close()
    restored = PQLSystem(c, checkpoint=path)
    try:
        actual = restored.replay.sample(4)
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    finally:
        restored.close()


def test_base_uses_hybrid_replay_in_real_training_checkpoint(tmp_path):
    from test_tracking_cleanup_order import base_config
    from uav_network.experiments.training import train
    c = base_config(tmp_path / "train")
    c.runtime.replay_cpu_fraction = .75
    train(c)
    saved = torch.load(tmp_path / "train" / "resume_latest.pt", weights_only=True)
    replay = saved["training"]["replay"]
    assert replay.get("layout") == "hybrid"
    assert replay["cpu_capacity"] == 48
    c.runtime.episodes = 4
    c.runtime.output_dir = str(tmp_path / "resumed")
    c.runtime.warm_start_checkpoint = str(tmp_path / "train" / "resume_latest.pt")
    result = train(c)
    assert result["episodes"] == 4


@pytest.mark.parametrize("fraction", [-.1, 1., float("nan"), float("inf")])
def test_rejects_invalid_hybrid_fraction(fraction):
    c = Config()
    c.runtime.replay_device = "cpu"
    c.runtime.replay_cpu_fraction = fraction
    with pytest.raises(ValueError, match="replay_cpu_fraction"):
        c.validate()


def test_frozen_hybrid_checkpoint_does_not_alias_measurement_rows():
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    from uav_network.experiments.fixed_checkpoints import replay_state
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = HybridMeasuredReplay(8, values[0].shape[1:], learner.state_dim,
                                 hot_device="cpu", cpu_fraction=.75)
    replay.add_batch(*values)
    saved = replay_state(replay)
    replay.sample(4)
    assert not bool(saved["seen"].any())


def test_hybrid_restore_rejects_nonfinite_rows_before_mutation():
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = HybridMeasuredReplay(8, values[0].shape[1:], learner.state_dim,
                                 hot_device="cpu", cpu_fraction=.75)
    replay.add_batch(*values)
    saved = replay.state_dict(copy=True)
    saved["cpu_data"]["reward"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replay.load_state_dict(saved)


def test_production_shape_probe_uses_requested_layout_and_real_training_factory():
    from uav_network.experiments.cpu_replay_probe import probe_config
    c = probe_config("hybrid75", production_shape=True)
    assert c.runtime.num_envs == 4096
    assert c.masac.batch_size == 8192
    assert c.masac.replay_capacity == 1_000_000
    assert c.runtime.replay_cpu_fraction == .75
    assert c.runtime.replay_device == "cpu"
    assert not c.runtime.compile_models


@pytest.mark.parametrize("collections", [1, 2, 3])
def test_hybrid_ring_matches_uniform_cpu_reference_after_wrap(collections):
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    from uav_network.experiments.pql_replay import MeasuredReplay
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = HybridMeasuredReplay(8, values[0].shape[1:], learner.state_dim, seed=7,
                                 hot_device="cpu", cpu_fraction=.75)
    reference = MeasuredReplay(8, values[0].shape[1:], learner.state_dim, seed=7)
    for offset in range(collections):
        batch = list(values)
        batch[3] = torch.arange(4, dtype=torch.float32) + 4 * offset
        replay.add_batch(*batch)
        reference.add_batch(*batch)
    expected, actual = reference.sample(4), replay.sample(4)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    assert replay.index == reference.index


def test_partial_ring_borrows_a_full_cpu_tier_during_synchronous_save():
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    learner = small_learner()
    batch = replay_batch(learner, count=6)
    batch["terminated"] = torch.zeros(6)
    replay = HybridMeasuredReplay(8, batch["observation"].shape[1:], learner.state_dim,
                                 hot_device="cpu", cpu_fraction=.75)
    replay.add_batch(*transition(batch))
    saved = replay.state_dict()
    assert saved["cpu_data"]["observation"].data_ptr() == replay.cpu_data["observation"].data_ptr()


@pytest.mark.parametrize("corruption", ["missing_counter", "cursor"])
def test_bad_hybrid_checkpoint_is_rejected_before_destination_mutation(corruption):
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    learner = small_learner()
    values = transition(replay_batch(learner))
    def new():
        return HybridMeasuredReplay(8, values[0].shape[1:], learner.state_dim,
                                    hot_device="cpu", cpu_fraction=.75)
    source, destination = new(), new()
    source.add_batch(*values)
    saved = source.state_dict(copy=True)
    if corruption == "missing_counter":
        saved.pop("collections")
    else:
        saved["index"] = 0
    before_rng = destination.rng.get_state().clone()
    with pytest.raises(ValueError):
        destination.load_state_dict(saved)
    assert destination.size == destination.index == 0
    torch.testing.assert_close(destination.rng.get_state(), before_rng, rtol=0, atol=0)


def test_comparison_pql_preserves_cpu_replay_and_legacy_rejects_it(tmp_path):
    import runpy
    from unittest.mock import patch
    from uav_network.configuration import load_config
    module = runpy.run_path("scripts/compare_training_gpu.py")
    c = load_config("configs/comparison-20k-hybrid75-1m.yaml")
    with patch.object(torch.cuda, "is_available", return_value=True), \
         patch.object(torch.cuda, "device_count", return_value=2):
        selected = module["case_config"](c, "pql-dual", "masac", tmp_path)
        assert selected.runtime.replay_device == "cpu"
        assert selected.runtime.replay_cpu_fraction == .75
        with pytest.raises(ValueError, match="[Ll]egacy"):
            module["case_config"](c, "legacy", "masac", tmp_path)


def test_harp_lawn_matched_profile_uses_one_kib_report():
    from uav_network.configuration import load_config
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    assert c.network.report_bytes == 1024
    assert c.network.payload_bytes == 1024
    assert c.network.buffer_bytes == 3072
    assert c.runtime.num_envs == 4096
    assert c.runtime.episodes == 20000
    assert c.world.max_steps == 1200
    assert c.runtime.amp_mode == "off"
    assert c.runtime.compile_models is True
    assert c.runtime.fused_optimizers is False
    assert "accel-policy" in c.tracking.group
    assert "compile" not in c.tracking.group.lower()
    assert "amp-off" in c.tracking.tags
    assert "torch-compile" in c.tracking.tags
