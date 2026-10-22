"""A reproducible diagnostic probe must preserve science and fill exact capacity."""
import torch

from uav_network.configuration import load_config
from uav_network.experiments.pql_replay import MeasuredReplay
from test_pql_system import transition
from test_split_updates import replay_batch, small_learner


def test_probe_profile_preserves_science_and_selects_cpu_replay():
    from uav_network.experiments.cpu_replay_probe import probe_config
    original = load_config("configs/comparison-20k-steps1800-scenario-v3.yaml")
    config = probe_config()
    for group in ("world", "sensor", "network", "energy", "reward"):
        assert config.to_dict()[group] == original.to_dict()[group]
    assert config.runtime.num_envs == 2048
    assert config.masac.batch_size == 4096
    assert config.masac.replay_capacity == 1_000_000
    assert config.masac.hidden_sizes == [256, 256]
    assert config.runtime.replay_device == "cpu"
    assert config.pql.collector_device == "cuda:0"
    assert config.pql.learner_device == "cuda:1"
    assert config.pql.policy_device == "cuda:1"
    assert config.runtime.amp_mode == "fp16"
    assert config.runtime.compile_models and config.runtime.fused_optimizers
    assert config.pql.enabled and config.pql.checkpoint_replay
    assert config.pql.warmup_collection_steps == 2
    assert config.pql.critic_sample_ratio == 16


def test_probe_fill_touches_every_row_without_overfilling_non_multiple_capacity():
    from uav_network.experiments.cpu_replay_probe import fill_replay
    torch.set_num_threads(1)
    learner = small_learner()
    values = transition(replay_batch(learner))
    replay = MeasuredReplay(7, values[0].shape[1:], learner.state_dim)
    inserted = fill_replay(replay, values)
    assert inserted == 7
    assert len(replay) == replay.capacity
    assert replay.index == 0
    assert replay.stats()["replay/inserted_rows"] == 7
    torch.testing.assert_close(replay.data["observation"][:4], values[0], rtol=0, atol=0)
    torch.testing.assert_close(replay.data["observation"][4:], values[0][:3], rtol=0, atol=0)


def test_probe_rejects_cpu_host_before_allocating_full_replay(tmp_path):
    import pytest
    from uav_network.experiments.cpu_replay_probe import run_probe
    if torch.cuda.device_count() >= 2:
        pytest.skip("two GPUs available")
    with pytest.raises(RuntimeError, match="exactly two T4 CUDA"):
        run_probe(tmp_path)


def test_probe_rejects_non_t4_two_gpu_host_before_large_allocation(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from uav_network.experiments.cpu_replay_probe import _require_two_t4
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(name="NVIDIA A100", total_memory=1,
                                                      major=8, minor=0, uuid=None))
    with pytest.raises(RuntimeError, match="exactly two T4 CUDA"):
        _require_two_t4()


def test_memory_probe_distinguishes_process_ram_from_effective_available_ram():
    from uav_network.experiments.cpu_replay_probe import _host_memory
    memory = _host_memory()
    assert memory["host_rss_bytes"] > 0
    assert 0 <= memory["effective_available_bytes"] <= memory["system_available_bytes"]
    if memory["cgroup_limit_bytes"] is not None:
        assert memory["effective_available_bytes"] <= max(
            0, memory["cgroup_limit_bytes"] - memory["cgroup_current_bytes"])


def test_probe_uses_current_warp_vector_constructor_without_removed_scheduler_keyword():
    import inspect
    from uav_network.environment.gpu.vector import WarpVectorEnv
    from uav_network.experiments import cpu_replay_probe
    assert "packet_scheduler" not in inspect.signature(WarpVectorEnv).parameters
    assert "packet_scheduler=" not in inspect.getsource(cpu_replay_probe.run_probe)


def test_hybrid_probe_replay_splits_storage_and_restores_rng_exactly():
    from uav_network.environment.env import Action
    from uav_network.experiments.cpu_replay_probe import HybridMeasuredReplay
    replay = HybridMeasuredReplay(8, (1, 2), 3, seed=7, hot_device="cpu")
    assert replay.cpu_capacity == 4
    assert replay.hot_capacity == 4
    values = {
        "observation": torch.arange(16, dtype=torch.float32).reshape(8, 1, 2),
        "state": torch.arange(24, dtype=torch.float32).reshape(8, 3),
        "continuous": torch.arange(24, dtype=torch.float32).reshape(8, 1, 3),
        "destination": torch.arange(8, dtype=torch.long).reshape(8, 1),
        "reward": torch.arange(8, dtype=torch.float32),
        "next_observation": torch.arange(16, 32, dtype=torch.float32).reshape(8, 1, 2),
        "next_state": torch.arange(24, 48, dtype=torch.float32).reshape(8, 3),
        "terminated": torch.zeros(8),
    }
    replay.add_batch(values["observation"], values["state"],
                     Action(values["continuous"], values["destination"]), values["reward"],
                     values["next_observation"], values["next_state"], values["terminated"])
    torch.testing.assert_close(replay.cpu_data["reward"], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(replay.hot_data["reward"], torch.arange(4, 8, dtype=torch.float32))
    saved = replay.state_dict()
    expected = replay.sample(6)
    restored = HybridMeasuredReplay(8, (1, 2), 3, seed=999, hot_device="cpu")
    restored.load_state_dict(saved)
    actual = restored.sample(6)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    assert restored.index == replay.index == 0
    assert len(restored) == len(replay) == 8


def test_pql_system_accepts_probe_replay_override_without_changing_default_path():
    from test_pql_system import pql_config
    from uav_network.experiments.cpu_replay_probe import HybridMeasuredReplay
    from uav_network.experiments.pql_system import PQLSystem
    c = pql_config()
    from uav_network.environment.observations import observation_dim, state_dim
    replay = HybridMeasuredReplay(c.algorithm_config.replay_capacity,
                                  (c.world.num_uavs, observation_dim(c)),
                                  state_dim(c), seed=c.runtime.seed, hot_device="cpu")
    engine = PQLSystem(c, replay=replay)
    try:
        assert engine.replay is replay
    finally:
        engine.close()


def test_probe_wandb_summary_update_uses_mapping_for_new_wandb_api():
    from uav_network.experiments.cpu_replay_probe import record_probe_summary
    class Summary:
        def __init__(self): self.values = {}
        def update(self, values): self.values.update(values)
    run = type("Run", (), {"summary": Summary()})()
    record_probe_summary(run, {"status": "passed"})
    assert run.summary.values == {
        "probe_passed": True,
        "probe_duplicated_replay": True,
        "episode_budget_complete": False,
    }
