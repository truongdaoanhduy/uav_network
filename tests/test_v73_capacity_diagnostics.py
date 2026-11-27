"""Behavioral guards for declared sizes, durable failures and real learner updates."""
import json
from pathlib import Path

import pytest
import torch

from uav_network.algorithms.configuration.loader import load_learner_config
from zhang_xue.gpu.runtime_v67 import HAS_WARP

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "configs/learners/e4096_b8192.yaml"
SCENARIO = ROOT / "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml"


def test_target_profile_reaches_consumer_without_size_fallback():
    cfg = load_learner_config(TARGET)
    assert cfg.runtime.num_envs == 4096
    assert cfg.matd3.batch_size == cfg.masac.batch_size == 8192
    assert cfg.continuous_dim == 2
    assert cfg.network.destination_mode == "compact_3"
    # A typed CLI override must actually control the same consumed setting.
    assert load_learner_config(TARGET, overrides=["runtime.num_envs=17"]).runtime.num_envs == 17


@pytest.mark.parametrize("value", ["0", "-1", "true", "2.5", "'4096'"])
def test_invalid_environment_count_rejected_before_allocation(value):
    with pytest.raises(ValueError, match="num_envs"):
        load_learner_config(overrides=[f"runtime.num_envs={value}"])


def test_phase_begin_survives_failure_and_is_readable_before_risky_work(tmp_path):
    from uav_network.experiments.resource_logging import ResourceJournal

    journal = ResourceJournal(tmp_path, devices=["cpu"])
    with pytest.raises(MemoryError, match="capacity exhausted"):
        with journal.phase("replay_allocate", capacity=50_000):
            rows = [json.loads(x) for x in (tmp_path / "resources.jsonl").read_text().splitlines()]
            assert rows[-1]["event"] == "phase_begin"
            assert rows[-1]["phase"] == "replay_allocate"
            assert rows[-1]["memory"]["host"]["rss_bytes"] > 0
            raise MemoryError("capacity exhausted")
    rows = [json.loads(x) for x in (tmp_path / "resources.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "phase_error"
    assert rows[-1]["error_type"] == "MemoryError"
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "replay_allocate"


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang is not installed: CPU Warp functional gate unavailable")
@pytest.mark.parametrize("algorithm,actor_updates", [("matd3", 1), ("masac", 2)])
def test_small_functional_probe_executes_real_updates_and_discloses_repeated_fixture(tmp_path, algorithm, actor_updates):
    from uav_network.experiments.capacity_probe import run_probe

    torch.set_num_threads(2)
    cfg = load_learner_config(TARGET)
    out = tmp_path / algorithm
    result = run_probe(cfg, scenario_path=SCENARIO, output_dir=out,
                       mode="functional", algorithm=algorithm,
                       network_device="cpu", learner_device="cpu",
                       functional_envs=2, functional_batch=4, functional_hidden=32)
    assert result["status"] == "FUNCTIONAL_PASS_NOT_FULL_RL"
    assert result["requested"]["num_envs"] == 4096
    assert result["requested"]["batch_size"] == 8192
    assert result["effective"]["num_envs"] == 2
    assert result["effective"]["batch_size"] == 4
    assert result["effective"]["observation_shape"] == [6, 518]
    assert result["effective"]["state_dim"] == 3108
    assert result["fixture"]["real_cpu_steps"] == 4
    assert result["fixture"]["repeated_for_capacity"] is True
    assert result["full_gpu_rl_training"] is False
    assert result["updates"] == 2
    assert result["actor_updates"] == actor_updates
    assert result["actor_parameter_delta_l2"] > 0
    assert result["sample_shapes"]["observation"] == [4, 6, 518]
    assert result["warp_packet_workload_executed"] is True
    resolved = json.loads((out / "scenario_config.json").read_text())
    assert resolved["num_uavs"] == 6
    assert resolved["map_size_m"] == 1000.0
    assert resolved["comm_action_mode"] == "compact_3"
    assert len(result["provenance"]["scenario_file_sha256"]) == 64
    assert len(result["provenance"]["source_tree_sha256"]) == 64
    assert json.loads((out / "result.json").read_text())["status"] == result["status"]
    rows = [json.loads(x) for x in (out / "resources.jsonl").read_text().splitlines()]
    assert any(x["event"] == "phase_end" and x["phase"] == "learner_update_2" for x in rows)


def test_capacity_refuses_silent_cpu_fallback_and_persists_reason(tmp_path):
    from uav_network.experiments.capacity_probe import run_probe

    cfg = load_learner_config(TARGET)
    result = run_probe(cfg, scenario_path=SCENARIO, output_dir=tmp_path,
                       mode="capacity", algorithm="matd3",
                       network_device="cpu", learner_device="cpu")
    assert result["status"] == "BLOCKED_CUDA_REQUIRED"
    assert result["requested"]["num_envs"] == 4096
    assert result["requested"]["batch_size"] == 8192
    assert result["failure_phase"] == "hardware_preflight"
    assert json.loads((tmp_path / "result.json").read_text())["status"] == result["status"]


@pytest.mark.parametrize("algorithm", ["matd3", "masac"])
def test_v72_version4_checkpoint_without_env_count_still_loads_identical_weights(tmp_path, algorithm):
    from uav_network.algorithms.core.checkpoints import make_learner, load_policy

    cfg = load_learner_config(TARGET)
    cfg.runtime.algorithm = algorithm
    getattr(cfg, algorithm).hidden_sizes = [16, 16]
    learner = make_learner(cfg, 518, 3108)
    path = tmp_path / "v72.pt"
    learner.save(path)
    payload = torch.load(path, weights_only=True)
    del payload["config"]["runtime"]["num_envs"]
    torch.save(payload, path)
    restored = load_policy(path)
    assert restored.config.runtime.num_envs == 1
    for name, value in learner.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    # An unrelated missing field remains a schema error, not a broad migration.
    del payload["config"]["runtime"]["amp_mode"]
    with pytest.raises(ValueError, match="runtime"):
        load_policy(path, payload=payload)


@pytest.mark.parametrize("phase,message,expected", [
    ("warp_radio_allocate_reset", "Failed to allocate 123456 bytes on device 'cuda:0'", "OOM"),
    ("warp_radio_allocate_reset", "CUDA kernel launch failed: illegal memory access", "FAIL"),
    ("learner_update_1", "Failed to allocate 123456 bytes on device 'cuda:0'", "FAIL"),
])
def test_warp_allocation_failure_is_distinguished_from_kernel_failure(phase, message, expected):
    from uav_network.experiments.capacity_probe import classify_failure

    assert classify_failure(RuntimeError(message), phase) == expected
