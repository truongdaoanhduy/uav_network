from pathlib import Path
import hashlib
import json

from uav_network.configuration import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_power_support_diagnostic_is_bounded_and_matches_8k_resume_scientific_config():
    diagnostic = load_config(ROOT / "configs/diagnostic-masac-power-support.yaml")
    assert 8192 < diagnostic.runtime.episodes == 9216 < 20000
    assert diagnostic.runtime.max_train_seconds == 3600
    assert diagnostic.runtime.power_support_fraction == 0.0
    assert diagnostic.pql.enabled is True
    assert diagnostic.runtime.compile_models is False
    assert diagnostic.runtime.seed == 44
    assert diagnostic.runtime.eval_seed == 2000000044
    assert diagnostic.runtime.num_envs == 4096
    assert diagnostic.runtime.environment_backend == "warp_packet"
    assert diagnostic.runtime.replay_cpu_fraction == 0.75
    assert diagnostic.runtime.critic_chunk_size == 1024
    assert diagnostic.tracking.evaluation_interval_steps == 0
    groups = ("world", "sensor", "network", "energy", "reward", "masac", "matd3", "pql")
    values = diagnostic.to_dict()
    payload = json.dumps({key: values[key] for key in groups}, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == "bd029f3047257b9c4496b6b8556db093af5840cbd504aa93b5dded656f441fed"


def test_power_support_intervention_override_changes_only_runtime_diagnostic_field():
    path = ROOT / "configs/diagnostic-masac-power-support.yaml"
    baseline = load_config(path)
    intervention = load_config(path, ["runtime.power_support_fraction=0.10"])
    assert baseline.runtime.power_support_fraction == 0.0
    assert intervention.runtime.power_support_fraction == 0.10
    before, after = baseline.to_dict(), intervention.to_dict()
    after["runtime"]["power_support_fraction"] = before["runtime"]["power_support_fraction"]
    assert after == before
