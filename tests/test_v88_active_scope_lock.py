"""Lock the currently approved scenario; explicit research ablations may override only with provenance."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml"

def test_active_scenario_is_not_replaced_by_historical_profile():
    s = yaml.safe_load(YAML.read_text())
    for key, expected in {
        "num_uavs": 6, "num_targets": 15, "map_size_m": 1000.0,
        "grid_w": 20, "uav_alt_m": 50.0, "safety_horizon_s": 900,
        "pd": 0.85, "pf": 0.03, "tx_power_w": 0.1,
    }.items():
        assert s[key] == expected, (key, s[key], expected)
    assert s["enable_tx_power_control"] is False

def test_default_runner_points_to_exact_active_yaml():
    run = (ROOT / "scripts/gpu/run_v81_real_rollout.py").read_text()
    assert 'scenario_v77_nfz_multi_1km.yaml' in run
    assert 'canonical_continuous' in run
    assert (ROOT / "docs/ACTIVE_SCENARIO_V88.md").is_file()
