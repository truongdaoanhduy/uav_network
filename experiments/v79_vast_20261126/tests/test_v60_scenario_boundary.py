"""Red/green guard for replacing the earlier scenario without changing algorithms."""
from __future__ import annotations
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_single_canonical_seven_radio_scenario():
    # Root configuration is now strictly separated by responsibility.
    # Old scenario/world YAML at root is still forbidden.
    assert not list((ROOT / "configs").glob("*.yaml")), "Old root scenario YAML must be retired"
    assert {p.name for p in (ROOT / "configs").iterdir()} <= {"data", "learners"}
    scenario = ROOT / "research/zhang_xue"
    assert (scenario / "configs/scenario_7radio.yaml").exists()
    assert (scenario / "configs/cpu_smoke.yaml").exists()
    assert not (ROOT / "research/zhang_xue_v58_20261125").exists()


def test_old_training_pipeline_and_async_variant_absent():
    assert not (ROOT / "uav_network/environment").exists()
    assert not (ROOT / "uav_network/communication").exists()
    assert not (ROOT / "uav_network/cloud").exists()
    assert not (ROOT / "uav_network/configuration.py").exists()
    assert not (ROOT / "uav_network/experiments/training.py").exists()
    assert not any((ROOT / "uav_network").rglob("*" + "pq" + "l" + "*.py"))


def test_kept_masac_matd3_and_generic_metrics_import_cleanly():
    from uav_network.algorithms.matd3 import MATD3, td3_target
    from uav_network.algorithms.masac import MASAC, soft_bellman_target
    from uav_network.algorithms.vanilla_baselines import VanillaMATD3, VanillaMASAC
    from uav_network.experiments.dual_fidelity_eval import EpisodeKPI, aggregate_kpis
    from uav_network.experiments.telemetry import diagnostic_rates
    assert MATD3.algorithm == "matd3"
    assert MASAC.algorithm == "masac"
    assert VanillaMATD3 is not None and VanillaMASAC is not None
    assert callable(aggregate_kpis) and callable(diagnostic_rates)


def test_no_stale_deployed_python_refs():
    kept = ROOT / "uav_network"
    forbidden = "pq" + "l"
    for file in kept.rglob("*.py"):
        with file.open(encoding="utf-8") as f:
            contents = f.read()
        assert forbidden not in contents.lower(), file
        ast.parse(contents)


def test_root_source_and_tests_governed_500_lines():
    for root in ("uav_network", "tests", "scripts", "research/zhang_xue/src"):
        p = ROOT / root
        for file in p.rglob("*.py"):
            assert len(file.read_text().splitlines()) <= 500, file


def test_release_source_has_no_old_config_or_notebooks():
    from scripts.maintenance.source_inventory import source_paths
    tracked = source_paths(ROOT)
    assert not any(
        name.startswith("configs/")
        and not name.startswith(("configs/data/", "configs/learners/"))
        for name in tracked
    )
    assert not any(name.endswith(".ipynb") for name in tracked)
    assert not any(name.startswith("third_party/") for name in tracked)
