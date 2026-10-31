from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from uav_network.configuration import Config, load_config
from uav_network.algorithms.checkpoints import make_learner, load_policy
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.tracking import episode_metrics
from uav_network.experiments.comparison_metrics import METRICS, MissionMetrics


ROOT = Path(__file__).resolve().parents[1]
XYZ_CONFIGS = [
    ROOT / "configs/comparison-20k-fullgpu-1kb-shared-actor-critic-xyz-sensing.yaml",
    ROOT / "configs/comparison-20k-hybrid75-1kb-shared-actor-critic-xyz-sensing.yaml",
    ROOT / "configs/comparison-20k-hybrid75-1kb-xyz-sensing.yaml",
]


def _episode_record():
    return {
        "episode": 1,
        "global_step": 2,
        "return": 0.0,
        "episode_length": 2,
        "search_rate_percent": 0.0,
        "delivery_rate_percent": 0.0,
        "success": False,
        "blocked_moves": 0,
        "boundary_clips": 0,
        "expired_reports": 0,
        "dropped_reports": 0,
        "energy": {"propulsion_energy_j": 0.0, "battery_fraction": [1.0], "active_uavs": 1},
        "network": {},
        "motion_counts": {
            "shield_interventions": 1,
            "shield_infeasible": 0,
            "shield_correction": 0.25,
            "commanded_stalls": 0,
        },
        "reward_components": {},
        "action_counts": {},
    }


def test_all_xyz_production_configs_select_cbf_and_keep_1500_steps():
    for path in XYZ_CONFIGS:
        c = load_config(path)
        assert c.world.motion_dimensions == 3
        assert c.world.motion_safety == "cbf", path.name
        assert c.world.max_steps == 1500


def test_cbf_is_valid_for_xyz_but_not_xy():
    xyz = load_config(XYZ_CONFIGS[1])
    xyz.world.motion_safety = "cbf"
    xyz.validate()

    xy = Config()
    xy.world.motion_safety = "cbf"
    try:
        xy.validate()
    except ValueError as exc:
        assert "XYZ" in str(exc) or "cbf" in str(exc).lower()
    else:
        raise AssertionError("2D/XY config must not silently select the XYZ CBF filter")


def test_cbf_episode_metrics_publish_canonical_names_and_keep_legacy_aliases():
    metrics = episode_metrics(_episode_record(), 1, motion_dim=3, motion_safety="cbf")
    assert metrics["safety/cbf_intervention_rate"] == 0.5
    assert metrics["safety/cbf_infeasible_rate"] == 0.0
    assert metrics["motion/cbf_correction"] == 0.25
    assert metrics["safety/shield_intervention_rate"] == 0.5
    assert metrics["motion/shield_correction"] == 0.25


def test_non_cbf_episode_metrics_do_not_claim_cbf_telemetry():
    metrics = episode_metrics(_episode_record(), 1, motion_dim=2, motion_safety="dynamic_window")
    assert "safety/cbf_intervention_rate" not in metrics
    assert "motion/cbf_correction" not in metrics
    assert metrics["safety/shield_intervention_rate"] == 0.5


def test_comparison_metrics_expose_cbf_canonical_and_legacy_aliases():
    assert "cbf_interventions_per_uav_step" in METRICS
    assert "cbf_correction_mean_per_step" in METRICS
    assert "shield_interventions_per_uav_step" in METRICS
    assert "shield_correction_mean_per_step" in METRICS


def test_explicit_legacy_dynamic_window_checkpoint_keeps_its_saved_mode():
    c = Config()
    c.matd3.hidden_sizes = [16, 16]
    c.runtime.algorithm = "matd3"
    c.world.motion_safety = "dynamic_window"
    learner = make_learner(c, observation_dim(c), state_dim(c))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "legacy-dwa.pt"
        learner.save(path)
        saved = torch.load(path, weights_only=True)
        assert saved["config"]["world"]["motion_safety"] == "dynamic_window"
        assert load_policy(path).config.world.motion_safety == "dynamic_window"


def test_cbf_training_diagnostics_use_canonical_warning_names(tmp_path):
    from uav_network.experiments.diagnostics import TrainingDiagnostics

    c = Config()
    c.world.motion_safety = "cbf"
    c.world.motion_dimensions = 3
    c.sensor.model = "bayesian"
    c.tracking.diagnostic_window = 1
    diagnostics = TrainingDiagnostics(c, tmp_path)
    record = _episode_record()
    record.update({"num_uavs": 1, "motion_dim": 3, "seed": 44, "termination_reason": "timeout"})
    values = diagnostics.episode(record)
    assert values["rolling/cbf_intervention_rate"] == 0.5
    assert values["rolling/shield_intervention_rate"] == 0.5
    assert "diagnostics/cbf_intervention_warning" in values


def test_gpu_cbf_fails_closed_until_warp_parity_exists():
    from uav_network.environment.gpu.state import params_from_config

    c = load_config(XYZ_CONFIGS[1])
    assert c.world.motion_safety == "cbf"
    try:
        params_from_config(c)
    except NotImplementedError as exc:
        assert "CBF" in str(exc) and "Warp" in str(exc)
    else:
        raise AssertionError("Warp must not silently execute a different safety mode for CBF configs")
