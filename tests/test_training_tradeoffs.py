"""Training trade-offs must reflect completed training episodes, not validation."""
import pytest

from uav_network.experiments.telemetry import training_tradeoff_metrics
from uav_network.experiments.tracking import episode_metrics


def test_four_training_metrics_from_real_episode_fields():
    record = {
        "targets_found": 8, "targets_delivered": 5,
        "report_diagnostics": {"first_discovery_to_delivery_mean_s": 26.},
        "coverage_percent": 73.5,
        "sensing": {"measurements": 200, "false_confirmation": 3},
    }
    assert training_tradeoff_metrics(record) == {
        "training/delivery_given_discovery_fraction": 5/8,
        "training/first_discovery_to_gcs_delay_delivered_mean_s": 26.,
        "training/false_confirmation_per_1000_measurements": 15.,
        "training/map_coverage_percent": 73.5,
    }


def test_no_targets_no_measurements_no_report_are_not_false_zeros():
    result = training_tradeoff_metrics({
        "targets_found": 0, "targets_delivered": 0,
        "sensing": {"measurements": 0, "false_confirmation": 0},
        "report_diagnostics": {"first_discovery_to_delivery_mean_s": None},
        "coverage_percent": 0,
    })
    assert result["training/delivery_given_discovery_fraction"] is None
    assert result["training/first_discovery_to_gcs_delay_delivered_mean_s"] is None
    assert result["training/false_confirmation_per_1000_measurements"] is None
    assert result["training/map_coverage_percent"] == 0
    assert training_tradeoff_metrics({}) == {}


@pytest.mark.parametrize('payload', [
    {"targets_found": 2, "targets_delivered": 3},
    {"sensing": {"measurements": 1, "false_confirmation": 2}},
    {"coverage_percent": 101.},
    {"coverage_percent": float('nan')},
])
def test_inconsistent_training_metrics_are_rejected(payload):
    with pytest.raises(ValueError):
        training_tradeoff_metrics(payload)


def test_training_episode_logging_namespaced_correctly():
    record = {
        "episode": 17, "global_step": 4096, "return": -3., "episode_length": 20,
        "targets_found": 2, "targets_delivered": 1,
        "search_rate_percent": 20., "delivery_rate_percent": 10.,
        "success": False, "blocked_moves": 0, "boundary_clips": 0,
        "expired_reports": 0, "dropped_reports": 0,
        "energy": {"propulsion_energy_j": 100., "battery_fraction": [.8], "active_uavs": 1},
        "network": {"uav_communication_energy_j": 1.},
        "report_diagnostics": {"first_discovery_to_delivery_mean_s": 8., "delivery_count": 1},
        "sensing": {"measurements": 100, "false_confirmation": 1},
        "coverage_percent": 12.5,
    }
    values=episode_metrics(record, num_uavs=1)
    assert values["episode/index"] == 17
    assert values["training/delivery_given_discovery_fraction"] == .5
    assert values["training/first_discovery_to_gcs_delay_delivered_mean_s"] == 8.
    assert values["training/false_confirmation_per_1000_measurements"] == 10.
    assert values["training/map_coverage_percent"] == 12.5
    assert "evaluation/map_coverage_percent" not in values
