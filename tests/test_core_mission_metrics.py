"""Five goal-aligned metrics, no reward or safety-based selection, no missing-data bias."""
from pathlib import Path
import numpy as np
import pytest

from uav_network.experiments.comparison_metrics import core_mission_metrics, plot_core_mission_metrics


def record(found, delivered, discovery_s, delivery_s, energy_j):
    return {"targets_found": found, "targets_delivered": delivered,
            "comparison_metrics": {
                "target_discovery_fraction": found/2,
                "gcs_delivery_fraction": delivered/2,
                "target_discovery_time_capped_mean_s": discovery_s,
                "target_delivery_time_capped_mean_s": delivery_s,
                "uav_energy_j": energy_j}}


def test_five_metrics_are_averaged_by_episode_and_energy_is_pooled(tmp_path):
    records = [record(2, 1, 2., 6., 100.), record(1, 0, 7., 10., 40.)]
    vals = core_mission_metrics(records)
    assert len(vals) == 5
    assert vals["discovery_rate_percent"] == 75.
    assert vals["gcs_delivery_rate_percent"] == 25.
    assert vals["discovery_time_capped_mean_s"] == 4.5
    assert vals["gcs_delivery_time_capped_mean_s"] == 8.
    assert vals["energy_per_delivered_target_j"] == 140.
    path = tmp_path / "five_metrics.png"
    assert Path(plot_core_mission_metrics(records,path)).stat().st_size > 500


def test_no_delivery_is_null_not_zero(tmp_path):
    vals = core_mission_metrics([record(0, 0, 10., 10., 99.)])
    assert vals["energy_per_delivered_target_j"] is None
    plot_core_mission_metrics([record(0, 0, 10., 10., 99.)], tmp_path / "none.png")
    assert (tmp_path / "none.png").is_file()


def test_empty_evaluation_rejected():
    with pytest.raises(ValueError):
        core_mission_metrics([])
