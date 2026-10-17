"""W&B arrival order and source checkpoint time are separate public axes."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.configuration import Config, load_config
from uav_network.experiments.tracking import OnlineTracker
from test_tracking import OnlineRun


def test_evaluation_metrics_define_checkpoint_axis_after_global_wildcard(tmp_path):
    run = OnlineRun()
    definitions = []
    run.define_metric = lambda name, **kwargs: definitions.append((name, kwargs))
    sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
    c = Config()
    c.tracking.credentials_file = str(tmp_path / "absent.txt")
    with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
        "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
        tracker = OnlineTracker(c, tmp_path)
        tracker.diagnostic({"evaluation/checkpoint_global_step": 10, "evaluation/mean_return": 2.}, 100)
        tracker.diagnostic({"evaluation/checkpoint_global_step": 20, "evaluation/mean_return": 3.}, 200)
        tracker.close()
    wildcard = definitions.index(("*", {"step_metric": "global_step", "step_sync": True}))
    for prefix in ("evaluation/*", "eval/*"):
        definition = (prefix, {"step_metric": "evaluation/checkpoint_global_step", "step_sync": True})
        assert definitions.index(definition) > wildcard
    assert [values["global_step"] for values in run.logged] == [100, 200]


def test_cpu_visualization_launcher_preserves_transition_validation_and_final_test30():
    from uav_network.cloud.remote import learner_command
    spec = {"config": "configs/comparison-25k-diagnostics.yaml",
            "overrides": [], "cpu_visualize": True}
    command, _ = learner_command(spec, "masac", {"id": "test", "name": "test"}, 0, "outputs/test")
    overrides = [command[i + 1] for i, value in enumerate(command) if value == "--set"]
    assert "tracking.evaluation_interval=0" in overrides
    assert not any(value.startswith("tracking.evaluation_interval_steps=") for value in overrides)
    c = load_config(spec["config"], overrides=overrides)
    assert c.tracking.evaluation_interval_steps == 7372800
    assert c.tracking.periodic_evaluation_episodes == 5
    assert c.runtime.eval_episodes == 30
    assert c.runtime.eval_seed == 2000000044


def test_workspace_evaluation_panels_and_columns_show_completed_outcome_source():
    from uav_network.experiments.dashboard import PANELS, panel_axis
    from uav_network.experiments.dashboard_columns import VISIBLE_COLUMNS
    assert "summary:cohort/return_mean" in VISIBLE_COLUMNS
    assert "summary:cohort/episodes" in VISIBLE_COLUMNS
    assert "summary:evaluation/mean_delivery_rate_percent" in VISIBLE_COLUMNS
    assert panel_axis("evaluation/mean_return") == "evaluation/checkpoint_global_step"
    assert panel_axis("eval/mean_return") == "evaluation/checkpoint_global_step"
    assert panel_axis("evaluation_status/failed") == "global_step"
    assert any(metric.startswith("cohort/") for panels in PANELS.values() for _, metric in panels)
