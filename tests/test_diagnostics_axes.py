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


def test_workspace_run_table_is_compact_and_decision_focused():
    from uav_network.experiments.dashboard import PANELS
    from uav_network.experiments.dashboard_columns import VISIBLE_COLUMNS

    expected = [
        "run:displayName", "run:state", "run:createdAt", "run:duration",
        "config:stage", "summary:evaluation/episodes_completed",
        "config:kaggle_account", "config:runtime.seed", "summary:global_step",
        "summary:train/episodes_completed", "summary:cohort/delivery_rate_percent_mean",
        "summary:cohort/success_mean", "summary:eval/mean_delivery_rate_percent",
        "summary:eval/success_rate", "summary:episode_budget_complete",
        "summary:stop_reason", "summary:diagnostics/failure_type",
    ]
    assert VISIBLE_COLUMNS == expected

    # Diagnostic detail stays available in charts without bloating the Runs table.
    hidden_from_runs = {
        "summary:train/updates", "summary:train/replay_size",
        "summary:learner/actor_loss", "summary:learner/critic_loss",
        "summary:learner/td_error_abs_mean", "summary:learner/q_abs_max",
        "summary:pql/transitions_per_s", "summary:pql/critic_updates_per_s",
        "summary:pql/queued_collections", "summary:cohort/episodes",
        "summary:evaluation/checkpoint_global_step",
    }
    assert hidden_from_runs.isdisjoint(VISIBLE_COLUMNS)

    all_metrics = {key for panels in PANELS.values() for _, metric in panels
                   for key in (metric if isinstance(metric, tuple) else (metric,))}
    assert {
        "train/episodes_completed", "train/joint_transitions_per_s",
        "cohort/episodes", "storage/disk_free_bytes", "learner/critic_loss", "learner/td_error_abs_mean",
        "cohort/network/packet_pdr_fraction", "cohort/delivery_rate_percent_mean",
        "episode/return", "rolling/return_std",
        "outcome/search_rate_percent", "outcome/delivery_rate_percent", "outcome/success", "rolling/success_rate",
        "learner/actor_loss", "system/gpu.0.gpu", "system/gpu.1.gpu", "recovery/elapsed_seconds",
    } <= all_metrics
    assert sum(len(panels) for panels in PANELS.values()) == 37


def test_workspace_evaluation_panels_and_columns_show_completed_outcome_source():
    from uav_network.experiments.dashboard import PANELS, panel_axis
    from uav_network.experiments.dashboard_columns import VISIBLE_COLUMNS
    assert "summary:cohort/return_mean" not in VISIBLE_COLUMNS
    assert "summary:cohort/delivery_rate_percent_mean" in VISIBLE_COLUMNS
    assert "summary:cohort/success_mean" in VISIBLE_COLUMNS
    assert panel_axis("episode/return") == "episode/index"
    assert panel_axis("outcome/search_rate_percent") == "episode/index"
    assert panel_axis("outcome/delivery_rate_percent") == "episode/index"
    assert panel_axis("outcome/success") == "episode/index"
    assert panel_axis("evaluation/mean_return") == "evaluation/checkpoint_global_step"
    assert panel_axis("eval/mean_return") == "evaluation/checkpoint_global_step"
    assert panel_axis("evaluation_status/failed") == "global_step"
    assert any(metric.startswith("cohort/") for panels in PANELS.values() for _, metric in panels)


def test_cpu_episode_metrics_define_episode_axis_without_changing_periodic_axis(tmp_path):
    run = OnlineRun()
    definitions = []
    run.define_metric = lambda name, **kwargs: definitions.append((name, kwargs))
    sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
    c = Config()
    c.tracking.credentials_file = str(tmp_path / 'absent.txt')
    with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, {
            'WANDB_API_KEY':'test-key','WANDB_ENTITY':'owner','WANDB_PROJECT':'project'}):
        tracker = OnlineTracker(c,tmp_path)
        tracker.close()
    for metric in ('evaluation/return','evaluation/search_rate_percent',
                   'evaluation/delivery_rate_percent','evaluation/success','evaluation/episodes_completed'):
        assert (metric, {'step_metric':'evaluation/episodes_completed','step_sync':True}) in definitions
    assert ('evaluation/*', {'step_metric':'evaluation/checkpoint_global_step','step_sync':True}) in definitions


def test_authored_cpu_episode_panels_use_episode_axis():
    from uav_network.experiments.dashboard import panel_axis
    for metric in ('evaluation/return','evaluation/search_rate_percent',
                   'evaluation/delivery_rate_percent','evaluation/success','evaluation/episodes_completed'):
        assert panel_axis(metric) == 'evaluation/episodes_completed'
    assert panel_axis('evaluation/mean_return') == 'evaluation/checkpoint_global_step'
    assert panel_axis('eval/mean_delivery_rate_percent') == 'evaluation/checkpoint_global_step'


def test_actor_loss_is_available_in_core_and_advanced_diagnostics():
    from uav_network.experiments.dashboard import PANELS, DIAGNOSTIC_PANELS
    assert any(metric=='learner/actor_loss' for panels in PANELS.values() for _,metric in panels)
    assert any(metric=='learner/actor_loss' for panels in DIAGNOSTIC_PANELS.values() for _,metric in panels)


def test_paper_and_benchmark_panels_use_scaled_axes_and_clean_rendering():
    from uav_network.experiments.dashboard import panel_axis, panel_options

    paper_title = "UAV-paper training return / reward"
    paper = panel_options(paper_title, "episode/return")
    assert panel_axis("episode/return", paper_title) == "episode/index"
    assert paper["smoothing_type"] == "exponential"
    assert paper["smoothing_factor"] == 0.99
    assert paper["smoothing_show_original"] is False
    assert paper["title_x"] == "Episode (×10²)"
    assert paper["xaxis_expression"] == "${episode/index} / 100"
    assert paper["point_visualization_method"] == "sampling"

    sample_title = "Sample-efficiency training return / reward"
    sample = panel_options(sample_title, "episode/return")
    assert panel_axis("episode/return", sample_title) == "global_step"
    assert sample["title_x"] == "Environment Steps (M)"
    assert sample["xaxis_expression"] == "${global_step} / 1000000"
    assert sample["point_visualization_method"] == "sampling"


    success_title = "UAV-paper mission success (%)"
    success = panel_options(success_title, "outcome/success")
    assert panel_axis("outcome/success", success_title) == "episode/index"
    assert success["title_x"] == "Episode (×10²)"
    assert success["xaxis_expression"] == "${episode/index} / 100"
    assert success["title_y"] == "Mission success (%)"
    assert success["custom_expressions"] == ["${outcome/success} * 100"]
    assert success["range_y"] == (0, 100)

    rolling_title = "Rolling mission success rate (%) - trend"
    rolling = panel_options(rolling_title, "rolling/success_rate")
    assert panel_axis("rolling/success_rate", rolling_title) == "global_step"
    assert rolling["title_x"] == "Environment Steps (M)"
    assert rolling["xaxis_expression"] == "${global_step} / 1000000"
    assert rolling["title_y"] == "Rolling success rate (%)"
    assert rolling["custom_expressions"] == ["${rolling/success_rate} * 100"]
    assert rolling["range_y"] == (0, 100)
    assert rolling["smoothing_type"] == "average"
    assert rolling["smoothing_show_original"] is True

    compute_title = "Compute-efficiency training return / reward"
    compute = panel_options(compute_title, "episode/return")
    assert panel_axis("episode/return", compute_title) == "_runtime"
    assert compute["title_x"] == "Wall Time (min)"
    assert compute["xaxis_expression"] == "${_runtime} / 60"
    assert compute["point_visualization_method"] == "sampling"

    loss = panel_options("Critic loss - diagnostic only", "learner/critic_loss")
    assert loss["smoothing_type"] == "average"
    assert loss["smoothing_factor"] == 0.8
    assert loss["smoothing_show_original"] is True
    raw = panel_options("Episode return / reward (raw audit)", "episode/return")
    assert raw == {"smoothing_type": "none"}
