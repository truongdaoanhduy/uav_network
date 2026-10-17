"""End-to-end tiny learners keep training identical while CPU evaluation arrives."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from uav_network.experiments.training import train
from test_async_evaluation import settings, wait_idle


@pytest.mark.parametrize("algorithm,pql", [("masac", False), ("matd3", False), ("masac", True), ("matd3", True)])
def test_real_base_and_pql_train_evaluate_and_retain_bounded_checkpoint_files(tmp_path, algorithm, pql):
    c = settings()
    c.runtime.algorithm, c.pql.enabled = algorithm, pql
    c.runtime.episodes = 4
    c.runtime.checkpoint_interval = 1
    c.runtime.log_interval_steps = 1
    c.algorithm_config.learning_starts = 2
    c.pql.mode = "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps = 1
    c.pql.critic_sample_ratio = 1
    c.runtime.output_dir = str(tmp_path / "evaluated")
    # CPU Warp exercises the vector information contract in PQL without CUDA.
    if pql:
        c.runtime.environment_backend = "warp_packet"
    from uav_network.experiments.async_evaluation import AsyncCheckpointEvaluator
    def manager(*args, **kwargs):
        instance = AsyncCheckpointEvaluator(*args, **kwargs)
        tick = instance.tick
        def await_first_result(step, *args, **kwargs):
            tick(step, *args, **kwargs)
            # Observe the real child before the tiny training run ends. Production
            # shutdown stays bounded at 10 s; cold imports can exceed that in CI.
            if instance.active is not None and not instance.counts["completed"]:
                wait_idle(instance, step=step)
        instance.tick = await_first_result
        return instance
    with patch("uav_network.experiments.training.AsyncCheckpointEvaluator", side_effect=manager), \
         patch("uav_network.experiments.pql_training.AsyncCheckpointEvaluator", side_effect=manager):
        result = train(c)
    assert result["updates"] > 0
    assert result["periodic_evaluation"]["completed"] >= 1
    files = sorted((tmp_path / "evaluated" / "evaluations").glob("*/metrics.json"))
    assert files
    evaluated = json.loads(files[0].read_text())
    assert evaluated["source"]["global_step"] == 3  # no episode finishes here
    assert evaluated["source"]["policy_version"] > 0
    assert evaluated["source"]["episodes"] == 1
    assert evaluated["episodes"] == 2
    assert [r["seed"] for r in evaluated["records"]] == [1000000044, 1000000045]
    assert "report_diagnostics" in evaluated["records"][0]
    assert len(list((tmp_path / "evaluated").glob("episode_*.pt"))) <= 2
    assert result["storage"]["storage/disk_free_bytes"] > 0
    assert result["replay"]["replay/policy_age_available"] == 0
    assert result["replay"]["replay/sample_age_measured_uses"] > 0
    no_eval = deepcopy(c)
    no_eval.tracking.evaluation_interval_steps = 0
    no_eval.runtime.output_dir = str(tmp_path / "control")
    control = train(no_eval)
    assert result["reproducibility"]["model_hash"] == control["reproducibility"]["model_hash"]
    assert result["reproducibility"]["episode_hash"] == control["reproducibility"]["episode_hash"]


def test_base_collects_full_completion_batch_before_logging_window(tmp_path):
    c = settings()
    c.runtime.episodes = c.runtime.num_envs = 2
    c.world.max_steps = 2
    c.runtime.output_dir = str(tmp_path / "batched")
    c.tracking.evaluation_interval_steps = 0
    logged = []
    tracker = SimpleNamespace(learner=lambda *args: None, episode=lambda *args: None,
        diagnostic=lambda metrics, step: logged.append((step, metrics)),
        progress=lambda *args, **kwargs: None, checkpoint=lambda *args, **kwargs: None,
        failure=lambda *args: None, identity=lambda: {"wandb_run_id": "test"},
        complete=lambda *args: {}, close=lambda **kwargs: None)
    with patch("uav_network.experiments.training.start_tracking", return_value=tracker):
        train(c)
    window = [v for step, v in logged if "cohort/episodes" in v]
    assert len(window) == 1
    assert window[0]["cohort/episodes"] == 2
    assert window[0]["cohort/source_end_step"] == 4


@pytest.mark.parametrize("pql", [False, True])
def test_training_error_cancels_and_reaps_live_evaluator_before_exit(tmp_path, pql):
    import sys
    from uav_network.algorithms.masac import MASAC
    from uav_network.experiments.async_evaluation import AsyncCheckpointEvaluator
    c = settings()
    c.runtime.output_dir = str(tmp_path / "failed")
    c.runtime.episodes = 4
    c.algorithm_config.learning_starts = 2
    c.tracking.evaluation_interval_steps = 1
    c.pql.enabled, c.pql.mode = pql, "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps = 1
    c.pql.critic_sample_ratio = 1
    instances = []
    def manager(*args, **kwargs):
        instance = AsyncCheckpointEvaluator(*args, **kwargs)
        instance.command = lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"]
        instances.append(instance)
        return instance
    update = MASAC.update
    def fail_after_submit(self, *args, **kwargs):
        if self.updates >= 1:
            raise FloatingPointError("controlled learner failure with live evaluator")
        return update(self, *args, **kwargs)
    with patch("uav_network.experiments.training.AsyncCheckpointEvaluator", side_effect=manager), \
         patch("uav_network.experiments.pql_training.AsyncCheckpointEvaluator", side_effect=manager), \
         patch.object(MASAC, "update", fail_after_submit):
        with pytest.raises(FloatingPointError, match="controlled learner failure"):
            train(c)
    assert instances[0].active is None
    assert instances[0].counts["cancelled"] == 1
    assert instances[0].closed


@pytest.mark.parametrize("pql", [False, True])
def test_episode_budget_cut_marks_final_completion_window_partial(tmp_path, pql):
    c = settings()
    c.runtime.output_dir = str(tmp_path / "budget-cut")
    c.runtime.environment_backend = "warp_packet"
    c.runtime.num_envs, c.runtime.episodes = 4, 6
    c.runtime.checkpoint_interval = 128
    c.tracking.evaluation_interval_steps = 0
    c.pql.enabled, c.pql.mode = pql, "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    result = train(c)
    windows = [json.loads(line) for line in (tmp_path / "budget-cut" / "cohorts.jsonl").read_text().splitlines()]
    assert result["episodes"] == 6
    assert [v["cohort/episodes"] for v in windows] == [4, 2]
    assert [v["cohort/partial"] for v in windows] == [0, 1]
    assert windows[-1]["diagnostics/return_drop_warning"] == 0
