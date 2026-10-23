"""A finished training segment must expose failed continuation without secrets."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from test_cloud_continuation import parent_spec, learner_result, write_summary
from uav_network.cloud import continuation, remote


def test_insufficient_quota_waits_without_marking_learner_failed():
    parent = parent_spec()
    parent["overrides"].append("tracking.artifact_timeout_seconds=900")
    with TemporaryDirectory() as directory, \
         patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-token",
                                  "WANDB_API_KEY": "synthetic-key"}), \
         patch.object(continuation, "account_quotas", return_value=[
             {"name": "selected", "ok": True, "remaining_hours": .45}]), \
         patch.object(continuation, "run_kaggle", side_effect=AssertionError("must not push")):
        result = continuation.submit_continuation(parent, learner_result(), Path(directory))
        payload = (Path(directory) / "continuation.json").read_text()
        assert result.get("status") == "waiting_for_quota"
        assert result.get("wait_reason") == "insufficient_quota"
        assert result.get("remaining_gpu_hours") == .45
        assert not (Path(directory) / "continuation-failure.json").exists()
        assert "synthetic-token" not in payload and "synthetic-key" not in payload


def test_failed_handoff_is_visible_in_existing_wandb_run():
    results = learner_result()
    results["masac"]["wandb_run_id"] = "original"
    summary = {}
    run = SimpleNamespace(summary=summary)
    api = SimpleNamespace(run=lambda path: run)
    with TemporaryDirectory() as directory:
        output = Path(directory)
        write_summary(output, results)

        def wait(*args, **kwargs):
            record = {"status": "waiting_for_quota", "wait_reason": "insufficient_quota",
                      "remaining_gpu_hours": .2}
            (output / "continuation.json").write_text(json.dumps(record))
            return record

        with patch.dict("os.environ", {"WANDB_API_KEY": "synthetic-key"}), \
             patch("wandb.Api", return_value=api), \
             patch.object(continuation, "submit_continuation", side_effect=wait):
            remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output)
        assert summary.get("pipeline/status") == "waiting_for_quota"
        assert summary.get("pipeline/next_stage") == "gpu"
        assert summary.get("pipeline/training_complete") is False
        assert summary.get("pipeline/wait_reason") == "insufficient_quota"
        assert "pipeline/failure_reason" not in summary
        assert "secret" not in json.dumps(summary)
