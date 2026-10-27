"""Publish scheduling outcome separately from a finished learner segment."""
import json
import os
from pathlib import Path


def publish_pipeline_status(spec, results, output, *, failed, next_stage):
    path = Path(output) / "continuation.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    waiting = not failed and record.get("status") == "waiting_for_quota"
    external = not failed and record.get("status") == "external_pending"
    payload = {"pipeline/status": ("failed" if failed else "waiting_for_quota" if waiting
                                   else "awaiting_external_supervisor" if external else "submitted"),
               "pipeline/next_stage": next_stage,
               "pipeline/training_complete": all(
                   r.get("episode_budget_complete") is True for r in results.values()) and bool(results)}
    if waiting:
        reason = record.get("wait_reason")
        if reason in {"quota_unavailable", "insufficient_quota"}:
            payload["pipeline/wait_reason"] = reason
    elif path.is_file():
        # These are internal reason codes; never copy arbitrary exception text.
        reason = record.get("failure_reason")
        if reason in {"quota_unavailable", "insufficient_quota", "credential_verification_failed", "submission_failed"}:
            payload["pipeline/failure_reason"] = reason
    (Path(output) / "pipeline-status.json").write_text(json.dumps(payload, indent=2))
    if not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb
        api = wandb.Api(timeout=15)
        for result in results.values():
            run_id = result.get("wandb_run_id")
            if run_id:
                run = api.run(f"{spec['entity']}/{spec['project']}/{run_id}")
                run.summary.update(payload)
    except Exception as error:
        # Tracking outages must not turn a successful submission into a duplicate retry.
        (Path(output) / "pipeline-tracking-failure.json").write_text(
            json.dumps({"error_type": type(error).__name__}))
