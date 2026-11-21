"""Publish compact Validation/Test comparison to the ORIGINAL online W&B run.

This post-processing step runs only after ONESESSION has completed real GPU
validation and held-out test. It does not recalculate or manufacture metrics.
Validation is a per-checkpoint history; Test is one held-out measurement.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

METRICS = {
    "search_rate_percent": ("mean_search_rate_percent", None, "higher"),
    "delivery_rate_percent": ("mean_delivery_rate_percent", None, "higher"),
    "discovery_auc": (None, "discovery_auc", "higher"),
    "delivery_auc": (None, "delivery_auc", "higher"),
    "energy_per_delivered_target_j": (None, "energy_per_delivered_target_j", "lower"),
    "gcs_delivery_time_capped_mean_s": (None, "gcs_delivery_time_capped_mean_s", "lower"),
    "map_coverage_percent": (None, "map_coverage_percent", "higher"),
    "coverage_auc": (None, "coverage_auc", "higher"),
    "mission_success_rate_percent": (None, "mission_success_rate_percent", "higher"),
}


def checked_number(value, label):
    if value is None:
        return None  # undefined denominator: NOT zero
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Invalid {label}: not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Invalid {label}: nonfinite")
    return value


def comparison_values(source):
    if not isinstance(source, dict):
        raise ValueError("Missing genuine evaluation JSON")
    mission = source.get("mission_metrics")
    if not isinstance(mission, dict):
        raise ValueError("Missing measured mission metrics")
    result = {}
    for name, (top_level, mission_key, _direction) in METRICS.items():
        value = source.get(top_level) if top_level else mission.get(mission_key)
        result[name] = checked_number(value, name)
    return result


def load_completed(work, spec):
    work = Path(work)
    report = json.loads((work / "one-session-train-eval-test-report.json").read_text())
    evaluation = json.loads((work / "gpu_batch_eval" / "report.json").read_text())
    validation = json.loads((work / "gpu_batch_eval" / "validation_results.json").read_text())
    final = json.loads((work / "gpu_batch_eval" / "final_test.json").read_text())
    algorithm = next(iter(spec["runs"]))
    run_id = spec["runs"][algorithm]["id"]
    # Full snapshot archive (117) and deliberately preselected Validation bank
    # (32) are DISTINCT populations. The old code erroneously equated them.
    protocol = spec.get("wandb_logging_protocol") or {}
    expected_val = int(protocol.get("validation_episodes", 0))
    expected_test = int(protocol.get("final_test_episodes", 0))
    expected_candidates = int(report.get("validation_candidates", 0))
    full_snapshot_count = int(report.get("snapshots", 0))
    if (expected_val < 1 or expected_test < 1 or expected_candidates < 1
            or expected_candidates > full_snapshot_count):
        raise RuntimeError("Missing/inconsistent predeclared Validation/Test budgets")
    if (report.get("algorithm") != algorithm
            or report.get("wandb_run_id") != run_id
            or report.get("gpu_val_status") != "complete"
            or report.get("gpu_test_status") != "complete"
            or report.get("snapshots_pruned", 0) != 0
            or int(report.get("validation_seed_count", -1)) != expected_val
            or evaluation.get("validation_completed") != expected_candidates
            or len(validation) != expected_candidates
            or any(int(row.get("episodes", -1)) != expected_val for row in validation)
            or evaluation.get("final_test", {}).get("episodes") != expected_test
            or final.get("episodes") != expected_test
            or int(report.get("test_episodes", -1)) != expected_test):
        raise RuntimeError("Training/Validation/Test incomplete; NEVER publish success")
    protocol = final.get("protocol", {})
    if (protocol.get("training_seed_overlap") is not False
            or protocol.get("validation_seed_overlap") is not False):
        raise RuntimeError("Final Test must use the held-out seed bank")
    best_step = int(evaluation["best_global_step"])
    best_rows = [row for row in validation
                 if int(row["source"]["global_step"]) == best_step]
    if len(best_rows) != 1:
        raise ValueError("Best checkpoint not unique in full Validation history")
    rows = sorted(validation, key=lambda r: (
        int(r["source"]["global_step"]), int(r["source"]["policy_version"])))
    steps = [int(row["source"]["global_step"]) for row in rows]
    if len(set(steps)) != len(steps):
        raise ValueError("Checkpoint global steps must be unique for the plot x-axis")
    if not all(int(row.get("episodes", 0)) == expected_val for row in rows):
        raise ValueError("Validation episodes do not match predeclared fixed seed count")
    return {
        "algorithm": algorithm, "run_id": run_id, "best_global_step": best_step,
        "train_episodes": int(report["training_completed_episodes"]),
        "validation": rows, "validation_episodes": expected_val,
        "snapshot_archive_count": full_snapshot_count,
        "best_validation": comparison_values(best_rows[0]),
        "test": comparison_values(final),
        "test_episodes": int(final["episodes"]),
        "test_mean_return": checked_number(final.get("mean_return"), "mean_return"),
    }


def local_comparison_table(data, work):
    directory = Path(work) / "gpu_batch_eval"
    path = directory / "wandb_comparison.csv"
    columns = ["phase", "global_step", "episodes", *METRICS]
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "phase": "validation_best",
            "global_step": data["best_global_step"],
            "episodes": data["validation_episodes"], **data["best_validation"]})
        writer.writerow({
            "phase": "held_out_test",
            "global_step": data["best_global_step"],
            "episodes": data["test_episodes"], **data["test"]})
    (directory / "wandb_comparison.json").write_text(json.dumps({
        "algorithm": data["algorithm"], "best_global_step": data["best_global_step"],
        "validation_best": data["best_validation"], "test": data["test"],
        "comparison_directions": {k: v[2] for k, v in METRICS.items()},
    }, indent=2, allow_nan=False))
    return path


def publish(work, spec):
    data = load_completed(work, spec)
    local_comparison_table(data, work)
    import wandb
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY Kaggle Secret not granted")
    entity, project = spec["entity"], spec["project"]
    run_id = data["run_id"]
    url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
    for attempt in range(1, 4):
        try:
            with wandb.init(
                    entity=entity, project=project, id=run_id,
                    resume="must", reinit="finish_previous", mode="online",
                    settings=wandb.Settings(init_timeout=90)) as run:
                # Explicit x axis to avoid mixing with the training run's global_step.
                run.define_metric("eval/checkpoint_global_step")
                for metric in METRICS:
                    run.define_metric("eval/" + metric,
                                      step_metric="eval/checkpoint_global_step")
                for sample in data["validation"]:
                    payload = {
                        "eval/checkpoint_global_step": int(sample["source"]["global_step"]),
                        "eval/checkpoint_episode": int(sample["source"]["episodes"]),
                        "eval/validation_episodes": int(sample["episodes"]),
                    }
                    for name, val in comparison_values(sample).items():
                        if val is not None:
                            payload["eval/" + name] = val
                    run.log(payload)
                # Test is ONE held-out bank, not a training curve.
                run.define_metric("test/checkpoint_global_step")
                for metric in METRICS:
                    run.define_metric("test/" + metric,
                                      step_metric="test/checkpoint_global_step")
                run.log({
                    "test/checkpoint_global_step": data["best_global_step"],
                    "test/episodes": data["test_episodes"],
                    **{"test/" + k: v for k, v in data["test"].items() if v is not None},
                })
                # Test is one measured held-out bank. Publish one history
                # point for the existing gpu_final/* W&B charts, in addition
                # to the final summary and test/* comparison.
                test_to_gpu_final = {
                    "search_rate_percent": "discovery_rate_percent",
                    "delivery_rate_percent": "gcs_delivery_rate_percent",
                    "discovery_auc": "discovery_auc",
                    "delivery_auc": "delivery_auc",
                    "energy_per_delivered_target_j": "energy_per_delivered_target_j",
                    "gcs_delivery_time_capped_mean_s": "gcs_delivery_time_capped_mean_s",
                    "map_coverage_percent": "map_coverage_percent",
                    "coverage_auc": "coverage_auc",
                    "mission_success_rate_percent": "mission_success_rate_percent",
                }
                final_numeric = {
                    "gpu_final/" + new: data["test"][old]
                    for old, new in test_to_gpu_final.items()
                    if data["test"].get(old) is not None
                }
                if data["test_mean_return"] is not None:
                    final_numeric["gpu_final/mean_return"] = data["test_mean_return"]
                run.define_metric("gpu_final/validation_best_global_step")
                run.define_metric(
                    "gpu_final/*", step_metric="gpu_final/validation_best_global_step")
                run.log({
                    "gpu_final/validation_best_global_step": data["best_global_step"],
                    "gpu_final/episodes": data["test_episodes"],
                    **final_numeric,
                })
                # Summary is the definitive side-by-side table in W&B Run.
                run.summary.update({
                    "gpu_final/episodes": data["test_episodes"],
                    "gpu_final/validation_best_global_step": data["best_global_step"],
                    **final_numeric,
                    "pipeline/training_finished": True,
                    "pipeline/validation_finished": True,
                    "pipeline/test_finished": True,
                    "pipeline/online_comparison_published": True,
                    "pipeline/session_ceiling_seconds": 42600,
                    "pipeline/training_episodes_completed": data["train_episodes"],
                    "eval/validated_checkpoints": len(data["validation"]),
                    "eval/episodes_per_checkpoint": data["validation_episodes"],
                    "eval/best_global_step": data["best_global_step"],
                    "test/episodes": data["test_episodes"],
                    "test/best_global_step": data["best_global_step"],
                    **{"eval/best_" + k: v for k, v in data["best_validation"].items()
                       if v is not None},
                    **{"test/" + k: v for k, v in data["test"].items() if v is not None},
                })
                columns = ["phase", "episodes", *METRICS]
                values = [
                    ["Validation best", data["validation_episodes"],
                     *[data["best_validation"][k] for k in METRICS]],
                    ["Held-out Test", data["test_episodes"],
                     *[data["test"][k] for k in METRICS]],
                ]
                run.log({"comparison/eval_vs_test": wandb.Table(
                    columns=columns, data=values)})
            # Verify the same run's cloud Summary; no new W&B run is created.
            verified = False
            for _ in range(3):
                online = wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
                summary = online.summary
                if (summary.get("pipeline/validation_finished") is True
                        and summary.get("pipeline/test_finished") is True
                        and int(summary.get("test/episodes", 0)) == data["test_episodes"]
                        and int(summary.get("gpu_final/episodes", 0)) == data["test_episodes"]):
                    verified = True
                    break
                time.sleep(4)
            if not verified:
                raise RuntimeError("W&B cloud summary not yet confirmed")
            receipt = {
                "status": "completed", "algorithm": data["algorithm"], "wandb_run": url,
                "validation_checkpoints": len(data["validation"]),
                "held_out_test_episodes": data["test_episodes"],
                "best_global_step": data["best_global_step"],
                "metrics": list(METRICS),
            }
            (Path(work) / "gpu_batch_eval" / "wandb_publish_receipt.json").write_text(
                json.dumps(receipt, indent=2))
            print("WANDB_FULLFLOW_EVAL_TEST_CONFIRMED", json.dumps(receipt), flush=True)
            return receipt
        except Exception as exc:
            print("WANDB_COMPARISON_PUBLISH_RETRY", attempt, type(exc).__name__,
                  str(exc)[:180], flush=True)
            if attempt == 3:
                raise RuntimeError(
                    "GPU Eval/Test finished locally, but online W&B not confirmed. "
                    "See /kaggle/working/gpu_batch_eval/ for recoverable results."
                ) from exc
            time.sleep(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    publish(args.work, json.loads(args.spec.read_text()))


if __name__ == "__main__":
    main()
