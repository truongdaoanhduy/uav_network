"""One Kaggle session: train, predeclared metadata-only GPU shortlist, validate, Best, held-out Final Test."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .deferred_gpu_evaluation import verify_manifest


def run(spec_file, *, policies_per_batch=2, final_test_batch_size=8,
        eval_world_pool_size=0, max_eval_worlds_per_gpu=0,
        max_test_worlds_per_gpu=0, work_dir=None):
    spec = json.loads(Path(spec_file).read_text())
    if (spec.get("stage") != "train" or len(spec.get("runs", {})) != 1
            or not set(spec["runs"]).issubset({"matd3", "masac"})
            or spec.get("gpus_per_run") != 2 or not spec.get("training_only")):
        raise ValueError("One-session flow requires exactly one MASAC or MATD3 training job on two GPUs")
    algorithm = next(iter(spec["runs"]))
    run_id = spec["runs"][algorithm]["id"]
    work = Path(work_dir) if work_dir is not None else Path("/kaggle/working")
    train_folder = work / "train" / algorithm
    result_path = train_folder / "summary.json"
    if not result_path.is_file():
        raise FileNotFoundError("Training must complete and write its summary before GPU evaluation")
    training = json.loads(result_path.read_text())
    pilot_steps = spec.get("matched_pilot_env_steps", 0)
    if pilot_steps and (training.get("requested_env_steps") != pilot_steps
                        or training.get("stop_reason") != "env_step_budget"
                        or training.get("global_step") != pilot_steps):
        raise RuntimeError("Training did not reach the matched environment-step budget")
    if not pilot_steps and int(training.get("episodes", 0)) < 1:
        raise RuntimeError("No complete training episodes; refusing GPU validation")
    if spec.get("require_full_episode_budget") and not training.get("episode_budget_complete"):
        raise RuntimeError("20,000-episode training incomplete; refusing best selection/final test")
    if not pilot_steps and spec.get("episode_budget_only") and not training.get("episode_budget_complete"):
        raise RuntimeError("Training did not finish all requested episodes; cannot perform full-flow validation")
    folder = train_folder / "deferred-gpu-eval"
    manifest = verify_manifest(folder)
    if not manifest["count"] or manifest.get("pruned") != 0 or not manifest.get("keep_all_distinct_versions"):
        raise RuntimeError("All-snapshot manifest must be complete, unpruned and keep_all")
    if manifest["count"] != manifest.get("collected"):
        raise RuntimeError("Incomplete snapshot bank; refusing to choose a best checkpoint")
    full_manifest = manifest
    if pilot_steps:
        stride = manifest["final_stride"] * training["num_envs"]
        expected = list(range(0, pilot_steps + 1, stride))
        if expected[-1] != pilot_steps:
            expected.append(pilot_steps)
        observed = [row["global_step"] for row in manifest["snapshots"]]
        if observed != expected:
            raise RuntimeError(f"Unmatched validation checkpoint grid: {observed}, expected {expected}")
        candidate_method = "fixed_environment_steps"
    else:
        from .session_shortlist import create_shortlist
        shortlist_dir = train_folder / "deferred-gpu-eval-session-shortlist"
        step_targets = spec.get("validation_step_targets")
        manifest = create_shortlist(
            folder, shortlist_dir, limit=int(spec.get("gpu_eval_candidate_limit", 32)),
            step_targets=step_targets)
        folder = shortlist_dir
        candidate_method = manifest["shortlist_provenance"]["method"]
    status = {
        "algorithm": algorithm,
        "training_requested_episodes": training["requested_episodes"],
        "training_completed_episodes": training["episodes"],
        "training_budget_complete": bool(pilot_steps or training.get("episode_budget_complete", False)),
        "training_global_step": training["global_step"],
        "training_stop_reason": training.get("stop_reason"),
        "snapshots": full_manifest["count"], "snapshots_pruned": full_manifest.get("pruned"),
        "validation_candidates": manifest["count"],
        "validation_candidate_method": candidate_method,
        "validation_seed_count": manifest["validation_episodes"],
        "full_snapshot_archive_untouched": True,
        "gpu_val_status": "not_started",
        "gpu_test_status": "skipped_for_pilot" if pilot_steps else "not_started",
        "training_seed": 44, "wandb_run_id": run_id,
    }
    report = work / "one-session-train-eval-test-report.json"
    def save():
        report.write_text(json.dumps(status, indent=2))
    save()
    # 100 episodes * 81 checkpoints = 8100 physical Warp worlds per T4.
    # Avoid silently launching oversized batches on the CUDA evaluator.
    if max_eval_worlds_per_gpu and policies_per_batch*manifest["validation_episodes"] > max_eval_worlds_per_gpu:
        raise ValueError("GPU Evaluation allocation exceeds per-T4 Warp world cap")
    # No program deadline: evaluate every selected Actor, then full Final Test.
    args = [
        sys.executable, "-m", "uav_network.experiments.deferred_gpu_evaluation",
        "--folder", str(folder), "--output", str(work / "gpu_batch_eval"),
        "--gpus", "2", "--policies-per-batch", str(policies_per_batch),
        "--world-pool-size", str(eval_world_pool_size),
        "--max-eval-worlds-per-gpu", str(max_eval_worlds_per_gpu),
        "--max-test-worlds-per-gpu", str(max_test_worlds_per_gpu),
        "--wandb-run-id", run_id, "--hf-best",
        "--external-wandb-owner",
        "--final-test-batch-size", str(final_test_batch_size),
    ]
    if not pilot_steps:
        args.append("--final-test")
    status["gpu_val_status"] = "running"
    save()
    print("ONE_SESSION_GPU_EVAL_START", json.dumps({"run_id":run_id, "all_snapshots":full_manifest["count"], "shortlisted_for_validation":manifest["count"], "validation_seeds_each":manifest["validation_episodes"], "internal_deadline":"disabled"}), flush=True)
    # Training tracker has closed its SDK run, so resume EXACTLY this W&B ID
    # in the long-lived post-training supervisor. It owns the run until
    # Validation AND Final Test finish. GPU child workers NEVER init this run.
    import wandb
    from .one_session_wandb import (
        EvalConvergenceStreamer, log_final_test, previously_logged_steps)
    from .post_training_runtime_heartbeat import PostTrainingRuntimeHeartbeat
    wandb_run = wandb.init(
        entity=spec["entity"], project=spec["project"], id=run_id,
        resume="must", reinit="finish_previous", dir=str(work), job_type="fullflow_eval")
    fullflow_completed = False
    runtime_monitor = None
    process = None
    try:
        wandb_run.summary.update({
            "fullflow/status":"running",
            "fullflow/phase":"gpu_validation",
            "fullflow/complete":False,
            "fullflow/train_complete":True,
            "fullflow/validation_complete":False,
            "fullflow/test_complete":False,
            "fullflow/validation_required_checkpoints":manifest["count"],
            "fullflow/validation_episodes_per_checkpoint":manifest["validation_episodes"],
            "fullflow/validation_checkpoint_batch_size_per_gpu":int(policies_per_batch),
            "fullflow/max_warp_worlds_per_gpu":int(max_eval_worlds_per_gpu),
        })
        existing=previously_logged_steps(
            wandb.Api(timeout=25),
            f'{spec["entity"]}/{spec["project"]}/{run_id}')
        streamer=EvalConvergenceStreamer(
            wandb_run, manifest["snapshots"],work/"gpu_batch_eval",
            validation_seed_first=manifest["validation_seed_first"],
            validation_episodes=manifest["validation_episodes"],
            existing_steps=existing)
        runtime_monitor = PostTrainingRuntimeHeartbeat(
            entity=spec["entity"], project=spec["project"], run_id=run_id,
            output=work / "gpu_batch_eval", interval=60,
            completed_episodes=int(training["episodes"]),
            target_episodes=int(training["requested_episodes"]),
            owner_run=wandb_run).start()
        process=subprocess.Popen(args)
        while True:
            streamer.poll()
            code=process.poll()
            if code is not None:
                break
            time.sleep(3)
        streamer.poll()
        if code!=0:
            raise subprocess.CalledProcessError(code,args)
        if not streamer.finished:
            raise RuntimeError("GPU evaluation subprocess exited without every checkpoint result")
        result = json.loads((work / "gpu_batch_eval" / "report.json").read_text())
        if result.get("validation_completed") != manifest["count"] or (not pilot_steps and "final_test" not in result):
            raise RuntimeError("Partial GPU validation/test; refusing to mark flow complete")
        status.update({"gpu_val_status":"complete",
                       "best_global_step":result["best_global_step"],
                       "best_checkpoint":result["best_checkpoint"]})
        wandb_run.summary.update({
            "fullflow/validation_complete":True,
            "fullflow/phase":"final_test_complete" if not pilot_steps else "pilot_complete",
            "fullflow/validation_logged_checkpoints":streamer.index,
        })
        if not pilot_steps:
            final = result["final_test"]
            expected_test_episodes=int(
                spec.get("dense_eval_schedule",{}).get("final_test_scenarios",4000))
            if int(final["episodes"]) != expected_test_episodes:
                raise RuntimeError("Final Test incomplete: expected %d held-out seeds" %
                                   expected_test_episodes)
            # This is a held-out scalar summary, never appended to the
            # 100-episode/checkpoint training convergence line.
            log_final_test(wandb_run,final,best_global_step=result["best_global_step"])
            status.update({"gpu_test_status":"complete",
                           "gpu_test_worker_count":final.get("gpu_test_worker_count"),
                           "gpu_test_assignments":final.get("gpu_test_assignments"),
                           "test_episodes":final["episodes"],
                           "test_delivery_rate_percent":final["delivery_rate_percent"]})
        hf_state=result.get("publication",{}).get("hf_best","unknown")
        wandb_run.summary.update({
            "fullflow/status":"finished",
            "fullflow/phase":"fullflow_complete",
            "fullflow/complete":True,
            "fullflow/test_complete":not bool(pilot_steps),
            "training_completed_awaiting_best_actor":False,
            "checkpoint_selection.kind":"gpu_validation_best",
            "checkpoint_selection.global_step":int(result["best_global_step"]),
            "checkpoint_publication_status":f"gpu_validated_hf_{hf_state}",
        })
        save()
        fullflow_completed = True
        marker = "MATCHED_PILOT_VALIDATION_COMPLETE" if pilot_steps else "ONE_SESSION_TRAIN_EVAL_TEST_COMPLETE"
        print(marker, json.dumps(status), flush=True)
        return status
    except Exception as error:
        status["gpu_val_status"] = "failed_or_incomplete"
        if not pilot_steps:
            status["gpu_test_status"] = "failed_or_incomplete"
        status["error_type"] = type(error).__name__
        save()
        wandb_run.summary.update({
            "fullflow/status":"failed_or_incomplete",
            "fullflow/phase":"gpu_eval_or_final_test_failed",
            "fullflow/complete":False,
            "fullflow/error_type":type(error).__name__,
        })
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if runtime_monitor is not None:
            runtime_monitor.stop(failed=not fullflow_completed)
        # W&B global run STATE becomes finished only when all Train/Eval/Test
        # stages have completed. On failure it becomes failed/crashed instead.
        wandb_run.finish(exit_code=0 if fullflow_completed else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--policies-per-batch", type=int, default=2)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--eval-world-pool-size", type=int, default=0)
    parser.add_argument("--max-eval-worlds-per-gpu", type=int, default=0)
    parser.add_argument("--max-test-worlds-per-gpu", type=int, default=0)
    opt = parser.parse_args()
    if (not 1 <= opt.policies_per_batch <= 256
            or opt.test_batch_size < 0 or opt.eval_world_pool_size < 0
            or opt.max_eval_worlds_per_gpu < 0 or opt.max_test_worlds_per_gpu < 0
            or (opt.max_eval_worlds_per_gpu and opt.eval_world_pool_size)
            or (opt.test_batch_size == 0 and opt.max_test_worlds_per_gpu == 0)):
        parser.error("Invalid batch sizes / Warp pool size")
    run(opt.spec, policies_per_batch=opt.policies_per_batch,
        final_test_batch_size=opt.test_batch_size,
        eval_world_pool_size=opt.eval_world_pool_size,
        max_eval_worlds_per_gpu=opt.max_eval_worlds_per_gpu,
        max_test_worlds_per_gpu=opt.max_test_worlds_per_gpu)


if __name__ == "__main__":
    main()
