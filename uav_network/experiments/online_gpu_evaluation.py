"""Opt-in online CUDA validation of immutable MATD3 Actors during Training.

The Kaggle notebook launches a separate worker pinned to physical GPU0.
It sees atomic actor-only .pt files as they appear, evaluates five fixed-seed
Warp worlds and saves each result atomically. Physical GPU1 now runs BOTH
Warp training collector and the SAC/TD3 learner; GPU0 is Eval-only. A STOP file tells the worker to finish its active evaluation and
exit, allowing the full two-GPU deferred evaluator to finish pending checkpoints
on the same Kaggle session.

Neither this worker nor the training logger creates a second W&B run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import torch

from .deferred_eval_snapshots import sha256
from .deferred_gpu_evaluation import evaluate_snapshot, result_from_records
from ..algorithms.checkpoints import validate_observation_version
from ..configuration import Config, _apply
from .gpu_mission_metrics import (PERIODIC_EVAL_AXIS, PERIODIC_EVAL_PREFIX,
                                  gpu_eval_wandb_payload, periodic_gpu_eval_wandb_payload)

SNAPSHOT_RE = re.compile(r"eval_step_[0-9]{12}_policy_[0-9]+(?:_[0-9]+)?\.pt$")


def ordered_snapshots(folder):
    """Order by immutable, zero-padded source global_step (never wall time)."""
    folder = Path(folder)
    return sorted((p for p in folder.glob("eval_step_*_policy_*.pt")
                   if SNAPSHOT_RE.fullmatch(p.name)), key=lambda p: p.name)


def logged_result_payload(result):
    """Live validation uses the same science metrics as post-train GPU Eval."""
    raw = gpu_eval_wandb_payload(result)
    payload = {**raw, "gpu_eval/online_during_training": 1}
    if result["source"].get("snapshot_reason") == "periodic":
        payload.update(periodic_gpu_eval_wandb_payload(raw))
    return payload



class LiveGPUResultLogger:
    """Called by the main training process; the GPU worker never touches W&B."""

    def __init__(self, folder, tracker, *, interval_seconds=4.0):
        self.folder = Path(folder)
        self.tracker = tracker
        self.interval_seconds = float(interval_seconds)
        self.last_poll = 0.0
        self.seen = set()
        self.failed = 0
        self.reported = 0
        self.defined = False
        self.defined_metric_keys = set()
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be nonnegative")

    def poll(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_poll < self.interval_seconds:
            return 0
        self.last_poll = now
        if not self.folder.is_dir():
            return 0
        completed = 0
        for path in sorted(self.folder.glob("eval_*.json")):
            if not re.fullmatch(r"eval_[0-9]{4}\.json", path.name) or path.name in self.seen:
                continue
            try:
                result = json.loads(path.read_text())
                if not result.get("records") or result.get("episodes") < 1:
                    raise ValueError("Missing real GPU evaluation episodes")
                payload = logged_result_payload(result)
                if self.tracker is not None:
                    # A no-delivery checkpoint correctly omits J/target.
                    # Register new metric names when they FIRST appear later;
                    # otherwise W&B would use the wrong x-axis for that series.
                    for axis in ("gpu_eval/checkpoint_global_step", PERIODIC_EVAL_AXIS):
                        if axis in payload and axis not in self.defined_metric_keys:
                            self.tracker.run.define_metric(axis)
                            self.defined_metric_keys.add(axis)
                    if "global_step" not in self.defined_metric_keys:
                        self.tracker.run.define_metric("global_step")
                        self.defined_metric_keys.add("global_step")
                    for key in payload:
                        if key not in self.defined_metric_keys:
                            if key == "global_step":
                                continue
                            self.tracker.run.define_metric(key, step_metric="global_step", step_sync=False)
                            self.defined_metric_keys.add(key)
                    self.defined = True
                    self.tracker.run.log(payload)
                self.seen.add(path.name)
                self.reported += 1
                completed += 1
            except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
                # Keep the result retryable, never fabricate an Eval observation.
                self.failed += 1
            except Exception:
                # A W&B outage must never fail healthy MATD3 training.
                self.failed += 1
        return completed

    def summary(self):
        return {"completed_and_logged": self.reported, "log_failures_or_retries": self.failed,
                "online_results_dir": str(self.folder)}


def evaluate_live_batch(entries, *, device):
    """Evaluate immutable Actor snapshots concurrently on the Eval-only GPU.

    A batch of K distinct shared policies controls K*5 independently seeded
    Warp worlds via torch.func.vmap, without creating any learner or replay.
    For a singleton or independent Actor, use the original parity-tested path.
    """
    if not entries:
        raise ValueError("No saved Actor to evaluate")
    pending = []
    for index, path, result_path in entries:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_observation_version(payload)
        pending.append((index, path, result_path, payload))
    config = Config()
    _apply(config, pending[0][3]["config"])
    config.validate()
    seed = config.tracking.periodic_evaluation_seed
    episodes = config.tracking.periodic_evaluation_episodes
    started = time.monotonic()
    if len(pending) > 1 and config.algorithm_config.share_actor_parameters:
        from .deferred_gpu_batched import evaluate_group
        by_policy = evaluate_group(config, [p[3] for p in pending],
                                   device=device, seed=seed, episodes=episodes)
        engine = "warp_vector_vmap_cuda_live" if torch.device(device).type == "cuda" else "warp_vector_vmap_cpu_live_smoke"
        results = [result_from_records(path, payload, records, device=device, engine=engine)
                   for (_, path, _, payload), records in zip(pending, by_policy)]
    else:
        results = [evaluate_snapshot(path, device=device, seed=seed, episodes=episodes)
                   for _, path, _, _ in pending]
    if len(results) != len(pending):
        raise RuntimeError("Live GPU Evaluation batch result count differs from Actor count")
    duration = time.monotonic() - started
    peak_mib = (torch.cuda.max_memory_allocated(device) / (1024**2)
                if torch.device(device).type == "cuda" else 0.0)
    for (index, path, output, payload), result in zip(pending, results):
        if len(pending) == 1:
            result["evaluation_backend"] = (
                "warp_vector_cuda_live" if torch.device(device).type == "cuda"
                else "warp_vector_cpu_live_smoke")
        if result["policy_provenance"]["checkpoint_sha256"] != sha256(path):
            raise RuntimeError("Actor changed while GPU0 evaluated its policy")
        result["gpu_eval_online_elapsed_seconds"] = duration
        result["gpu_eval_batch_policies"] = len(pending)
        result["gpu_eval_batch_time_s"] = duration
        result["gpu_eval_gpu_peak_allocated_mib"] = peak_mib
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
        temporary.replace(output)
        print("ONLINE_GPU_EVAL_COMPLETE", index, result["checkpoint_global_step"],
              "batch", len(pending), "seconds", round(duration, 2), flush=True)
    return len(results)


def run_worker(*, snapshots, results, device="cuda:0", poll_seconds=1.0,
               min_free_gpu_mib=1536, parent_pid=None, policies_per_batch=8,
               batch_wait_seconds=60.0):
    snapshots, results = Path(snapshots), Path(results)
    # Do not create output before learner startup: it requires empty output.
    if (poll_seconds <= 0 or min_free_gpu_mib < 0 or policies_per_batch < 1
            or batch_wait_seconds < 0):
        raise ValueError("Invalid live GPU Eval scheduling or memory limits")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Online GPU Evaluation requires CUDA; no CPU fallback")
    torch.set_num_threads(1)
    done_flag = snapshots / "LIVE_GPU_EVAL_STOP"
    results_count = 0
    wait_started = None
    while True:
        if parent_pid is not None and os.getppid() != parent_pid:
            print("ONLINE_GPU_EVAL_ORPHAN_STOP", flush=True)
            return results_count
        if done_flag.exists():
            print("ONLINE_GPU_EVAL_HANDOFF_TO_TWO_GPU", results_count, flush=True)
            break
        queue = ordered_snapshots(snapshots)
        pending = []
        for index, path in enumerate(queue):
            result_path = results / f"eval_{index:04d}.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous.get("policy_provenance", {}).get("checkpoint_sha256") != sha256(path):
                    raise RuntimeError("Existing online Eval record changed Actor identity")
                continue
            pending.append((index, path, result_path))
            if len(pending) >= policies_per_batch:
                break
        if not pending:
            wait_started = None
            time.sleep(poll_seconds)
            continue
        # Wait a bounded interval for more distinct checkpoint policies to
        # enable vmap grouping. Training proceeds uninterrupted on GPU1.
        if wait_started is None:
            wait_started = time.monotonic()
        if len(pending) < policies_per_batch and (time.monotonic() - wait_started) < batch_wait_seconds:
            time.sleep(poll_seconds)
            continue
        if torch.device(device).type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if free < min_free_gpu_mib * 1024 * 1024:
                print("ONLINE_GPU_EVAL_GPU0_MEMORY_GUARD", round(free/2**20),
                      "mib_free", flush=True)
                time.sleep(max(poll_seconds, 2.0))
                continue
        results.mkdir(parents=True, exist_ok=True)
        results_count += evaluate_live_batch(pending, device=device)
        wait_started = None
    return results_count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=1536)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--policies-per-batch", type=int, default=8)
    parser.add_argument("--batch-wait-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    total = run_worker(
        snapshots=args.snapshots, results=args.results, device=args.device,
        poll_seconds=args.poll_seconds, min_free_gpu_mib=args.min_free_gpu_mib,
        parent_pid=args.parent_pid, policies_per_batch=args.policies_per_batch,
        batch_wait_seconds=args.batch_wait_seconds)
    print("ONLINE_GPU_EVAL_FINISHED", total, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
