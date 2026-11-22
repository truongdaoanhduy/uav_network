"""Standalone, no-training V9 Val100/Test4000 benchmark using 18d756ef-style batching.

Reads immutable Actor snapshots from HF dataset; never resumes/edits the source
W&B run, publishes its best checkpoint into the source model repo, or trains.
Only the benchmark's own W&B run and own HF dataset prefix may be written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
from uuid import uuid4

DATASET_REPO = "duy95/uav-search-diagnostics"
WANDB_ENTITY = "truongdaoanhduy-uit"
WANDB_PROJECT = "uav_search_target"
BASELINE_ID = "18d756ef"
GRID = 100 * 4096


def snapshot_hub_path(source_run_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{8}", source_run_id):
        raise ValueError("Source run ID must be exactly eight hexadecimal characters")
    return f"deferred-gpu-eval/{source_run_id}/deferred-gpu-eval.tar.gz"


def unique_benchmark_run_name(source_run_id: str, benchmark_id: str) -> str:
    snapshot_hub_path(source_run_id)
    if not re.fullmatch(r"[a-f0-9]{8}", benchmark_id):
        raise ValueError("Benchmark run ID must be exactly eight hexadecimal characters")
    if benchmark_id == source_run_id:
        raise ValueError("Benchmark W&B identity may never equal source training run")
    return f"MATD3-V9-VAL100-TEST4000-BASELINEBATCH-{source_run_id}-{benchmark_id}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_layout_summary(checkpoints: int, validation_episodes: int,
                         policies_per_batch: int, max_eval_worlds_per_gpu: int,
                         test_episodes: int, max_test_worlds_per_gpu: int) -> dict:
    from uav_network.experiments.gpu_eval_layout import plan_validation, plan_test
    if (not all(isinstance(x, int) and x > 0 for x in
                (checkpoints, validation_episodes, policies_per_batch,
                 max_eval_worlds_per_gpu, test_episodes, max_test_worlds_per_gpu))
            or policies_per_batch * validation_episodes > max_eval_worlds_per_gpu):
        raise ValueError("Invalid world capacity: policies_per_batch * Val100 must fit on GPU")
    layout = {"validation": {}, "test": {}, "validation_checkpoints": checkpoints}
    for gpu_index in (0, 1):
        batches = plan_validation(checkpoints, validation_episodes,
                                  policies_per_batch * validation_episodes,
                                  gpu_index=gpu_index, gpu_count=2)
        layout["validation"][f"gpu{gpu_index}"] = {
            "batches": [batch.worlds for batch in batches],
            "checkpoint_indices": [idx for b in batches for idx in b.checkpoint_indices],
            "worlds_max": max((batch.worlds for batch in batches), default=0),
        }
    shards = plan_test(test_episodes, 2, max_test_worlds_per_gpu)
    for gpu_index, ranges in enumerate(shards):
        layout["test"][f"gpu{gpu_index}"] = {
            "batches": [stop - start for start, stop in ranges],
            "seed_offsets": [list(pair) for pair in ranges],
        }
    all_idx = sorted(layout["validation"][f"gpu{gpu}"]["checkpoint_indices"]
                     [i] for gpu in (0, 1)
                     for i in range(len(layout["validation"][f"gpu{gpu}"]["checkpoint_indices"])))
    if all_idx != list(range(checkpoints)):
        raise AssertionError("Missing or duplicated checkpoint allocation")
    layout["validation"]["total_episode_rollouts"] = checkpoints * validation_episodes
    return layout


def extract_actor_snapshot_archive(archive: Path, destination: Path) -> Path:
    """Atomic safe extraction, no symlinks, preserves manifest and original bytes."""
    archive, destination = Path(archive), Path(destination)
    folder = destination / "deferred-gpu-eval"
    if folder.exists():
        from uav_network.experiments.deferred_gpu_evaluation import verify_manifest
        verify_manifest(folder)
        return folder
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Snapshot archive extraction destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / ".extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    seen = set()
    total_bytes = 0
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for item in tf:
                name = PurePosixPath(item.name)
                parts = name.parts
                if (name.is_absolute() or ".." in parts or not parts
                        or parts[0] != "deferred-gpu-eval" or len(parts) > 2
                        or item.issym() or item.islnk() or not (item.isfile() or item.isdir())):
                    raise ValueError("Unsafe actor snapshot archive entry")
                if item.isdir():
                    if len(parts) != 1:
                        raise ValueError("Unexpected archive directory")
                    continue
                if (len(parts) != 2 or parts[1] in seen
                        or not (parts[1] == "manifest.json" or parts[1].endswith(".pt")
                                or parts[1].endswith(".jsonl") or parts[1].endswith(".json"))):
                    raise ValueError("Unexpected actor snapshot archive file")
                if item.size < 0 or item.size > 250_000_000:
                    raise ValueError("Unexpected actor snapshot archive size")
                total_bytes += item.size
                if total_bytes > 12_000_000_000:
                    raise ValueError("Actor snapshot archive exceeds extraction budget")
                seen.add(parts[1])
                target = staging / parts[1]
                source = tf.extractfile(item)
                if source is None:
                    raise ValueError("Missing tar member data")
                with target.open("wb") as fd:
                    shutil.copyfileobj(source, fd)
        if "manifest.json" not in seen:
            raise ValueError("Snapshot archive is missing its manifest")
        manifest = json.loads((staging / "manifest.json").read_text())
        if (manifest.get("schema") != 1 or manifest.get("kind") != "deferred_gpu_eval_actor_snapshots"
                or manifest.get("count") != len(manifest.get("snapshots", []))
                or not manifest.get("keep_all_distinct_versions") or manifest.get("pruned") != 0
                or not manifest.get("snapshots")):
            raise ValueError("Incomplete or pruned Actor snapshot archive")
        for row in manifest["snapshots"]:
            basename = row["file"]
            if (PurePosixPath(basename).name != basename
                    or basename not in seen or sha256(staging / basename) != row["sha256"]):
                raise ValueError("Snapshot manifest SHA256/file mismatch")
        staging.replace(folder)
        return folder
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_source_snapshots(args, output: Path):
    from huggingface_hub import hf_hub_download
    from uav_network.experiments.deferred_gpu_evaluation import verify_manifest
    from uav_network.experiments.session_shortlist import create_shortlist
    source_path = snapshot_hub_path(args.source_run_id)
    archive = hf_hub_download(repo_id=args.hf_dataset_repo, repo_type="dataset",
                              filename=source_path, token=os.environ.get("HF_TOKEN"),
                              local_dir=output / "hf_cache")
    source = extract_actor_snapshot_archive(Path(archive), output / "source_snapshots")
    full = verify_manifest(source)
    if full["count"] != full["collected"] or full["pruned"] != 0:
        raise RuntimeError("Source snapshot archive incomplete or pruned")
    selection = output / "validation_selection"
    if selection.exists():
        selected = verify_manifest(selection)
        if selected.get("shortlist_provenance", {}).get("source_manifest_sha256") != sha256(source / "manifest.json"):
            raise RuntimeError("Existing shortlist comes from a different source manifest")
    else:
        selected = create_shortlist(source, selection, limit=512,
                                   step_targets=list(range(0, 80_000_001, GRID)))
    if selected["validation_episodes"] != args.validation_episodes:
        raise ValueError("Revalidation must use the original 100 validation seeds")
    return source, selection, selected, Path(archive)


def benchmark(args) -> dict:
    import torch
    import wandb
    from uav_network.experiments.one_session_wandb import (
        EvalConvergenceStreamer, log_final_test)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Two CUDA GPUs required for comparable 18d756ef-style benchmark")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source, folder, selected, archive = load_source_snapshots(args, output)
    layout = build_layout_summary(selected["count"], args.validation_episodes,
                                  args.policies_per_batch, args.max_eval_worlds_per_gpu,
                                  args.test_episodes, args.max_test_worlds_per_gpu)
    if selected.get("validation_seed_first") != 1000000044:
        raise ValueError("Source validation seed bank differs from the training experiment")
    print("ACTOR_ARCHIVE_SHA256", sha256(archive), flush=True)
    print("BENCHMARK_LAYOUT", json.dumps(layout, separators=(",", ":")), flush=True)
    status_file = output / "benchmark_status.json"
    if status_file.is_file():
        run_id = json.loads(status_file.read_text())["benchmark_run_id"]
    else:
        run_id = uuid4().hex[:8]
        while run_id == args.source_run_id:
            run_id = uuid4().hex[:8]
    name = unique_benchmark_run_name(args.source_run_id, run_id)
    status = {"kind": "independent_evaltest_benchmark", "source_run_id": args.source_run_id,
              "source_wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{args.source_run_id}",
              "benchmark_run_id": run_id, "name": name, "status": "starting", "layout": layout,
              "source_archive_sha256": sha256(archive),
              "source_manifest_sha256": sha256(source / "manifest.json"),
              "selected_manifest_sha256": sha256(folder / "manifest.json")}
    status_file.write_text(json.dumps(status, indent=2))
    print("BENCHMARK_WANDB_URL",f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",flush=True)
    gpu_output = output / "gpu_batch_eval"
    gpu_output.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["WANDB_ENTITY"] = WANDB_ENTITY
    env["WANDB_PROJECT"] = WANDB_PROJECT
    command = [sys.executable, "-m", "uav_network.experiments.deferred_gpu_evaluation",
               "--folder", str(folder), "--output", str(gpu_output),
               "--gpus", "2", "--policies-per-batch", str(args.policies_per_batch),
               "--world-pool-size", "0", "--max-eval-worlds-per-gpu", str(args.max_eval_worlds_per_gpu),
               "--max-test-worlds-per-gpu", str(args.max_test_worlds_per_gpu),
               "--final-test-batch-size", "0", "--final-test"]
    if args.dry_run:
        return {**status, "status": "dry_run", "command": command}

    begin = time.monotonic()
    eval_finished = None
    proc = None
    success = False
    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                    id=run_id, resume="allow", name=name, job_type="gpu_evaltest_benchmark",
                    group=f"V9-{args.source_run_id}-evaltest-comparison",
                    dir=str(output), config={
                        "benchmark/immutable_source_run": args.source_run_id,
                        "benchmark/source_archive_sha256": status["source_archive_sha256"],
                        "benchmark/source_manifest_sha256": status["source_manifest_sha256"],
                        "benchmark/validation_episodes": args.validation_episodes,
                        "benchmark/validation_checkpoints": selected["count"],
                        "benchmark/test_episodes": args.test_episodes,
                        "benchmark/policies_per_gpu_batch": args.policies_per_batch,
                        "benchmark/max_eval_worlds_per_gpu": args.max_eval_worlds_per_gpu,
                        "benchmark/max_test_worlds_per_gpu": args.max_test_worlds_per_gpu,
                        "benchmark/warp_pool_size": 0,
                    }) as run:
        streamer = EvalConvergenceStreamer(run, selected["snapshots"], gpu_output,
                                           validation_seed_first=selected["validation_seed_first"],
                                           validation_episodes=args.validation_episodes)
        run.summary.update({"benchmark/source_run_id": args.source_run_id,
                            "benchmark/source_run_untouched": True,
                            "benchmark/status": "running", "benchmark/complete": False})
        try:
            print("BENCHMARK_GPU_START", " ".join(command), flush=True)
            proc = subprocess.Popen(command, env=env)
            while proc.poll() is None:
                streamer.poll()
                if eval_finished is None and (gpu_output / "best.pt").is_file():
                    eval_finished = time.monotonic()
                    run.summary["benchmark/eval_wall_s"] = round(eval_finished - begin, 3)
                time.sleep(3)
            streamer.poll()
            if proc.returncode != 0:
                raise RuntimeError(f"GPU eval/test subprocess failed, return code {proc.returncode}")
            if not streamer.finished:
                raise RuntimeError("Not all validation checkpoint records were produced")
            report = json.loads((gpu_output / "report.json").read_text())
            final = json.loads((gpu_output / "final_test.json").read_text())
            if (report.get("validation_completed") != selected["count"]
                    or final["episodes"] != args.test_episodes
                    or report.get("best_global_step") is None):
                raise RuntimeError("Incomplete validation or held-out final test")
            log_final_test(run, report["final_test"], best_global_step=report["best_global_step"])
            stop = time.monotonic()
            if eval_finished is None:
                eval_finished = stop
            records = [json.loads((gpu_output / f"eval_{i:04d}.json").read_text())
                       for i in range(selected["count"])]
            validation_steps = sum(sum(int(r["episode_length"]) for r in x["records"]) for x in records)
            test_steps = sum(int(r["episode_length"]) for r in final["records"])
            score = {
                "validation_wall_s": eval_finished - begin,
                "test_wall_s": stop - eval_finished,
                "total_wall_s": stop - begin,
                "validation_episode_rollouts": selected["count"] * args.validation_episodes,
                "test_episode_rollouts": final["episodes"],
                "validation_simulated_world_steps": validation_steps,
                "test_simulated_world_steps": test_steps,
                "validation_simulated_steps_per_s": validation_steps / max(1e-6, eval_finished - begin),
                "test_simulated_steps_per_s": test_steps / max(1e-6, stop - eval_finished),
                "gpu_peak_allocated_mib_observed": max(float(x.get("gpu_eval_gpu_peak_allocated_mib", 0)) for x in records),
                "mean_validation_return": sum(x["mean_return"] for x in records) / len(records),
                "best_global_step": int(report["best_global_step"]),
                "best_actor_sha256": sha256(gpu_output / "best.pt"),
            }
            status.update({"status": "completed", "benchmark_metrics": score,
                           "final_mean_return": report["final_test"]["mean_return"],
                           "final_delivery_rate_percent": report["final_test"]["delivery_rate_percent"],
                           "final_discovery_rate_percent": report["final_test"]["search_rate_percent"]})
            status_file.write_text(json.dumps(status, indent=2))
            (output / "benchmark_summary.json").write_text(json.dumps(status, indent=2))
            run.summary.update({**{f"benchmark/{key}": value for key, value in score.items()},
                                "benchmark/status": "completed", "benchmark/complete": True})
            run.log({f"benchmark/{key}": value for key, value in score.items()})
            success = True
        except BaseException as error:
            status["status"] = "failed"
            status["error_type"] = type(error).__name__
            status["error_message"] = str(error)[:400]
            status_file.write_text(json.dumps(status, indent=2))
            run.summary.update({"benchmark/status": "failed", "benchmark/complete": False,
                                "benchmark/error_type": type(error).__name__})
            print("BENCHMARK_FAILED", type(error).__name__, str(error)[:400], flush=True)
            raise
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if not success:
                run.summary["benchmark/complete"] = False
    if args.hf_upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            prefix = f"evaltest-benchmarks/{args.source_run_id}/{run_id}"
            for relative in ("benchmark_summary.json", "gpu_batch_eval/report.json",
                             "gpu_batch_eval/validation_results.csv",
                             "gpu_batch_eval/validation_results.json",
                             "gpu_batch_eval/final_test.json", "gpu_batch_eval/best.json"):
                candidate = output / relative
                if candidate.is_file():
                    api.upload_file(path_or_fileobj=str(candidate), path_in_repo=f"{prefix}/{relative}",
                                    repo_id=args.hf_dataset_repo, repo_type="dataset",
                                    commit_message=f"Independent Eval100/Test4000 benchmark {run_id}")
            status["hf_benchmark_url"] = f"https://huggingface.co/datasets/{args.hf_dataset_repo}/tree/main/{prefix}"
            status_file.write_text(json.dumps(status, indent=2))
            print("BENCHMARK_HF_RESULTS", status["hf_benchmark_url"], flush=True)
        except Exception as error:
            status["hf_upload_error_type"] = type(error).__name__
            status_file.write_text(json.dumps(status, indent=2))
            print("BENCHMARK_HF_UPLOAD_WARNING", type(error).__name__, flush=True)
    print("BENCHMARK_COMPLETE", json.dumps(status["benchmark_metrics"]), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default="e079eace")
    parser.add_argument("--hf-dataset-repo", default=DATASET_REPO)
    parser.add_argument("--output", default="/kaggle/working/v9-baseline-batched-evaltest")
    parser.add_argument("--validation-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=4000)
    parser.add_argument("--policies-per-batch", type=int, default=81)
    parser.add_argument("--max-eval-worlds-per-gpu", type=int, default=8192)
    parser.add_argument("--max-test-worlds-per-gpu", type=int, default=4096)
    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    snapshot_hub_path(args.source_run_id)
    benchmark(args)


if __name__ == "__main__":
    main()
