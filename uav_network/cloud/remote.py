"""Kaggle-side validation/training; one independent learner per available GPU."""
import argparse
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import torch

from ..experiments.hardware import hardware_info
from ..configuration import load_config


def wait_all(processes):
    while True:
        codes = [p.poll() for p in processes]
        if all(code is not None for code in codes):
            return codes
        time.sleep(2)


def stop_process_tree(process, grace_seconds=20):
    """Give the learner time to close workers/W&B, then kill its POSIX group."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(.1)
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def learner_command(spec, algorithm, run, gpu_index, destination, max_seconds=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index),
               WANDB_RUN_ID=run["id"], WANDB_NAME=run["name"])
    env.pop("KAGGLE_API_TOKEN", None)
    if spec.get("stage") == "train":
        env["UAV_BENCHMARK_FILE"] = str(Path(destination).parents[1] / f"{algorithm}-benchmark.json")
    if algorithm in spec.get("warm_start_artifacts", {}):
        env["UAV_INPUT_ARTIFACT"] = spec["warm_start_artifacts"][algorithm]
    args = [sys.executable, "-m", "uav_network", "train", "--config", spec["config"],
            "--algorithm", algorithm]
    overrides = [*spec["overrides"], "tracking.enabled=true", "runtime.device=cuda",
                 "runtime.replay_device=cuda", f"runtime.output_dir={destination}"]
    if spec.get("cpu_visualize"):
        overrides.append("tracking.evaluation_interval=0")
    if algorithm in spec.get("warm_start_paths", {}):
        overrides.append(f"runtime.warm_start_checkpoint={spec['warm_start_paths'][algorithm]}")
    if max_seconds is not None:
        overrides.append(f"runtime.max_train_seconds={max_seconds}")
    for override in overrides:
        args += ["--set", override]
    return args, env


def prepare_checkpoints(spec, output, config=None):
    if not spec.get("warm_start_artifacts"):
        return {}
    import wandb
    api = wandb.Api()
    paths = {}
    for algorithm, ref in spec["warm_start_artifacts"].items():
        artifact = api.artifact(ref, type="model")
        pql = config is not None and config.pql.enabled
        entry = "resume.pt" if pql else "checkpoint.pt"
        path = artifact.get_entry(entry).download(root=str(output / "inputs" / algorithm))
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("algorithm") != algorithm or "progress" not in saved:
            raise ValueError("Input artifact must match the algorithm and contain verified progress")
        if pql and (saved.get("version") != 3 or "training" not in saved):
            raise ValueError("PQL input artifact must contain full version-3 training state in resume.pt")
        paths[algorithm] = path
    return paths


def learner_time_limit(spec, config, gpu_count, index, elapsed):
    waves = math.ceil((len(spec["runs"]) - index) / gpu_count)
    reserve = max(600, config.tracking.artifact_timeout_seconds + 120) if config.pql.enabled else 600
    available = (min(spec.get("timeout_seconds", 39600), 39600) - elapsed - reserve) / waves
    if available <= 0:
        raise RuntimeError("No safe training time remains before the kernel ceiling")
    return min(config.runtime.max_train_seconds or available, available)


def gpu_allocation(config, count, index):
    if count < 1:
        raise ValueError("At least one CUDA GPU is required")
    # One PQL experiment owns all GPUs; independent algorithms run in waves.
    return (1, ",".join(str(i) for i in range(count))) if config.pql.enabled else (count, str(index % count))


def pql_benchmarks(spec, config, output):
    results = {}
    for algorithm in spec["runs"]:
        batch = getattr(config, algorithm).batch_size
        rounds = max(40, config.pql.warmup_collection_steps + 8,
                     math.ceil(batch / config.runtime.num_envs) + 1)
        path = output / f"{algorithm}-benchmark.json"
        args = [sys.executable, "scripts/benchmark_pql.py", "--config", spec["config"],
                "--algorithms", algorithm, "--envs", str(config.runtime.num_envs),
                "--batches", str(batch), "--steps", str(rounds), "--output", str(path)]
        for override in [*spec["overrides"], "tracking.enabled=true",
                         "runtime.device=cuda", "runtime.replay_device=cuda"]:
            args += ["--set", override]
        with (output / f"{algorithm}-benchmark.log").open("w") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(path.read_text())
        if (not report.get("results") or any(row.get("status") != "passed"
                or not row.get("cuda_pipeline_verified") for row in report["results"])):
            raise RuntimeError(f"PQL benchmark failed for {algorithm}; inspect {path}; batch remains {batch}")
        results[algorithm] = report
    return results


def run_job(spec):
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle job requires usable CUDA; refusing a CPU fallback")
    output = Path("/kaggle/working")
    started = float(os.environ.get("UAV_JOB_STARTED_AT", time.monotonic()))
    config = load_config(spec["config"], spec["overrides"])
    spec["warm_start_paths"] = prepare_checkpoints(spec, output, config=config)
    (output / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    checks = [("test_cuda.py", "cuda-tests.log"),
              ("test_training_audit_regressions.py", "mission-regressions.log")]
    if config.pql.enabled:
        checks.append(("test_pql_training.py", "pql-smoke-tests.log"))
    for pattern, log_name in checks:
        with (output / log_name).open("w") as log:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                            "-p", pattern, "-v"], stdout=log, stderr=subprocess.STDOUT, check=True)
    if spec["stage"] == "benchmark" and not config.pql.enabled:
        raise ValueError("benchmark cloud stage requires pql.enabled=true")
    if config.pql.enabled and spec["stage"] in {"train", "benchmark"}:
        reports = pql_benchmarks(spec, config, output)
        if spec["stage"] == "benchmark":
            (output / "job-result.json").write_text(json.dumps(reports, indent=2))
            print(json.dumps({algorithm: report["results"][0]["status"]
                              for algorithm, report in reports.items()}), flush=True)
            return reports
    elif spec["stage"] == "train":
        for algorithm in spec["runs"]:
            args = [sys.executable, "-m", "uav_network", "benchmark", "--config", spec["config"],
                    "--env-counts", str(config.runtime.num_envs), "--steps", "32",
                    "--learner-updates", "20", "--output", str(output / f"{algorithm}-benchmark.json")]
            for override in [*spec["overrides"], f"runtime.algorithm={algorithm}",
                             "runtime.device=cuda", "runtime.replay_device=cuda"]:
                args += ["--set", override]
            with (output / f"{algorithm}-benchmark.log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    count = torch.cuda.device_count()
    processes = []
    logs = []
    try:
        for index, (algorithm, run) in enumerate(spec["runs"].items()):
            slots, visible = gpu_allocation(config, count, index)
            if index and index % slots == 0:
                wait_all(processes[-slots:])
            destination = output / spec["stage"] / algorithm
            limit = learner_time_limit(spec, config, slots, index, time.monotonic() - started)
            args, env = learner_command(spec, algorithm, run, visible, destination, limit)
            log = (output / f"{algorithm}-{spec['stage']}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        wait_all(processes)
    finally:
        for process in processes:
            stop_process_tree(process)
        for log in logs:
            log.close()
    return finalize_runs(spec, processes, output)


def finalize_runs(spec, processes, output):
    results, failures, evaluation_failures = {}, {}, {}
    for algorithm, process in zip(spec["runs"], processes):
        if process.returncode != 0:
            failures[algorithm] = {"exit_code": process.returncode,
                                   "log": str(output / f"{algorithm}-{spec['stage']}.log")}
        folder = output / spec["stage"] / algorithm
        summary = folder / "summary.json"
        if summary.is_file():
            results[algorithm] = json.loads(summary.read_text())
        elif process.returncode == 0:
            failures[algorithm] = {"exit_code": 0, "reason": "missing_summary"}
    # Preserve completed training before any optional evaluation can fail.
    (output / "job-result.json").write_text(json.dumps(results, indent=2))
    if failures:
        (output / "job-failures.json").write_text(json.dumps(failures, indent=2))
    cpu_failure = None
    if spec.get("cpu_visualize"):
        eligible = {a: result for a, result in results.items() if a not in failures}
        if eligible:
            from .visualize import submit_visualization
            try:
                submit_visualization(spec, eligible, output)
            except Exception as error:
                cpu_failure = {"error_type": type(error).__name__}
                (output / "cpu-visualization-failure.json").write_text(json.dumps(cpu_failure, indent=2))
    for algorithm in (() if spec.get("cpu_visualize") else results):
        folder = output / spec["stage"] / algorithm
        with (output / f"{algorithm}-evaluation.log").open("w") as log:
            try:
                subprocess.run([sys.executable, "-m", "uav_network", "evaluate",
                                "--checkpoint", str(folder / "final.pt"), "--episodes", "1"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                evaluation_failures[algorithm] = {"exit_code": error.returncode,
                                                 "log": str(output / f"{algorithm}-evaluation.log")}
    if evaluation_failures:
        (output / "evaluation-failures.json").write_text(json.dumps(evaluation_failures, indent=2))
    print(json.dumps({a: {k: v.get(k) for k in ("episodes", "global_step", "updates", "wandb_url")}
                      for a, v in results.items()}, indent=2), flush=True)
    if cpu_failure:
        raise RuntimeError(f"CPU visualization submission failed; learner failures: {list(failures)}. Training results retained")
    if failures or evaluation_failures:
        raise RuntimeError(f"Learner failures: {list(failures)}; evaluation failures: {list(evaluation_failures)}. Results retained")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    run_job(json.loads(Path(args.spec).read_text()))


if __name__ == "__main__":
    main()
