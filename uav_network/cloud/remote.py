"""Kaggle-side validation/training; one independent learner per available GPU."""
import argparse
from copy import deepcopy
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index), PYTHONFAULTHANDLER="1",
               WANDB_RUN_ID=run["id"], WANDB_NAME=run["name"])
    env.pop("KAGGLE_API_TOKEN", None)
    env.pop("UAV_STAGED_RESUME_CHECKPOINT", None)
    if spec.get("stage") == "train":
        env["UAV_BENCHMARK_FILE"] = str(Path(destination).parents[1] / f"{algorithm}-benchmark.json")
    if algorithm in spec.get("warm_start_artifacts", {}):
        env["UAV_INPUT_ARTIFACT"] = spec.get("resolved_warm_start_artifacts", {}).get(
            algorithm, spec["warm_start_artifacts"][algorithm])
    args = [sys.executable, "-m", "uav_network", "train", "--config", spec["config"],
            "--algorithm", algorithm]
    overrides = [*spec["overrides"], "tracking.enabled=true", "runtime.device=cuda",
                 f"runtime.output_dir={destination}"]
    if spec.get("cpu_visualize"):
        overrides.append("tracking.evaluation_interval=0")
    if algorithm in spec.get("warm_start_paths", {}):
        overrides.append(f"runtime.warm_start_checkpoint={spec['warm_start_paths'][algorithm]}")
        env["UAV_STAGED_RESUME_CHECKPOINT"] = str(spec["warm_start_paths"][algorithm])
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
        from ..experiments.checkpoint_store import resolve_checkpoint_artifact
        artifact = resolve_checkpoint_artifact(api, ref)
        spec.setdefault("resolved_warm_start_artifacts", {})[algorithm] = getattr(artifact, "qualified_name", ref)
        pql = config is not None and config.pql.enabled
        full_resume = pql or spec.get("auto_continue")
        entry = "resume.pt" if full_resume else "checkpoint.pt"
        from ..experiments.checkpoint_store import download_checkpoint_file
        path = download_checkpoint_file(artifact, entry, output / "inputs" / algorithm)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("algorithm") != algorithm or "progress" not in saved:
            raise ValueError("Input artifact must match the algorithm and contain verified progress")
        if full_resume and (saved.get("version") != 3 or not isinstance(saved.get("training"), dict)):
            raise ValueError("GPU continuation input must contain full version-3 training state in resume.pt")
        if spec.get("auto_continue"):
            if config is None:
                raise ValueError("Automatic GPU continuation requires its verified training configuration")
            selected = deepcopy(config)
            selected.runtime.algorithm = algorithm
            if pql:
                from ..experiments.pql_checkpoints import validate_pql_checkpoint
                progress = validate_pql_checkpoint(saved, selected)
            else:
                from ..experiments.fixed_checkpoints import validate_fixed_checkpoint
                progress = validate_fixed_checkpoint(saved, selected)
            expected = spec.get("continuation_expected_episodes", {}).get(algorithm)
            if expected is not None and progress["episodes"] != expected:
                raise ValueError("Downloaded resume checkpoint differs from the expected episode prefix")
            expected_step = spec.get("continuation_expected_global_step", {}).get(algorithm)
            if expected_step is not None and progress["global_step"] != expected_step:
                raise ValueError("Downloaded resume checkpoint differs from expected global-step prefix")
            environment = saved["training"].get("environment")
            if spec.get("continuation_requires_environment_state") and environment is None:
                raise ValueError("Zero-episode continuation requires exact environment checkpoint")
            if environment is not None:
                from ..experiments.environment_checkpoints import validate_environment
                validate_environment(environment, selected, progress)
            spec.setdefault("continuation_expected_episodes", {})[algorithm] = progress["episodes"]
            spec.setdefault("continuation_expected_global_step", {})[algorithm] = progress["global_step"]
        paths[algorithm] = path
    return paths


def learner_time_limit(spec, config, gpu_count, index, elapsed):
    waves = math.ceil((len(spec["runs"]) - index) / gpu_count)
    reserve = (max(600, config.tracking.artifact_timeout_seconds + 120)
               if config.pql.enabled or spec.get("auto_continue") else 600)
    available = (min(spec.get("timeout_seconds", 39600), 39600) - elapsed - reserve) / waves
    if available <= 0:
        raise RuntimeError("No safe training time remains before the kernel ceiling")
    return min(config.runtime.max_train_seconds or available, available)


def gpu_allocation(config, count, index, gpus_per_run=1):
    if count < 1:
        raise ValueError("At least one CUDA GPU is required")
    # One PQL experiment owns all GPUs; BASE defaults to one GPU unless the job explicitly requests two.
    if config.pql.enabled:
        return 1, ",".join(str(i) for i in range(count))
    if gpus_per_run not in {1, 2} or gpus_per_run > count:
        raise ValueError("Requested GPUs per BASE run are unavailable")
    slots = count // gpus_per_run
    slot = index % slots
    start = slot * gpus_per_run
    return slots, ",".join(str(i) for i in range(start, start + gpus_per_run))


def base_benchmarks(spec, config, output):
    results = {}
    for algorithm in spec["runs"]:
        path = output / f"{algorithm}-benchmark.json"
        args = [sys.executable, "-m", "uav_network", "benchmark", "--config", spec["config"],
                "--env-counts", str(config.runtime.num_envs), "--steps", str(spec.get("benchmark_steps", 2)),
                "--learner-updates", str(spec.get("benchmark_learner_updates", 2)), "--output", str(path)]
        for override in [*spec["overrides"], f"runtime.algorithm={algorithm}",
                         "runtime.device=cuda"]:
            args += ["--set", override]
        subprocess.run(args, check=True)
        results[algorithm] = json.loads(path.read_text())
    return results


def pql_benchmarks(spec, config, output, require_pass=True):
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
                         "runtime.device=cuda"]:
            args += ["--set", override]
        benchmark_env = dict(os.environ)
        benchmark_env["UAV_STAGE"] = "benchmark"
        benchmark_env["UAV_TRAINING_MODE"] = "PQL"
        for name in ("WANDB_RUN_ID", "WANDB_NAME", "KAGGLE_API_TOKEN"):
            benchmark_env.pop(name, None)
        if spec.get("stage") == "benchmark":
            benchmark_env["WANDB_RUN_ID"] = spec["runs"][algorithm]["id"]
        with (output / f"{algorithm}-benchmark.log").open("w") as log:
            subprocess.run(args, env=benchmark_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(path.read_text())
        if require_pass and (not report.get("results") or any(row.get("status") != "passed"
                or not row.get("cuda_pipeline_verified") for row in report["results"])):
            raise RuntimeError(f"PQL benchmark failed for {algorithm}; inspect {path}; batch remains {batch}")
        results[algorithm] = report
    return results


def comparison_benchmarks(spec, config, output):
    rounds = max(40, config.pql.warmup_collection_steps + 8,
                 math.ceil(max(getattr(config, a).batch_size for a in spec["runs"]) /
                           config.runtime.num_envs) + 1)
    path = output / "comparison-benchmark.json"
    args = [sys.executable, "scripts/compare_training_gpu.py", "--config", spec["config"],
            "--algorithms", *spec["runs"], "--steps", str(rounds), "--output", str(path)]
    for override in [*spec["overrides"], "runtime.device=cuda"]:
        args += ["--set", override]
    env = dict(os.environ)
    for key in ("KAGGLE_API_TOKEN", "WANDB_RUN_ID", "WANDB_NAME"):
        env.pop(key, None)
    with (output / "comparison-benchmark.log").open("w") as log:
        completed = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not path.is_file():
        raise RuntimeError("GPU comparison produced no report; inspect comparison-benchmark.log")
    report = json.loads(path.read_text())
    report["runner_exit_code"] = getattr(completed, "returncode", None)
    if not report.get("results"):
        raise RuntimeError("GPU comparison produced no cases")
    expected = {(algorithm, case) for algorithm in spec["runs"]
                for case in ("legacy", "pql-single", "pql-dual")}
    actual = {(row.get("algorithm"), row.get("case")) for row in report["results"]}
    complete = actual == expected and len(report["results"]) == len(expected)
    recorded_statuses = {row.get("status") for row in report["results"]}
    expected_exit = 1 if "oom" in recorded_statuses else 0 if "passed" in recorded_statuses else 2
    infrastructure_ok = (report["runner_exit_code"] == expected_exit
                         and recorded_statuses <= {"passed", "oom", "unavailable"})
    report["comparison_completed"] = complete and infrastructure_ok
    if not report["comparison_completed"]:
        report["gpu_comparison_verified"] = False
    # Keep every requested OOM at the original batch; interrupted runs cannot claim completion.
    (output / "job-result.json").write_text(json.dumps(report, indent=2))
    if not report["comparison_completed"]:
        raise RuntimeError("GPU comparison incomplete or interrupted; partial results retained")
    return report



def cpu_replay_diagnostic(spec, output):
    diagnostic = spec.get("diagnostic")
    if diagnostic not in {"cpu-replay-1m", "gpu-replay-1m", "hybrid-replay-1m", "hybrid75-replay-1m", "hybrid75-4096-1m"} or set(spec.get("runs", {})) != {"masac"}:
        raise ValueError("Replay diagnostic requires exactly one supported MASAC 1M case")
    run = spec["runs"]["masac"]
    destination = Path(output) / "cpu-replay"
    env = dict(os.environ, WANDB_RUN_ID=run["id"], WANDB_NAME=run["name"],
               PROBE_COMMIT=spec["commit"])
    env.pop("KAGGLE_API_TOKEN", None)
    layout = {"cpu-replay-1m": "cpu", "gpu-replay-1m": "gpu",
              "hybrid-replay-1m": "hybrid50", "hybrid75-replay-1m": "hybrid75", "hybrid75-4096-1m": "hybrid75"}[diagnostic]
    args = [sys.executable, "-m", "uav_network.experiments.cpu_replay_probe",
            "--output", str(destination), "--layout", layout]
    if diagnostic == "hybrid75-4096-1m":
        args.append("--production-shape")
    subprocess.run(args, env=env, check=True)
    result_path = destination / "cpu-replay-result.json"
    if not result_path.is_file():
        raise RuntimeError("CPU replay diagnostic produced no result JSON")
    result = json.loads(result_path.read_text())
    (Path(output) / "job-result.json").write_text(json.dumps(result, indent=2))
    return result

def run_job(spec):
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle job requires usable CUDA; refusing a CPU fallback")
    output = Path("/kaggle/working")
    started = float(os.environ.get("UAV_JOB_STARTED_AT", time.monotonic()))
    (output / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    if spec.get("diagnostic") in {"cpu-replay-1m", "gpu-replay-1m", "hybrid-replay-1m", "hybrid75-replay-1m", "hybrid75-4096-1m"}:
        return cpu_replay_diagnostic(spec, output)
    config = load_config(spec["config"], spec["overrides"])
    spec["warm_start_paths"] = prepare_checkpoints(spec, output, config=config)
    checks = [("test_cuda.py", "cuda-tests.log"),
              ("test_cuda_scenario_v2.py", "scenario-v2-cuda-tests.log"),
              ("test_training_audit_regressions.py", "mission-regressions.log")]
    if config.pql.enabled:
        checks.append(("test_pql_training.py", "pql-smoke-tests.log"))
    for pattern, log_name in checks:
        with (output / log_name).open("w") as log:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                            "-p", pattern, "-v"], stdout=log, stderr=subprocess.STDOUT, check=True)
    if (config.runtime.amp_mode == "fp16" or config.runtime.compile_models
            or config.runtime.fused_optimizers):
        for algorithm in spec["runs"]:
            args = [sys.executable, "-m", "uav_network.cloud.execution_check",
                    "--config", spec["config"], "--algorithm", algorithm,
                    "--output", str(output / f"{algorithm}-execution-check.json")]
            for override in spec["overrides"]:
                args += ["--set", override]
            with (output / f"{algorithm}-execution-check.log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    if spec.get("compare_legacy"):
        if spec["stage"] != "benchmark" or not config.pql.enabled:
            raise ValueError("Matched comparison only supports the PQL benchmark stage")
        return comparison_benchmarks(spec, config, output)
    if spec["stage"] == "benchmark" and not config.pql.enabled:
        reports = base_benchmarks(spec, config, output)
        (output / "job-result.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps({algorithm: report["results"][0]["status"]
                          for algorithm, report in reports.items()}), flush=True)
        return reports
    if config.pql.enabled and spec["stage"] in {"train", "benchmark"}:
        reports = pql_benchmarks(spec, config, output, require_pass=spec["stage"] != "benchmark")
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
                             "runtime.device=cuda"]:
                args += ["--set", override]
            with (output / f"{algorithm}-benchmark.log").open("w") as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    count = torch.cuda.device_count()
    processes = []
    logs = []
    scheduling_error = None
    try:
        for index, (algorithm, run) in enumerate(spec["runs"].items()):
            slots, visible = gpu_allocation(config, count, index, spec.get("gpus_per_run", 1))
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
    except Exception as error:
        scheduling_error = error
        (output / "job-scheduling-failure.json").write_text(json.dumps(
            {"algorithm": algorithm, "error_type": type(error).__name__}, indent=2))
    finally:
        for process in processes:
            stop_process_tree(process)
        for log in logs:
            log.close()
    try:
        return finalize_runs(spec, processes, output)
    finally:
        if scheduling_error is not None:
            raise scheduling_error


def finalize_runs(spec, processes, output):
    results, failures, evaluation_failures = {}, {}, {}
    for algorithm, process in zip(spec["runs"], processes):
        if process.returncode != 0:
            failures[algorithm] = {"exit_code": process.returncode, "cause_confirmed": False,
                                   "log": str(output / f"{algorithm}-{spec['stage']}.log")}
            if isinstance(process.returncode, int) and process.returncode < 0:
                try:
                    failures[algorithm]["signal_name"] = signal.Signals(-process.returncode).name
                except ValueError:
                    failures[algorithm]["signal_name"] = None
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
    cpu_failure = continuation_failure = None
    if spec.get("auto_continue"):
        from .continuation import next_stage, submit_continuation, write_failure
        action = None
        try:
            if failures or (output / "job-scheduling-failure.json").is_file():
                raise ValueError("Automatic continuation cannot follow a learner or scheduling failure")
            action = next_stage(spec, results)
            if spec.get("orchestration") == "github_actions":
                result = next(iter(results.values()))
                (output / "continuation.json").write_text(json.dumps({
                    "status": "external_pending", "next_stage": action,
                    "root_kernel_ref": spec.get("root_kernel_ref", spec["kernel_ref"]),
                    "parent_kernel_ref": spec["kernel_ref"],
                    "requested_episodes": result["requested_episodes"],
                    "episodes": result["episodes"], "global_step": result["global_step"],
                    "episode_budget_complete": result["episode_budget_complete"],
                    "checkpoint_artifact": result["checkpoint_artifact"]}, indent=2))
            elif action == "gpu":
                submit_continuation(spec, results, output)
            else:
                from .visualize import submit_visualization
                child = submit_visualization(spec, results, output)
                (output / "continuation.json").write_text(json.dumps({
                    "status": "complete", "next_stage": "cpu",
                    "root_kernel_ref": spec.get("root_kernel_ref", spec["kernel_ref"]),
                    "parent_kernel_ref": spec["kernel_ref"], "cpu_kernel_ref": child["kernel_ref"],
                    "requested_episodes": next(iter(results.values()))["requested_episodes"],
                    "episodes": next(iter(results.values()))["episodes"]}, indent=2))
        except Exception as error:
            continuation_failure = {"error_type": type(error).__name__}
            write_failure(output, error)
            if action == "cpu":
                (output / "cpu-visualization-failure.json").write_text(json.dumps(continuation_failure, indent=2))
        from .pipeline_status import publish_pipeline_status
        publish_pipeline_status(spec, results, output, failed=continuation_failure is not None,
                                next_stage=action or "unknown")
    elif spec.get("cpu_visualize"):
        eligible = {a: result for a, result in results.items() if a not in failures}
        if eligible:
            from .visualize import submit_visualization
            try:
                submit_visualization(spec, eligible, output)
            except Exception as error:
                cpu_failure = {"error_type": type(error).__name__}
                (output / "cpu-visualization-failure.json").write_text(json.dumps(cpu_failure, indent=2))
    for algorithm in (() if spec.get("cpu_visualize") or spec.get("auto_continue") else results):
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
    if continuation_failure:
        raise RuntimeError(f"Automatic continuation failed ({continuation_failure['error_type']}); training results retained")
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
