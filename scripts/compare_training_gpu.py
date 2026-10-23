"""Matched fixed-scheduler versus asynchronous PQL comparison in fresh child processes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

# Direct script execution must not import another checkout from an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from uav_network.configuration import load_config
from uav_network.experiments.hardware import hardware_info
from uav_network.experiments.legacy_benchmark import LEGACY_REFERENCE_COMMIT, benchmark_legacy
from uav_network.experiments.pql_benchmark import benchmark_pql


CASES = ("legacy", "pql-single", "pql-dual")


class UnavailableCase(RuntimeError):
    """The requested physical GPU arrangement is absent, rather than a failed run."""


def case_config(config, case, algorithm, output, *, allow_cpu=False):
    c = deepcopy(config)
    if case not in CASES or algorithm not in ("masac", "matd3"):
        raise ValueError("Unknown comparison case or algorithm")
    c.runtime.algorithm = algorithm
    c.runtime.output_dir = str(output)
    c.tracking.enabled = False
    c.pql.enabled, c.pql.mode = case != "legacy", "async"
    ratio = float(c.pql.critic_sample_ratio)
    if not ratio.is_integer():
        raise ValueError("Matched legacy comparison requires an integer critic_sample_ratio")
    c.algorithm_config.learning_starts = c.pql.warmup_collection_steps * c.runtime.num_envs
    c.algorithm_config.updates_per_step = int(ratio)
    if c.runtime.warm_start_checkpoint != "none":
        raise ValueError("Comparison cases require fresh learners without warm-start checkpoints")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cpu = c.runtime.device == "cpu" or (c.runtime.device == "auto" and not available and allow_cpu)
    if case == "pql-dual" and (cpu or available < 2):
        raise UnavailableCase("pql-dual requires two usable CUDA devices; no single-GPU or CPU substitution")
    if cpu and not allow_cpu:
        raise UnavailableCase("CUDA measurement required; use --allow-cpu with runtime.device=cpu for a CPU smoke test")
    if not cpu and not available:
        raise UnavailableCase("CUDA unavailable; no GPU measurement performed")
    if case == "legacy" and (c.runtime.replay_cpu_fraction or
                             (not cpu and c.runtime.replay_device == "cpu")):
        raise ValueError("Legacy comparison requires its original same-device replay layout")
    c.runtime.device = "cpu" if cpu else "cuda"
    if cpu:
        c.runtime.replay_device = "cpu"
    collector = "cpu" if cpu else "cuda:0"
    learner = "cuda:1" if case == "pql-dual" else collector
    c.pql.collector_device, c.pql.learner_device, c.pql.policy_device = collector, learner, learner
    return c.validate()


def run_case(config, case, algorithm, steps, output, *, allow_cpu=False):
    """Write a JSON result even for unavailable hardware, validation failures or OOM."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    c = deepcopy(config)
    c.runtime.algorithm = algorithm
    c.runtime.output_dir = str(output.parent / output.stem)
    c.tracking.enabled = False
    report = {"status": "failed", "cuda_pipeline_verified": False, "target_t4_scenario_verified": False}
    try:
        c = case_config(c, case, algorithm, c.runtime.output_dir, allow_cpu=allow_cpu)
        report = (benchmark_legacy(c, steps) if case == "legacy" else benchmark_pql(c, steps))
    except Exception as error:
        cause = error
        while cause.__cause__ is not None:
            cause = cause.__cause__
        status = ("unavailable" if isinstance(cause, UnavailableCase) else
                  "oom" if isinstance(cause, torch.OutOfMemoryError) else "failed")
        report.update(status=status, error_type=type(cause).__name__, error=str(cause))
    report.update(case=case, algorithm=algorithm, process_id=os.getpid(),
                  source_git_commit=os.environ.get("UAV_GIT_COMMIT", "local"),
                  legacy_reference_commit=LEGACY_REFERENCE_COMMIT,
                  num_envs=c.runtime.num_envs, batch_size=c.algorithm_config.batch_size,
                  replay_capacity=c.algorithm_config.replay_capacity,
                  learning_starts=c.algorithm_config.learning_starts,
                  updates_per_step=c.algorithm_config.updates_per_step,
                  warmup_collection_steps=c.pql.warmup_collection_steps,
                  critic_sample_ratio=c.pql.critic_sample_ratio,
                  algorithm_settings=c.to_dict()[algorithm], runtime_settings=c.to_dict()["runtime"],
                  pql_settings=c.to_dict()["pql"], tracking_enabled=False, world=c.to_dict()["world"],
                  resolved_config_path=str(Path(c.runtime.output_dir) / "resolved_config.yaml"))
    folder = Path(c.runtime.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(c.to_dict(), sort_keys=False))
    for name in ("diagnostics.jsonl", "episodes.jsonl"):
        (folder / name).touch()
    # Benchmark functions record timing diagnostics; unavailable/error cases still get a record.
    if not (folder / "diagnostics.jsonl").stat().st_size:
        with (folder / "diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps({"status": report["status"], "case": case,
                                     "error_type": report.get("error_type")}, allow_nan=False) + "\n")
    (folder / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def comparison_report(results, source_commit):
    passed = [case for case in results if case["status"] == "passed"]
    failed = [case for case in results if case["status"] not in ("passed", "unavailable")]
    status = ("failed" if failed else "passed" if len(passed) == len(results) and results else
              "partial" if passed else "unavailable")
    verified = bool(results) and len(passed) == len(results) and all(
        case.get("cuda_pipeline_verified") for case in results)
    if verified:
        for algorithm in {case["algorithm"] for case in results}:
            group = [case for case in results if case["algorithm"] == algorithm]
            verified = verified and len(group) == 3 and len({
                (case["num_envs"], case["batch_size"], case["replay_capacity"],
                 case["learning_starts"], case["updates_per_step"],
                 case["critic_updates"], case["actor_updates"]) for case in group}) == 1
    return {"status": status, "gpu_comparison_verified": bool(verified),
            "source_git_commit": source_commit, "legacy_reference_commit": LEGACY_REFERENCE_COMMIT,
            "measurement_scope": "Fresh process per case; collection through final learner drain; startup excluded",
            "baseline_implementation": "Pre-PQL fixed scheduler with TensorReplayBuffer and the current combined "
                                       "update wrapper verified against the frozen legacy update oracle",
            "results": results}


def run_comparison(config_path, algorithms, steps, overrides, output, *, allow_cpu=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    source_commit = environment.get("UAV_GIT_COMMIT")
    if not source_commit:
        try:
            source_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                text=True, check=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            source_commit = "unknown"
    environment["UAV_GIT_COMMIT"] = source_commit
    results = []
    script = str(Path(__file__).resolve())
    for algorithm in algorithms:
        for case in CASES:
            target = output.parent / f"{algorithm}-{case}.json"
            command = [sys.executable, script, "--config", str(Path(config_path).resolve()),
                       "--case", case, "--algorithm", algorithm, "--steps", str(steps),
                       "--output", str(target)]
            for override in overrides:
                command.extend(("--set", override))
            if allow_cpu:
                command.append("--allow-cpu")
            # Delete only this runner's result file to prevent a crashed child from reusing stale JSON.
            target.unlink(missing_ok=True)
            process = subprocess.run(command, capture_output=True, text=True, env=environment)
            (output.parent / f"{algorithm}-{case}.stdout.log").write_text(process.stdout)
            (output.parent / f"{algorithm}-{case}.stderr.log").write_text(process.stderr)
            try:
                result = json.loads(target.read_text())
            except (OSError, ValueError):
                result = {"status": "failed", "case": case, "algorithm": algorithm,
                          "error_type": "ChildProcessError",
                          "error": "Child did not write a valid comparison result",
                          "stderr_tail": process.stderr[-3000:]}
            result["child_returncode"] = process.returncode
            if process.returncode and result["status"] == "passed":
                result.update(status="failed", error_type="ChildProcessError",
                              error=f"Child exited with status {process.returncode} after reporting success")
            results.append(result)
            report = comparison_report(results, source_commit)
            output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(result, allow_nan=False), flush=True)
    report = comparison_report(results, source_commit)
    report["hardware"] = hardware_info()
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion-pql.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=["masac", "matd3"], default=["masac", "matd3"])
    parser.add_argument("--case", choices=CASES, help="Internal child case; omitted to launch all three cases")
    parser.add_argument("--algorithm", choices=["masac", "matd3"], help="Algorithm for an internal child case")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--allow-cpu", action="store_true", help="Permit an explicit CPU smoke test; dual GPU stays unavailable")
    parser.add_argument("--output", default="outputs/gpu-comparison/report.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.case:
        if not args.algorithm:
            parser.error("--case requires --algorithm")
        try:
            config = load_config(args.config, args.overrides)
        except Exception as error:
            target = Path(args.output).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            result = {"status": "failed", "case": args.case, "algorithm": args.algorithm,
                      "process_id": os.getpid(), "error_type": type(error).__name__, "error": str(error)}
            target.write_text(json.dumps(result, indent=2, allow_nan=False))
        else:
            result = run_case(config, args.case, args.algorithm, args.steps, args.output, allow_cpu=args.allow_cpu)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0 if result["status"] == "passed" else 2 if result["status"] == "unavailable" else 1
    report = run_comparison(args.config, args.algorithms, args.steps, args.overrides, args.output,
                            allow_cpu=args.allow_cpu)
    return 1 if report["status"] == "failed" else 2 if report["status"] == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
