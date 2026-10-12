"""Kaggle-side validation/training; one independent learner per available GPU."""
import argparse
import json
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


def learner_command(spec, algorithm, run, gpu_index, destination):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index),
               WANDB_RUN_ID=run["id"], WANDB_NAME=run["name"])
    if spec.get("stage") == "train":
        env["UAV_BENCHMARK_FILE"] = str(Path(destination).parents[1] / f"{algorithm}-benchmark.json")
    args = [sys.executable, "-m", "uav_network", "train", "--config", spec["config"],
            "--algorithm", algorithm]
    overrides = [*spec["overrides"], "tracking.enabled=true", "runtime.device=cuda",
                 "runtime.replay_device=cuda", f"runtime.output_dir={destination}"]
    for override in overrides:
        args += ["--set", override]
    return args, env


def run_job(spec):
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle job requires usable CUDA; refusing a CPU fallback")
    output = Path("/kaggle/working")
    (output / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    with (output / "cuda-tests.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                        "-p", "test_cuda.py", "-v"], stdout=log, stderr=subprocess.STDOUT, check=True)
    if spec["stage"] == "train":
        config = load_config(spec["config"], spec["overrides"])
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
            # A single GPU runs algorithms sequentially; multiple GPUs run them independently.
            if index and index % count == 0:
                wait_all(processes[-count:])
            destination = output / spec["stage"] / algorithm
            args, env = learner_command(spec, algorithm, run, index % count, destination)
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
            continue
        folder = output / spec["stage"] / algorithm
        results[algorithm] = json.loads((folder / "summary.json").read_text())
    # Preserve completed training before any optional evaluation can fail.
    (output / "job-result.json").write_text(json.dumps(results, indent=2))
    if failures:
        (output / "job-failures.json").write_text(json.dumps(failures, indent=2))
    for algorithm in results:
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
    print(json.dumps({a: {k: v[k] for k in ("episodes", "global_step", "updates", "wandb_url")}
                      for a, v in results.items()}, indent=2), flush=True)
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
