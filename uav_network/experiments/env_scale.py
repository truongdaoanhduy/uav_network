"""Measured native or device-env capacity sweep, stopped at its first failure.

Keep imports light: spawned simulator workers must not initialize CUDA.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read_number(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def cgroup_directories(proc, controller):
    """Resolve membership against actual mounts, including namespace mount roots."""
    memberships = [line.split(":", 2) for line in (proc / "self/cgroup").read_text().splitlines()]
    for line in (proc / "self/mountinfo").read_text().splitlines():
        fields = line.split()
        separator = fields.index("-")
        filesystem = fields[separator + 1]
        if filesystem not in ("cgroup", "cgroup2"):
            continue
        if filesystem == "cgroup" and controller not in fields[separator + 3].split(","):
            continue
        unescape = lambda value: value.replace(r"\040", " ").replace(r"\011", "\t").replace(r"\134", "\\")
        base, mount_root = Path(unescape(fields[4])), Path(unescape(fields[3]))
        for _, controllers, membership in memberships:
            if (filesystem == "cgroup2" and controllers) or (filesystem == "cgroup" and controller not in controllers.split(",")):
                continue
            group = Path(membership)
            try:
                relative = group.relative_to(mount_root)
                current = base / relative if ".." not in relative.parts else base
            except ValueError:
                current = base
            if not current.exists():
                current = base
            while True:
                yield current, filesystem
                if current == base:
                    break
                current = current.parent


def host_resources(proc=Path("/proc")):
    proc = Path(proc)
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        memory[name] = int(value.split()[0]) * 1024
    total = memory["MemTotal"]
    result = {"memory_limit_bytes": total,
              "memory_used_bytes": total - memory["MemAvailable"]}
    result["memory_pressure_fraction"] = result["memory_used_bytes"] / total
    for controller in ("memory", "pids"):
        for base, filesystem in cgroup_directories(proc, controller):
            limit, used = ("memory.max", "memory.current") if controller == "memory" else ("pids.max", "pids.current")
            if controller == "memory" and filesystem == "cgroup":
                limit, used = "memory.limit_in_bytes", "memory.usage_in_bytes"
            ceiling, current = read_number(base / limit), read_number(base / used)
            if ceiling and current is not None:
                pressure = controller + "_pressure_fraction"
                result[pressure] = max(result.get(pressure, 0), current / ceiling)
                limit_key = "memory_limit_bytes" if controller == "memory" else "pids_limit"
                used_key = "memory_used_bytes" if controller == "memory" else "pids_current"
                if ceiling < result.get(limit_key, float("inf")):
                    result.update({limit_key: ceiling, used_key: current, controller + "_limit_cgroup": str(base)})
    ticks = [int(x) for x in (proc / "stat").read_text().splitlines()[0].split()[1:9]]
    result["cpu_ticks"] = [sum(ticks), ticks[3] + ticks[4]]
    return result


def guard_reason(sample, memory_fraction):
    if (sample.get("memory_pressure_fraction", 0) >= memory_fraction or
        sample.get("memory_used_bytes", 0) >= memory_fraction * sample.get("memory_limit_bytes", float("inf"))):
        return "cpu_ram_guard"
    if sample.get("pids_pressure_fraction", 0) >= .9 or sample.get("pids_current", 0) >= .9 * sample.get("pids_limit", float("inf")):
        return "cpu_pid_guard"
    return None


def gpu_sample():
    try:
        text = subprocess.check_output(["nvidia-smi", "-i", "0",
            "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2)
        utilization, used, total = [float(x.strip()) for x in text.splitlines()[0].split(",")]
        return {"gpu_util_percent": utilization, "gpu_memory_used_mib": used, "gpu_memory_total_mib": total}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def summarize(samples, measurement=None):
    result = {"samples": len(samples)}
    for key in ("gpu_memory_used_mib", "memory_used_bytes", "pids_current"):
        values = [s[key] for s in samples if key in s]
        if values:
            result[key + "_peak"] = max(values)
    for phase in ("all", "measure"):
        for key in ("gpu_util_percent", "cpu_util_percent"):
            interval = key.split("_")[0] + "_interval_start_mono"
            selected = samples if phase == "all" else [s for s in samples if measurement and
                s.get(interval, float("-inf")) >= measurement[0] and
                s.get("sample_end_mono", float("inf")) <= measurement[1]]
            values = [s[key] for s in selected if key in s]
            result[key + "_samples_" + phase] = len(values)
            if values:
                result[key + "_mean_" + phase] = sum(values) / len(values)
                result[key + "_peak_" + phase] = max(values)
    counts = [result[key + "_samples_measure"] for key in ("gpu_util_percent", "cpu_util_percent")]
    result["measurement_status"] = "complete" if min(counts) >= 3 else "incomplete" if max(counts) else "unavailable"
    return result


def supervise(command, folder, timeout=600, memory_fraction=.85):
    from ..cloud.remote import stop_process_tree
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "case.json"
    started, samples, previous = time.monotonic(), [], None
    reason = None
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                current = json.loads(path.read_text()) if path.exists() else {"phase": "python_startup"}
                sample_started = time.monotonic()
                sample = {"elapsed_s": sample_started - started, "phase": current["phase"],
                          **host_resources(), **gpu_sample()}
                sample["sample_end_mono"] = time.monotonic()
                sample["gpu_interval_start_mono"] = sample_started - 1.
                ticks = sample.pop("cpu_ticks")
                if previous and ticks[0] > previous[0][0]:
                    sample["cpu_util_percent"] = 100 * (1 - (ticks[1] - previous[0][1]) / (ticks[0] - previous[0][0]))
                    sample["cpu_interval_start_mono"] = previous[1]
                previous = (ticks, sample_started)
                samples.append(sample)
                with (folder / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                if current.get("status") not in (None, "running", "ok"):
                    reason = current["status"]
                else:
                    reason = guard_reason(sample, memory_fraction)
                    if not reason and sample["elapsed_s"] >= timeout:
                        reason = "case_timeout"
                if reason:
                    break
                time.sleep(.25)
        finally:
            stop_process_tree(process, grace_seconds=2)
    result = json.loads(path.read_text()) if path.exists() else {"phase": "python_startup"}
    measurement = ((result["measure_start_mono"], result["measure_end_mono"])
                   if "measure_end_mono" in result else None)
    result.update(exit_code=process.returncode, wall_seconds=time.monotonic() - started,
                  resources=summarize(samples, measurement))
    result["status"] = reason or result.get("status", "worker_error")
    if result["status"] == "running" or (process.returncode != 0 and result["status"] == "ok"):
        result["status"] = "worker_error"
    if result["status"] == "ok" and result.get("device") == "cuda" and result["resources"]["measurement_status"] != "complete":
        result["status"] = "measurement_unavailable"
    write_json(path, result)
    return result


def sweep(counts, algorithms, output, runner):
    if not counts or counts != sorted(set(counts)) or any(n < 1 for n in counts):
        raise ValueError("Counts must be positive, unique and increasing")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory for a new sweep")
    output.mkdir(parents=True, exist_ok=True)
    report = {"requested_counts": counts, "algorithms": algorithms, "results": [],
              "stop_reason": None, "skipped_counts": []}
    for count in counts:
        for algorithm in algorithms:
            folder = output / f"{count}-{algorithm}"
            folder.mkdir()
            report["current_case"] = {"num_envs": count, "algorithm": algorithm}
            write_json(output / "results.json", report)
            result = runner(count, algorithm, folder)
            result.update(num_envs=count, algorithm=algorithm)
            report["results"].append(result)
            report["current_case"] = None
            if result["status"] != "ok":
                report["stop_reason"] = result["status"]
                report["skipped_counts"] = [n for n in counts if n > count]
            write_json(output / "results.json", report)
            print(json.dumps({k: result.get(k) for k in ("num_envs", "algorithm", "status", "phase", "resources")}), flush=True)
            if report["stop_reason"]:
                return report
    return report


def failure_status(error, device):
    """Warp allocator failure lacks a CUDA error code; do not invent an OOM."""
    import re
    import torch
    if device == "cuda" and isinstance(error, torch.cuda.OutOfMemoryError):
        return "cuda_oom"
    if (device == "cuda" and isinstance(error, RuntimeError) and
        re.fullmatch(r"Failed to allocate [0-9]+ bytes on device 'cuda(:[0-9]+)?'", str(error))):
        return "cuda_allocation_failure"
    return "cpu_oom" if isinstance(error, MemoryError) else "worker_error"


def probe(config, output, steps=16, min_seconds=0):
    import numpy as np
    import torch
    from ..algorithms.checkpoints import make_learner
    from ..algorithms.replay import TensorReplayBuffer
    from ..environment.observations import observation_dim, state_dim
    from ..environment.vector import make_vector, random_actions
    from .hardware import hardware_info, synchronize
    from .training import seed_runtime, choose_device
    config.validate()
    seed_runtime(config)
    device = choose_device(config.runtime.device)
    count, settings = config.runtime.num_envs, config.algorithm_config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "phase": "gpu_allocation", "num_envs": count,
              "algorithm": config.runtime.algorithm, "device": device, "hardware": hardware_info(),
              "network_backend": "uavnetsim", "config": config.to_dict()}
    device_env = config.runtime.environment_backend == "warp_packet"
    result["environment_backend"] = config.runtime.environment_backend
    if device_env:
        result["network_backend"] = "warp_packet"
    vector = None
    started = time.perf_counter()

    def publish(phase):
        result["phase"] = phase
        if device == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / "case.json", result)

    try:
        if not config.network.enabled:
            raise ValueError("This probe requires packet communication to be enabled")
        publish("gpu_allocation")
        learner = make_learner(config, observation_dim(config), state_dim(config), device)
        replay = TensorReplayBuffer(settings.replay_capacity, (config.world.num_uavs, observation_dim(config)),
                                    state_dim(config), config.runtime.seed, device)
        result["replay_bytes"] = replay.allocated_bytes
        publish("vector_startup")
        vector = make_vector(config)
        obs, infos = vector.reset(seed=config.runtime.seed)
        result["native_worker_pids"] = [int(x) for x in infos.get("worker_pid",[])]
        result["num_cpu_env_workers"] = vector.num_cpu_workers if device_env else len(result["native_worker_pids"])
        if device_env:
            result["gpu_environment_bytes"] = vector.allocated_bytes if device == "cuda" else 0
            result["environment_device"] = str(vector.device)
        result["startup_s"] = time.perf_counter() - started
        rng = np.random.default_rng(config.runtime.seed + 1)
        global_step, next_seed = 0, config.runtime.seed + count

        def collect(action, update=False):
            nonlocal obs, global_step, next_seed
            actions = {"continuous": action.continuous, "destination": action.destination}
            overlap = count > 1 and config.runtime.collector_overlap
            if overlap:
                vector.step_async(actions)
            if update:
                for _ in range(settings.updates_per_step):
                    learner.update(replay.sample(settings.batch_size), collect_metrics=False)
            final, reward, terminated, truncated, infos = vector.step_wait() if overlap else vector.step(actions)
            replay.add_batch(obs["observation"], obs["state"], action, reward,
                             final["observation"], final["state"], terminated)
            global_step += count
            obs = final
            done = terminated | truncated
            indices = list(infos["device_episode_infos"]) if device_env else np.flatnonzero(done).tolist()
            if indices:
                seeds = [None] * count
                for i in indices:
                    seeds[i], next_seed = next_seed, next_seed + 1
                mask = done.cpu().numpy() if device_env else done
                obs, _ = vector.reset(seed=seeds, options={"reset_mask": mask})

        publish("replay_warmup")
        while global_step < max(settings.learning_starts, settings.batch_size):
            collect(vector.random_actions(obs["observation"]) if device_env else
                    random_actions(obs["observation"],rng,config.world.num_uavs+1))
        publish("learner_warmup")
        for _ in range(2):
            learner.update(replay.sample(settings.batch_size), collect_metrics=False)
        synchronize(device)
        result["warmup_global_step"] = global_step
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result["measure_start_mono"] = time.monotonic()
        publish("measure")
        begin, inference_s, initial_updates, measured_steps = time.perf_counter(), 0., learner.updates, 0
        while measured_steps < steps or time.perf_counter() - begin < min_seconds:
            inference_started = time.perf_counter()
            action = learner.act_batch(obs["observation"],step=global_step,return_tensors=device_env)
            inference_s += time.perf_counter() - inference_started
            collect(action, update=True)
            measured_steps += 1
        synchronize(device)
        result["measure_end_mono"] = time.monotonic()
        elapsed = time.perf_counter() - begin
        result.update(status="ok", global_step=global_step, measured_vector_steps=measured_steps,
                      measured_joint_transitions=count * measured_steps, measured_seconds=elapsed,
                      joint_transitions_per_s=count * measured_steps / elapsed, inference_seconds=inference_s,
                      total_updates=learner.updates, measured_updates=learner.updates - initial_updates,
                      sample_uses_per_new_transition=settings.updates_per_step * settings.batch_size / count)
        publish("done")
    except Exception as error:
        import traceback
        traceback.print_exc()
        result.update(error_type=type(error).__name__, error_message=str(error), failed_phase=result["phase"])
        result["status"] = failure_status(error, device)
        publish("failed")
    finally:
        if vector is not None:
            vector.close()
    return result


def run_case(count, algorithm, folder, args):
    command = [sys.executable, "-m", "uav_network.experiments.env_scale", "--case", "--config", args.config,
               "--counts", str(count), "--algorithms", algorithm, "--steps", str(args.steps),
               "--min-seconds", str(args.min_seconds),
               "--device", args.device, "--backend",args.backend,"--output", str(folder)]
    return supervise(command, folder, args.case_timeout, args.memory_fraction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion-long.yaml")
    parser.add_argument("--counts", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--algorithms", nargs="+", choices=["masac", "matd3"], default=["masac", "matd3"])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--min-seconds", type=float, default=10, help="Minimum timed policy/update interval")
    parser.add_argument("--case-timeout", type=float, default=600)
    parser.add_argument("--memory-fraction", type=float, default=.85)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--backend",choices=["native","warp_packet"],default="native")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.min_seconds < 0 or args.case_timeout <= 0 or not 0 < args.memory_fraction < 1:
        parser.error("Positive steps/timeout and memory fraction in (0,1) required")
    if args.case:
        from ..configuration import load_config
        config = load_config(args.config, [f"runtime.num_envs={args.counts[0]}",
            f"runtime.algorithm={args.algorithms[0]}", f"runtime.device={args.device}",
            f"runtime.replay_device={args.device}",f"runtime.environment_backend={args.backend}", "tracking.enabled=false"])
        result = probe(config, args.output, args.steps, args.min_seconds)
        raise SystemExit(0 if result["status"] == "ok" else 1)
    sweep(args.counts, args.algorithms, args.output, partial(run_case, args=args))


if __name__ == "__main__":
    main()
