"""Bounded dual-T4 MASAC BASE memory diagnostic; never a production training entrypoint.

Run from the repository root. Only this process temporarily replaces MASAC's
optimizer helpers and compiles its outer vectorized critic callable. Scientific
loss/target/action equations are always the original MASAC methods.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch


def replay_bytes(capacity, observation_shape, state_dim):
    """Exact original TensorReplayBuffer FP32/int64 payload, without allocating."""
    if capacity < 1 or state_dim < 1 or any(dim < 1 for dim in observation_shape):
        raise ValueError("Replay dimensions and capacity must be positive")
    n, obs = observation_shape
    return capacity * (4 * (2 * n * obs + 2 * state_dim + 3 * n + 2) + 8 * n)


def require_two_t4():
    """Fail before models, worlds, replay, or any large CPU fallback allocation."""
    message = "This diagnostic requires exactly two T4 CUDA devices; no CPU fallback"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(message)
    devices = [torch.cuda.get_device_properties(index) for index in range(2)]
    if any("t4" not in device.name.lower() for device in devices):
        raise RuntimeError(message)
    return [{"index": index, "name": device.name, "total_bytes": device.total_memory,
             "compute_capability": [device.major, device.minor]}
            for index, device in enumerate(devices)]


class ProbeAMP:
    """Single-threaded, process-local adapter with independent scaler clocks.

    Actor gradients are all unscaled and checked before any actor Adam step.
    An overflow in one actor skips every actor, including its Adam state update.
    CPU device selection exists only for small optimizer arithmetic unit tests.
    """
    def __init__(self, learner, device_type="cuda", init_scale=65536.0):
        self.learner = learner
        self.device_type = device_type
        self.scalers = {name: torch.amp.GradScaler(device_type, init_scale=init_scale)
                        for name in ("critic", "actor", "alpha")}
        self.counts = {name: {"attempted": 0, "applied": 0, "skipped": 0}
                       for name in self.scalers}
        self.original = None

    def _step(self, group, losses, optimizers, parameter_groups, grad_clip, check_finite=True):
        with torch.autocast(self.device_type, enabled=False):
            if check_finite and not torch.isfinite(torch.stack(losses)).all():
                raise FloatingPointError("Non-finite learner loss")
            scaler = self.scalers[group]
            count = self.counts[group]
            count["attempted"] += 1
            parameters = [list(values) for values in parameter_groups]
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward([scaler.scale(loss) for loss in losses])
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            # Norms are explicitly measured on unscaled gradients, before clipping.
            norms = [torch.nn.utils.clip_grad_norm_(values, grad_clip,
                                                   error_if_nonfinite=False).detach()
                     for values in parameters]
            finite = bool(torch.isfinite(torch.stack(norms)).all())
            if finite:
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                count["applied"] += 1
            else:
                # Do not call any optimizer step. Explicit backoff also covers a
                # nonfinite aggregate norm from individually finite large gradients.
                scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
                count["skipped"] += 1
            return norms

    def single_step(self, loss, optimizer, parameters, grad_clip, check_finite=True):
        group = "critic" if optimizer is self.learner.critic_optimizer else "alpha"
        return self._step(group, [loss], [optimizer], [parameters], grad_clip, check_finite)[0]

    def actor_steps(self, losses, optimizers, parameter_groups, grad_clip, check_finite=True):
        return self._step("actor", losses, optimizers, parameter_groups, grad_clip, check_finite)

    def __enter__(self):
        import uav_network.algorithms.masac as module
        self.original = (module.step_optimizer, module.step_actor_optimizers)
        module.step_optimizer, module.step_actor_optimizers = self.single_step, self.actor_steps
        return self

    def __exit__(self, *exc):
        import uav_network.algorithms.masac as module
        module.step_optimizer, module.step_actor_optimizers = self.original

    def evidence(self):
        return {"counts": self.counts,
                "scales": {name: scaler.get_scale() for name, scaler in self.scalers.items()},
                "unscale_before_clip": True, "actor_all_or_none": True,
                "scope": "standalone probe only; original MASAC equations"}


class ProbeCompiler:
    """Compile the outer vmap function, never an inner critic called by vmap."""
    def __init__(self, learner, evidence, enabled=True):
        self.learner, self.evidence, self.enabled = learner, evidence, enabled
        self.restores = []

    def _observe(self, name, function):
        record = self.evidence.setdefault(name, {"completed_calls": 0, "fp16_outputs": 0,
                                               "output_dtypes": []})

        def observed(*args, **kwargs):
            output = function(*args, **kwargs)
            tensors = output if isinstance(output, tuple) else (output,)
            record["completed_calls"] += 1
            record["fp16_outputs"] += sum(value.dtype == torch.float16 for value in tensors)
            record["output_dtypes"] = sorted(set(record["output_dtypes"]) |
                                             {str(value.dtype) for value in tensors})
            return output
        return observed

    def __enter__(self):
        import uav_network.algorithms.masac as module
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        self.before = {key: dict(value) for key, value in counters.items()}
        self.kernels_before = metrics.generated_kernel_count
        self.evidence["requested"] = self.enabled
        try:
            original = module.critic_values
            self.restores.append((module, "critic_values", original))
            function = torch.compile(original, dynamic=True, backend="inductor") if self.enabled else original
            module.critic_values = self._observe("critic", function)
            for actor in self.learner.actors:
                original = actor.encoder.forward
                self.restores.append((actor.encoder, "forward", original))
                function = torch.compile(original, dynamic=True, backend="inductor") if self.enabled else original
                actor.encoder.forward = self._observe("actors", function)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.evidence["dynamo"] = {
            group: {str(key): value - self.before.get(group, {}).get(key, 0)
                    for key, value in values.items()
                    if value != self.before.get(group, {}).get(key, 0)}
            for group, values in counters.items()}
        self.evidence["inductor_generated_kernels"] = metrics.generated_kernel_count - self.kernels_before


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--capacity", type=int, choices=[1_000_000, 2_000_000], required=True)
    result.add_argument("--config", default="configs/comparison-20k-steps1800-scenario-v3.yaml")
    result.add_argument("--output", default="outputs/masac-memory-probe")
    result.add_argument("--envs", type=int, choices=[2048], default=2048)
    result.add_argument("--batch", type=int, choices=[4096], default=4096)
    result.add_argument("--rounds", type=int, default=4)
    result.add_argument("--updates-per-round", type=int, default=8)
    result.add_argument("--amp", choices=["fp16"], default="fp16")
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--wandb-project", default="uav_search_target")
    result.add_argument("--wandb-entity", default=None)
    result.add_argument("--wandb-name", default=None)
    result.add_argument("--wandb-run-id", default=None)
    return result


def run_probe(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = sampler = vector = learner = replay = amp = None
    phase = "hardware_guard"
    report = {"schema_version": 1, "status": "running", "phase": phase,
              "scope": "bounded allocation and real-update diagnostic; not full training or convergence",
              "requested": vars(args), "stages": [], "updates": [], "collection_steps": 0,
              "devices": {"worlds": "cuda:0", "collector": "cuda:0",
                          "learner": "cuda:1", "replay": "cuda:1"},
              "source_git_commit": os.environ.get("UAV_GIT_COMMIT"),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "compiler": {}, "full_training_verified": False}

    def persist():
        report["wall_seconds"] = time.perf_counter() - started
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False))
        temporary.replace(output / "result.json")

    def snapshot(name, synchronize=True):
        nonlocal phase
        phase = report["phase"] = name
        memory = []
        if report.get("hardware"):
            for index in range(2):
                if synchronize:
                    torch.cuda.synchronize(index)
                memory.append({"device": index, "allocated_bytes": torch.cuda.memory_allocated(index),
                               "reserved_bytes": torch.cuda.memory_reserved(index),
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)})
        stage = {"phase": name, "wall_seconds": time.perf_counter() - started, "torch": memory}
        if sampler:
            sampler._sample()
            stage["physical_samples"] = sampler.samples[-2:]
        report["stages"].append(stage)
        persist()
        if run:
            run.log({"probe/phase": name, "probe/wall_seconds": stage["wall_seconds"],
                     **{f"memory/gpu{row['device']}/{key}": value
                        for row in memory for key, value in row.items() if key != "device"}})

    try:
        report["hardware"] = require_two_t4()
        torch.cuda.init()
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        from uav_network.experiments.gpu_resources import GPUResourceSampler
        sampler = GPUResourceSampler()
        sampler.start()
        snapshot("tracking_init")
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise RuntimeError("Online W&B requires WANDB_API_KEY before any large allocations")
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name or f"MASAC-memory-E2048-B4096-R{args.capacity}",
                         id=args.wandb_run_id, resume="never", mode="online", dir=str(output),
                         job_type="memory-probe", tags=["diagnostic", "masac", "dual-t4", "fp16"],
                         config={"probe": vars(args)})
        if run is None or run.settings.mode != "online":
            raise RuntimeError("W&B online run was not initialized")
        report["wandb"] = {"id": run.id, "url": run.url, "mode": run.settings.mode}
        from uav_network.configuration import load_config
        from uav_network.algorithms.masac import MASAC
        from uav_network.algorithms.replay import TensorReplayBuffer
        from uav_network.environment.observations import observation_dim, state_dim
        from uav_network.environment.vector import make_vector
        from uav_network.experiments.pql_workers import inference_copy
        from uav_network.experiments.training import seed_runtime, refresh_base_collector
        config = load_config(args.config)
        original_starts = config.masac.learning_starts
        config.runtime.algorithm, config.pql.enabled = "masac", False
        config.runtime.environment_backend = "warp_packet"
        config.runtime.device = config.runtime.replay_device = "cuda"
        config.runtime.num_envs = args.envs
        config.runtime.compile_models, config.runtime.fused_optimizers = args.compile, args.fused
        config.runtime.vectorize_critics = True
        config.world.max_steps = 1800
        config.masac.batch_size, config.masac.replay_capacity = args.batch, args.capacity
        config.masac.learning_starts, config.masac.updates_per_step = 2 * args.envs, args.updates_per_round
        config.tracking.enabled = False  # Only the explicit online diagnostic run; no checkpoint machinery.
        config.validate()
        if not 1 <= args.rounds <= 16 or not 1 <= args.updates_per_round <= 32:
            raise ValueError("Bounded probe requires 1..16 rounds and 1..32 updates per round")
        report["warmup_override"] = {"diagnostic_only": True, "original_learning_starts": original_starts,
                                     "diagnostic_learning_starts": 2 * args.envs,
                                     "warmup_vector_steps": 2}
        report["replay_expected_bytes"] = replay_bytes(
            args.capacity, (config.world.num_uavs, observation_dim(config)), state_dim(config))
        report["resolved_config"] = config.to_dict()
        (output / "resolved_config.json").write_text(json.dumps(report["resolved_config"], indent=2))
        run.config.update({"resolved": report["resolved_config"],
                           "warmup_override": report["warmup_override"]})
        if report["source_git_commit"] is None:
            report["source_git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip()
        seed_runtime(config)
        snapshot("world_allocation")
        vector = make_vector(config, device="cuda:0")
        snapshot("world_reset")
        obs, _ = vector.reset(seed=config.runtime.seed)
        report["environment"] = {"allocated_bytes": vector.allocated_bytes,
                                 "device": str(vector.device), "backend": vector.environment_backend,
                                 "num_cpu_workers": vector.num_cpu_workers,
                                 "reset_observation_device": str(obs["observation"].device)}
        snapshot("learner_allocation")
        learner = MASAC(config, observation_dim(config), state_dim(config), device="cuda:1")
        snapshot("collector_allocation")
        collector = inference_copy(learner, "cuda:0")
        snapshot("collector_forward")
        collector.act_batch(obs["observation"], return_tensors=True)
        report["collector_forward"] = {"executed": True, "device": str(collector.device),
                                        "batch": args.envs}
        report["parameters"] = {"dtypes": sorted({str(p.dtype) for p in learner.parameters()}),
                                "devices": sorted({str(p.device) for p in learner.parameters()})}
        all_optimizers = [learner.critic_optimizer, *learner.actor_optimizers, learner.alpha_optimizer]
        report["fused"] = {"requested": args.fused,
                           "all_param_groups_fused": all(group.get("fused") is True
                               for optimizer in all_optimizers for group in optimizer.param_groups)}
        snapshot("replay_allocation")
        # Full requested original FP32/int64 storage on GPU1. No offload/shard/tuning.
        replay = TensorReplayBuffer(args.capacity, obs["observation"].shape[1:],
                                    state_dim(config), config.runtime.seed, device="cuda:1")
        report["replay"] = {"allocated_bytes": replay.allocated_bytes, "capacity": replay.capacity,
                            "storage": {key: {"dtype": str(value.dtype), "device": str(value.device),
                                               "bytes": value.numel() * value.element_size()}
                                        for key, value in replay.data.items()}}
        snapshot("replay_allocated")
        next_seed = config.runtime.seed + args.envs

        def collect(warm=False):
            nonlocal obs, next_seed
            action = (vector.random_actions(obs["observation"]) if warm else
                      collector.act_batch(obs["observation"], return_tensors=True))
            following, reward, terminated, truncated, infos = vector.step(
                {"continuous": action.continuous, "destination": action.destination})
            replay.add_batch(obs["observation"], obs["state"], action, reward,
                             following["observation"], following["state"], terminated)
            report["collection_steps"] += 1
            report["joint_transitions"] = report["collection_steps"] * args.envs
            obs = following
            indices = list(infos["device_episode_infos"])
            if indices:
                seeds = [None] * args.envs
                for index in indices:
                    seeds[index] = next_seed
                    next_seed += 1
                obs, _ = vector.reset(seed=seeds,
                                     options={"reset_mask": (terminated | truncated).cpu().numpy()})

        for step in range(2):
            snapshot(f"warmup_collection_{step + 1}")
            collect(warm=True)
        amp = ProbeAMP(learner)
        with ExitStack() as stack:
            stack.enter_context(ProbeCompiler(learner, report["compiler"], enabled=args.compile))
            stack.enter_context(amp)
            for round_index in range(args.rounds):
                snapshot(f"round_{round_index + 1}_collection")
                collect()
                for update in range(args.updates_per_round):
                    phase = report["phase"] = f"round_{round_index + 1}_update_{update + 1}"
                    persist()  # Preserve active phase even if the external time limit kills the process.
                    before = {name: value["applied"] for name, value in amp.counts.items()}
                    begin = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.float16):
                        losses = learner.update(replay.sample(args.batch), collect_metrics=True)
                    torch.cuda.synchronize(1)
                    row = {"round": round_index + 1, "update": update + 1,
                           "seconds": time.perf_counter() - begin, "losses": json_safe(losses),
                           "applied": {name: value["applied"] > before[name]
                                       for name, value in amp.counts.items()}}
                    report["updates"].append(row)
                    report["amp"] = amp.evidence()
                    run.log({f"learner/{key}": value for key, value in json_safe(losses).items()},
                            commit=True)
                refresh_base_collector(collector, learner)
                snapshot(f"round_{round_index + 1}_complete")
        report["amp"] = amp.evidence()
        compiler = report["compiler"]
        compiled_executed = (compiler.get("dynamo", {}).get("stats", {}).get("unique_graphs", 0) > 0
                             and compiler.get("inductor_generated_kernels", 0) > 0)
        fp16_executed = all(compiler.get(name, {}).get("fp16_outputs", 0) > 0
                            for name in ("actors", "critic"))
        actual_steps = all(value["applied"] > 0 for value in amp.counts.values())
        if args.compile and not compiled_executed:
            raise RuntimeError("Requested compilation has no executed Dynamo/Inductor graph evidence")
        if not fp16_executed or not actual_steps:
            raise RuntimeError("Missing FP16 forward evidence or an optimizer group never applied a step")
        report["evidence_verified"] = {"fp16_forward": fp16_executed,
                                        "compiled_graphs": compiled_executed,
                                        "all_optimizer_groups_applied": actual_steps,
                                        "fused_optimizer_steps": args.fused and actual_steps and
                                            report["fused"]["all_param_groups_fused"]}
        report["status"] = "passed"
        snapshot("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                      phase=phase, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if amp:
            report["amp"] = amp.evidence()
        # Keep successful world/model stage evidence and per-GPU allocator peaks,
        # even when replay construction failed partway through its dict.
        try:
            snapshot(phase, synchronize=False)
        except Exception as measurement_error:
            report["failure_measurement_error"] = str(measurement_error)
    finally:
        if sampler:
            report["gpu_resources"] = sampler.finish(started, time.perf_counter())
        persist()
        if run:
            try:
                run.summary.update(json_safe({
                    "status": report["status"], "phase": report["phase"],
                    "wall_seconds": report["wall_seconds"], "scope": report["scope"],
                    "source_git_commit": report["source_git_commit"],
                    "probe_sha256": report["probe_sha256"],
                    "replay_expected_bytes": report.get("replay_expected_bytes"),
                    "replay_allocated_bytes": report.get("replay", {}).get("allocated_bytes"),
                    "amp": report.get("amp"), "evidence_verified": report.get("evidence_verified"),
                    "error_type": report.get("error_type"), "error": report.get("error"),
                    "torch_memory": report["stages"][-1]["torch"] if report["stages"] else [],
                    "physical_memory_peaks": [
                        {"uuid": gpu["uuid"], "physical_index": gpu["physical_index"],
                         "bytes": gpu["physical_memory_used_bytes_max"]}
                        for gpu in report.get("gpu_resources", {}).get("gpus", [])],
                    "inductor_generated_kernels": report["compiler"].get("inductor_generated_kernels"),
                    "dynamo_stats": report["compiler"].get("dynamo", {}).get("stats", {})}))
                for filename in ("result.json", "resolved_config.json"):
                    if (output / filename).exists():
                        run.save(str((output / filename).resolve()), base_path=str(output.resolve()), policy="now")
                run.finish(exit_code=0 if report["status"] == "passed" else 1)
            except Exception as error:
                report["tracking_finalize_error"] = f"{type(error).__name__}: {error}"
                report["status"] = "failed"
        if vector:
            try:
                vector.close()
            except Exception as error:
                report["environment_close_error"] = str(error)
                report["status"] = "failed"
        persist()
    return json_safe(report)


if __name__ == "__main__":
    result = run_probe(parser().parse_args())
    print(json.dumps({"status": result["status"], "phase": result["phase"],
                      "result": str(Path(result["requested"]["output"]) / "result.json")}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
