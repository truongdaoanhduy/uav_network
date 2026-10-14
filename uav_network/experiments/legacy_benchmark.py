"""Benchmark the pre-PQL fixed scheduler with the oracle-verified combined update."""
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from ..algorithms.checkpoints import make_learner
from ..algorithms.optimization import compile_models
from ..algorithms.replay import TensorReplayBuffer
from ..environment.observations import observation_dim, state_dim
from ..environment.vector import make_vector, random_actions, unpack_info
from .gpu_resources import GPUResourceSampler
from .hardware import hardware_info, synchronize
from .pql_benchmark import peak_memory
from .training import choose_device, seed_runtime


LEGACY_REFERENCE_COMMIT = "afeaa3dba27edd736be13f84b602f84c268e3a22"


def benchmark_legacy(config, steps=40):
    """Preserve step_async -> previous replay updates -> step_wait and final drain."""
    c = deepcopy(config)
    c.pql.enabled = False
    c.tracking.enabled = False
    ratio = float(c.pql.critic_sample_ratio)
    if not ratio.is_integer():
        raise ValueError("Legacy fixed scheduler requires an integer critic_sample_ratio")
    settings = c.algorithm_config
    settings.learning_starts = c.pql.warmup_collection_steps * c.runtime.num_envs
    settings.updates_per_step = int(ratio)
    c.validate()
    if c.runtime.warm_start_checkpoint != "none":
        raise ValueError("Comparison benchmark requires a fresh learner without a warm-start checkpoint")
    if steps <= c.pql.warmup_collection_steps or steps * c.runtime.num_envs < settings.batch_size:
        raise ValueError("Benchmark must include post-warmup learner updates")
    first_update_collection = max(c.pql.warmup_collection_steps,
                                  math.ceil(settings.batch_size / c.runtime.num_envs), 1)
    expected_updates = (steps - first_update_collection + 1) * settings.updates_per_step
    delay = c.matd3.policy_delay if c.runtime.algorithm == "matd3" else 1
    if expected_updates < delay:
        raise ValueError("Benchmark budget must exercise both critic and policy updates")

    seed_runtime(c)
    output = Path(c.runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(c.to_dict(), sort_keys=False))
    for name in ("diagnostics.jsonl", "episodes.jsonl"):
        (output / name).touch()
    # CUDA allocator counters are invalid until runtime initialization in a fresh child.
    if torch.cuda.device_count():
        torch.cuda.init()
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    started = time.perf_counter()
    sampler = GPUResourceSampler()
    vector = learner = replay = None
    window_start = window_end = None
    global_step = completed_episodes = reset_collections = 0
    episode_records = []
    latest = {}
    learner_s = collection_wait_s = 0.0
    next_metrics_at = 0
    overlap = c.runtime.num_envs > 1 and c.runtime.collector_overlap
    report = {"status": "failed", "cuda_pipeline_verified": False, "target_t4_scenario_verified": False,
              "source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
              "legacy_reference_commit": LEGACY_REFERENCE_COMMIT,
              "baseline_implementation": "Legacy fixed scheduler and TensorReplayBuffer, using the current "
                                         "combined update wrapper verified against frozen legacy equations",
              "algorithm": c.runtime.algorithm, "num_envs": c.runtime.num_envs,
              "batch_size": settings.batch_size, "replay_capacity": settings.replay_capacity,
              "learning_starts": settings.learning_starts, "updates_per_step": settings.updates_per_step,
              "critic_sample_ratio": ratio, "warmup_collection_steps": c.pql.warmup_collection_steps,
              "collector_overlap": overlap, "world": c.to_dict()["world"],
              "hardware": hardware_info(), "steps": steps,
              "target_sample_uses_per_new_row": ratio * settings.batch_size / c.runtime.num_envs,
              "tracking_enabled": False,
              "timing_scope": "Collection and completed learner work, including warmup; startup excluded",
              "torch_memory_scope": "Process-local PyTorch caching allocator peaks; Warp allocations excluded"}

    def learn(force_metrics=False):
        nonlocal latest, learner_s, next_metrics_at
        if global_step < settings.learning_starts or len(replay) < settings.batch_size:
            return
        checked = force_metrics or global_step >= next_metrics_at
        begin = time.perf_counter()
        for update in range(settings.updates_per_step):
            batch = replay.sample(settings.batch_size)
            try:
                metrics = learner.update(batch, collect_metrics=checked and update == settings.updates_per_step - 1)
            except (FloatingPointError, RuntimeError):
                torch.save({key: value.detach().cpu() for key, value in batch.items()}, output / "failed_batch.pt")
                raise
            if metrics:
                latest = metrics
        learner_s += time.perf_counter() - begin
        if checked:
            next_metrics_at = global_step + c.runtime.log_interval_steps

    try:
        sampler.start()
        selected = choose_device(c.runtime.device)
        device = torch.device("cuda:0" if selected == "cuda" else selected)
        # The original same-GPU baseline keeps learner and replay on collector GPU0.
        replay_device = device
        learner = make_learner(c, observation_dim(c), state_dim(c), device)
        if c.runtime.compile_models:
            compile_models(learner)
        vector = make_vector(c, device=device)
        obs, _ = vector.reset(seed=c.runtime.seed)
        replay = TensorReplayBuffer(settings.replay_capacity, obs["observation"].shape[1:],
                                    state_dim(c), c.runtime.seed, replay_device)
        rng = np.random.default_rng(c.runtime.seed + 1)
        next_seed = c.runtime.seed + c.runtime.num_envs
        device_env = c.runtime.environment_backend == "warp_packet"
        synchronize(device)
        report["startup_s"] = time.perf_counter() - started
        window_start = time.perf_counter()
        for iteration in range(steps):
            if global_step < settings.learning_starts:
                action = (vector.random_actions(obs["observation"]) if device_env else
                          random_actions(obs["observation"], rng, c.world.num_uavs + 1))
            else:
                action = learner.act_batch(obs["observation"], step=global_step, return_tensors=device_env)
            actions = {"continuous": action.continuous, "destination": action.destination}
            if overlap:
                vector.step_async(actions)
                learn()
            begin = time.perf_counter()
            following, reward, terminated, truncated, infos = (
                vector.step_wait() if overlap else vector.step(actions))
            collection_wait_s += time.perf_counter() - begin
            replay.add_batch(obs["observation"], obs["state"], action, reward,
                             following["observation"], following["state"], terminated)
            global_step += c.runtime.num_envs
            done = terminated | truncated
            indices = list(infos["device_episode_infos"]) if device_env else np.flatnonzero(done).tolist()
            final_batch = iteration == steps - 1
            if not overlap or final_batch:
                learn(force_metrics=final_batch)
            for index in indices:
                completed_episodes += 1
                record = {"episode": completed_episodes, "worker": int(index),
                          "global_step": global_step, **unpack_info(infos, index)}
                episode_records.append(record)
            obs = following
            if indices and not final_batch:
                reset_collections += 1
                seeds = [None] * c.runtime.num_envs
                for index in indices:
                    seeds[index] = next_seed
                    next_seed += 1
                obs, _ = vector.reset(seed=seeds, options={
                    "reset_mask": done.cpu().numpy() if device_env else done})
        synchronize(device)
        window_end = time.perf_counter()
        elapsed = window_end - window_start
        with (output / "episodes.jsonl").open("w") as stream:
            for record in episode_records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        report.update(status="passed", elapsed_s=elapsed, completed_episodes=completed_episodes,
                      reset_collections=reset_collections,
                      collector_iterations_per_s=steps / elapsed,
                      joint_transitions_per_s=global_step / elapsed,
                      critic_updates=learner.updates, actor_updates=learner.actor_updates,
                      critic_updates_per_s=learner.updates / elapsed,
                      actor_updates_per_s=learner.actor_updates / elapsed,
                      combined_learner_seconds=learner_s, collection_wait_seconds=collection_wait_s,
                      replay_bytes=replay.allocated_bytes,
                      environment_bytes=getattr(vector, "allocated_bytes", None),
                      num_cpu_env_workers=vector.num_cpu_workers if device_env else
                                          (c.runtime.num_envs if c.runtime.num_envs > 1 else 0),
                      devices={"collector": str(device), "critic": str(device), "policy": str(device)},
                      learning_metrics=latest)
        if learner.updates != expected_updates or learner.actor_updates != expected_updates // delay:
            raise RuntimeError("Legacy benchmark did not complete the matched critic and policy update budget")
        report["cuda_pipeline_verified"] = device.type == "cuda"
        report["peak_memory"] = peak_memory()
    except Exception as error:
        cause = error
        while cause.__cause__ is not None:
            cause = cause.__cause__
        report.update(status="oom" if isinstance(cause, torch.OutOfMemoryError) else "failed",
                      error_type=type(cause).__name__, error=str(cause), peak_memory=peak_memory())
    finally:
        if window_end is None:
            window_end = time.perf_counter()
        if window_start is None:
            window_start = window_end
        report["gpu_resources"] = sampler.finish(window_start, window_end)
        if vector:
            vector.close()
        report["total_wall_seconds"] = time.perf_counter() - started
        with (output / "diagnostics.jsonl").open("a") as diagnostics:
            diagnostics.write(json.dumps({"status": report["status"], "global_step": global_step,
                "timing/collector_wait_seconds": collection_wait_s,
                "timing/combined_learner_seconds": learner_s,
                "critic_updates": getattr(learner, "updates", 0),
                "actor_updates": getattr(learner, "actor_updates", 0)}, allow_nan=False) + "\n")
        (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        del learner, vector, replay
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report
