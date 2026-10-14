"""Measure the actual PQL pipeline and peak VRAM; report failures without tuning."""
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from ..environment.vector import make_vector, random_actions
from .hardware import hardware_info
from .pql_system import PQLSystem
from .tracking import start_tracking
from .training import seed_runtime


def peak_memory():
    return [{"device": index, "name": torch.cuda.get_device_name(index),
             "total_bytes": torch.cuda.get_device_properties(index).total_memory,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)}
            for index in range(torch.cuda.device_count())]


def benchmark_pql(config, steps=40):
    c = deepcopy(config).validate()
    if not c.pql.enabled:
        raise ValueError("PQL benchmark requires pql.enabled=true")
    if steps <= c.pql.warmup_collection_steps or steps * c.runtime.num_envs < c.algorithm_config.batch_size:
        raise ValueError("Benchmark must include post-warmup learner updates")
    first_update_collection = max(c.pql.warmup_collection_steps,
                                  math.ceil(c.algorithm_config.batch_size / c.runtime.num_envs), 1)
    expected_updates = math.floor((steps - first_update_collection + 1) * c.pql.critic_sample_ratio + 1e-10)
    delay = c.matd3.policy_delay if c.runtime.algorithm == "matd3" else 1
    if expected_updates < delay:
        raise ValueError("Benchmark budget must exercise both critic and policy updates")
    seed_runtime(c)
    output = Path(c.runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(__import__("yaml").safe_dump(c.to_dict(), sort_keys=False))
    for name in ("diagnostics.jsonl", "episodes.jsonl"):
        (output / name).touch()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    engine = vector = tracker = None
    started = time.perf_counter()
    report = {"status": "failed", "cuda_pipeline_verified": False, "target_t4_scenario_verified": False,
              "source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
              "algorithm": c.runtime.algorithm, "num_envs": c.runtime.num_envs,
              "batch_size": c.algorithm_config.batch_size, "replay_capacity": c.algorithm_config.replay_capacity,
              "critic_sample_ratio": c.pql.critic_sample_ratio, "world": c.to_dict()["world"],
              "hardware": hardware_info(), "steps": steps,
              "target_sample_uses_per_new_row": c.pql.critic_sample_ratio *
                                              c.algorithm_config.batch_size / c.runtime.num_envs}
    try:
        tracker = start_tracking(c, output)
        engine = PQLSystem(c)
        vector = make_vector(c, device=engine.collector_device)
        obs, _ = vector.reset(seed=c.runtime.seed)
        report["startup_s"] = time.perf_counter() - started
        rng = np.random.default_rng(c.runtime.seed + 1)
        next_seed = c.runtime.seed + c.runtime.num_envs
        collected_s = time.perf_counter()
        for iteration in range(steps):
            begin = time.perf_counter()
            warm = engine.collections < c.pql.warmup_collection_steps
            device_env = c.runtime.environment_backend == "warp_packet"
            if warm:
                action = vector.random_actions(obs["observation"]) if device_env else random_actions(
                    obs["observation"], rng, c.world.num_uavs + 1)
            else:
                action = engine.act_batch(obs["observation"], step=iteration * c.runtime.num_envs,
                                          return_tensors=device_env)
            following, reward, terminated, truncated, infos = vector.step({
                "continuous": action.continuous, "destination": action.destination})
            engine.record_collection_time(time.perf_counter() - begin)
            engine.submit((obs["observation"], obs["state"], action, reward,
                           following["observation"], following["state"], terminated))
            obs = following
            done = terminated | truncated
            indices = list(infos["device_episode_infos"]) if device_env else np.flatnonzero(done).tolist()
            if indices:
                seeds = [None] * c.runtime.num_envs
                for index in indices:
                    seeds[index] = next_seed
                    next_seed += 1
                obs, _ = vector.reset(seed=seeds, options={"reset_mask": done.cpu().numpy() if device_env else done})
        engine.flush()
        elapsed = time.perf_counter() - collected_s
        metrics = engine.metrics()
        report.update(status="passed", elapsed_s=elapsed,
                      collector_iterations_per_s=steps / elapsed,
                      joint_transitions_per_s=steps * c.runtime.num_envs / elapsed,
                      critic_updates=engine.critic.updates, actor_updates=engine.policy.actor_updates,
                      critic_updates_per_s=engine.critic.updates / elapsed,
                      actor_updates_per_s=engine.policy.actor_updates / elapsed,
                      critic_active_seconds=engine.critic_s - engine.policy_wait_s,
                      policy_seconds=engine.actor_s, replay_bytes=engine.replay.allocated_bytes,
                      environment_bytes=getattr(vector, "allocated_bytes", None),
                      num_cpu_env_workers=vector.num_cpu_workers if device_env else c.runtime.num_envs,
                      devices={"collector": str(engine.collector_device), "critic": str(engine.critic_device),
                               "policy": str(engine.policy_device)}, metrics=metrics)
        if engine.critic.updates < 1 or engine.policy.actor_updates < 1:
            raise RuntimeError("Benchmark did not complete both critic and policy updates")
        report["cuda_pipeline_verified"] = all(device.type == "cuda" for device in
            (engine.collector_device, engine.critic_device, engine.policy_device))
        report["target_t4_scenario_verified"] = (
            report["cuda_pipeline_verified"] and
            (str(engine.collector_device), str(engine.critic_device), str(engine.policy_device)) ==
                ("cuda:0", "cuda:1", "cuda:1") and
            all("t4" in torch.cuda.get_device_name(index).lower() for index in (0, 1)) and
            (c.world.num_uavs, c.world.num_targets, c.world.num_obstacles, c.world.max_steps,
             c.runtime.num_envs, c.algorithm_config.batch_size, c.algorithm_config.replay_capacity) ==
                (6, 25, 15, 1000, 4096, 8192, 500000))
        report["peak_memory"] = peak_memory()
        report["learning_metrics"] = dict(engine.latest)
        if tracker:
            tracker.diagnostic(metrics, steps * c.runtime.num_envs)
            tracker.learner(engine.latest, steps * c.runtime.num_envs)
    except Exception as error:
        cause = error
        while cause.__cause__ is not None:
            cause = cause.__cause__
        report.update(status="oom" if isinstance(cause, torch.OutOfMemoryError) else "failed",
                      error_type=type(cause).__name__, error=str(cause), peak_memory=peak_memory())
    finally:
        try:
            if engine:
                engine.close()
        finally:
            if vector:
                vector.close()
            report["total_wall_seconds"] = time.perf_counter() - started
            path = output / "benchmark.json"
            path.write_text(json.dumps(report, indent=2, allow_nan=False))
            if tracker:
                report.update(tracker.identity())
                tracker.run.summary.update(report)
                tracker.run.save(str(path), base_path=str(output), policy="now")
                tracker.close(failed=report["status"] != "passed")
            del engine, vector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return report
