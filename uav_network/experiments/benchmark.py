"""Measure process collection and warmed learner throughput separately."""
from copy import deepcopy
import math
import time

import numpy as np
import torch

from ..algorithms.checkpoints import make_learner
from ..algorithms.optimization import compile_models
from ..algorithms.replay import TensorReplayBuffer
from ..environment.observations import observation_dim, state_dim
from ..environment.vector import make_vector, random_actions
from .hardware import hardware_info, synchronize
from .training import resolve_base_devices, seed_runtime


def peak_memory():
    return [{"device": index, "name": torch.cuda.get_device_name(index),
             "total_bytes": torch.cuda.get_device_properties(index).total_memory,
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)}
            for index in range(torch.cuda.device_count())]


def dual_gpu_verified(devices, actor_updates, peak_memory):
    if (devices.get("collector"), devices.get("learner"), devices.get("replay")) != ("cuda:0", "cuda:1", "cuda:1"):
        return False
    used = {row.get("device"): row.get("peak_allocated_bytes", 0) for row in peak_memory}
    return actor_updates > 0 and used.get(0, 0) > 0 and used.get(1, 0) > 0


def benchmark(config, env_counts=(1, 2, 4), steps=32, learner_updates=10):
    if steps < 1 or learner_updates < 1 or any(n < 1 for n in env_counts):
        raise ValueError("Benchmark counts, steps and updates must be positive")
    results = []
    for count in env_counts:
        c = deepcopy(config)
        c.runtime.num_envs = count
        c.validate()
        seed_runtime(c)
        collector_device, learner_device, replay_device = resolve_base_devices(c)
        if torch.cuda.device_count():
            torch.cuda.init()
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
        settings = c.algorithm_config
        rounds = max(steps, math.ceil(settings.batch_size / count))
        started = time.perf_counter()
        vector = None
        try:
            vector = make_vector(c, device=collector_device)
            obs, _ = vector.reset(seed=c.runtime.seed)
            startup = time.perf_counter() - started
            replay = TensorReplayBuffer(settings.replay_capacity, obs["observation"].shape[1:],
                                        state_dim(c), c.runtime.seed, replay_device,
                                        pin_memory=learner_device.type == "cuda" and replay_device.type == "cpu")
            rng = np.random.default_rng(c.runtime.seed + 1)
            next_seed = c.runtime.seed + count
            collector_s = reset_s = 0.0
            device_env = c.runtime.environment_backend == "warp_packet"
            for _ in range(rounds):
                begin = time.perf_counter()
                action = (vector.random_actions(obs["observation"]) if device_env else
                          random_actions(obs["observation"], rng, c.world.num_uavs + 1))
                final, rewards, terminated, truncated, infos = vector.step({
                    "continuous": action.continuous, "destination": action.destination})
                collector_s += time.perf_counter() - begin
                replay.add_batch(obs["observation"], obs["state"], action, rewards,
                                 final["observation"], final["state"], terminated)
                obs = final
                done = terminated | truncated
                indices = list(infos["device_episode_infos"]) if device_env else np.flatnonzero(done).tolist()
                if indices:
                    begin = time.perf_counter()
                    seeds = [None] * count
                    for i in indices:
                        seeds[i] = next_seed
                        next_seed += 1
                    obs, _ = vector.reset(seed=seeds, options={
                        "reset_mask": done.cpu().numpy() if device_env else done})
                    reset_s += time.perf_counter() - begin
            learner = make_learner(c, observation_dim(c), state_dim(c), learner_device)
            if c.runtime.compile_models:
                compile_models(learner)
            begin = time.perf_counter()
            for _ in range(2):
                learner.update(replay.sample(settings.batch_size))
            synchronize(learner_device)
            warmup_s = time.perf_counter() - begin
            begin = time.perf_counter()
            for _ in range(learner_updates):
                learner.update(replay.sample(settings.batch_size), collect_metrics=False)
            synchronize(learner_device)
            learner_s = time.perf_counter() - begin
            devices = {"collector": str(collector_device), "learner": str(learner_device),
                       "replay": str(replay_device)}
            peaks = peak_memory() if torch.cuda.device_count() else []
            results.append({"status": "passed", "num_envs": count, "algorithm": learner.algorithm,
                            "device": str(learner_device), "replay_device": str(replay_device),
                            "devices": devices, "vector_steps": rounds,
                            "joint_transitions": rounds * count,
                            "startup_s": startup, "collector_s": collector_s, "reset_s": reset_s,
                            "joint_transitions_per_s": rounds * count / collector_s,
                            "learner_batch_size": settings.batch_size,
                            "learner_warmup_s": warmup_s, "learner_updates": learner_updates,
                            "actor_updates": learner.actor_updates,
                            "learner_s": learner_s, "learner_updates_per_s": learner_updates / learner_s,
                            "replay_bytes": replay.allocated_bytes,
                            "environment_bytes": getattr(vector, "allocated_bytes", None),
                            "peak_memory": peaks,
                            "cuda_pipeline_verified": all(value.startswith("cuda:") for value in devices.values())
                                                      and learner.actor_updates > 0,
                            "dual_gpu_verified": dual_gpu_verified(devices, learner.actor_updates, peaks),
                            "compile_models": c.runtime.compile_models,
                            "coalesce_idle_polls": c.runtime.coalesce_idle_polls,
                            "fused_optimizers": c.runtime.fused_optimizers and learner_device.type == "cuda",
                            "vectorize_critics": c.runtime.vectorize_critics})
        except torch.OutOfMemoryError as error:
            devices = {"collector": str(collector_device), "learner": str(learner_device),
                       "replay": str(replay_device)}
            results.append({"status": "oom", "num_envs": count,
                            "algorithm": c.runtime.algorithm, "devices": devices,
                            "replay_capacity": settings.replay_capacity,
                            "learner_batch_size": settings.batch_size,
                            "error_type": type(error).__name__, "error": str(error),
                            "peak_memory": peak_memory() if torch.cuda.device_count() else []})
        finally:
            if vector is not None:
                vector.close()
    return {"hardware": hardware_info(), "network_enabled": config.network.enabled,
            "num_uavs_per_env": config.world.num_uavs, "results": results,
            "timing_notes": "Collection includes random action sampling and worker wait, excludes replay writes/startup/resets. Learner timing is warmed and CUDA-synchronized."}
