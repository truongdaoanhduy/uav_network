"""Measure process collection and warmed learner throughput separately."""
from copy import deepcopy
import math
import time

import numpy as np

from ..algorithms.checkpoints import make_learner
from ..algorithms.optimization import compile_models
from ..algorithms.replay import TensorReplayBuffer
from ..environment.observations import observation_dim, state_dim
from ..environment.vector import make_vector, random_actions
from .hardware import hardware_info, synchronize
from .training import choose_device, seed_runtime


def benchmark(config, env_counts=(1, 2, 4), steps=32, learner_updates=10):
    if steps < 1 or learner_updates < 1 or any(n < 1 for n in env_counts):
        raise ValueError("Benchmark counts, steps and updates must be positive")
    results = []
    for count in env_counts:
        c = deepcopy(config)
        c.runtime.num_envs = count
        c.validate()
        seed_runtime(c)
        device = choose_device(c.runtime.device)
        replay_device = device if c.runtime.replay_device == "auto" else choose_device(c.runtime.replay_device)
        settings = c.algorithm_config
        rounds = max(steps, math.ceil(settings.batch_size / count))
        started = time.perf_counter()
        vector = make_vector(c)
        try:
            obs, _ = vector.reset(seed=c.runtime.seed)
            startup = time.perf_counter() - started
            replay = TensorReplayBuffer(settings.replay_capacity, obs["observation"].shape[1:],
                                        state_dim(c), c.runtime.seed, replay_device,
                                        pin_memory=device == "cuda" and replay_device == "cpu")
            rng = np.random.default_rng(c.runtime.seed + 1)
            next_seed = c.runtime.seed + count
            collector_s = reset_s = 0.0
            for _ in range(rounds):
                begin = time.perf_counter()
                action = random_actions(obs["observation"], rng, c.world.num_uavs + 1)
                final, rewards, terminated, truncated, _ = vector.step({
                    "continuous": action.continuous, "destination": action.destination})
                collector_s += time.perf_counter() - begin
                replay.add_batch(obs["observation"], obs["state"], action, rewards,
                                 final["observation"], final["state"], terminated)
                obs = final
                done = terminated | truncated
                if done.any():
                    begin = time.perf_counter()
                    seeds = [None] * count
                    for i in np.flatnonzero(done):
                        seeds[i] = next_seed
                        next_seed += 1
                    obs, _ = vector.reset(seed=seeds, options={"reset_mask": done})
                    reset_s += time.perf_counter() - begin
            learner = make_learner(c, observation_dim(c), state_dim(c), device)
            if c.runtime.compile_models:
                compile_models(learner)
            begin = time.perf_counter()
            for _ in range(2):
                learner.update(replay.sample(settings.batch_size))
            synchronize(device)
            warmup_s = time.perf_counter() - begin
            begin = time.perf_counter()
            for _ in range(learner_updates):
                learner.update(replay.sample(settings.batch_size), collect_metrics=False)
            synchronize(device)
            learner_s = time.perf_counter() - begin
            results.append({"num_envs": count, "algorithm": learner.algorithm,
                            "device": device, "replay_device": replay_device,
                            "vector_steps": rounds, "joint_transitions": rounds * count,
                            "startup_s": startup, "collector_s": collector_s, "reset_s": reset_s,
                            "joint_transitions_per_s": rounds * count / collector_s,
                            "learner_batch_size": settings.batch_size,
                            "learner_warmup_s": warmup_s, "learner_updates": learner_updates,
                            "learner_s": learner_s, "learner_updates_per_s": learner_updates / learner_s,
                            "replay_bytes": replay.allocated_bytes,
                            "compile_models": c.runtime.compile_models,
                            "coalesce_idle_polls": c.runtime.coalesce_idle_polls,
                            "fused_optimizers": c.runtime.fused_optimizers and device == "cuda",
                            "vectorize_critics": c.runtime.vectorize_critics})
        finally:
            vector.close()
    return {"hardware": hardware_info(), "network_enabled": config.network.enabled,
            "num_uavs_per_env": config.world.num_uavs, "results": results,
            "timing_notes": "Collection includes random action sampling and worker wait, excludes replay writes/startup/resets. Learner timing is warmed and CUDA-synchronized."}
