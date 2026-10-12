"""Off-policy training with spawned CPU simulators and one batched learner."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from ..algorithms.checkpoints import make_learner
from ..algorithms.optimization import compile_models
from ..algorithms.replay import TensorReplayBuffer
from ..environment.observations import observation_dim, state_dim
from ..environment.vector import make_vector, random_actions, unpack_info


def seed_runtime(config):
    seed = config.runtime.seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Native numerical libraries in spawned workers must not create large pools.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.runtime.torch_threads)
    torch.use_deterministic_algorithms(config.runtime.deterministic)


def choose_device(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def train(config):
    config.validate()
    seed_runtime(config)
    device = choose_device(config.runtime.device)
    replay_device = device if config.runtime.replay_device == "auto" else choose_device(config.runtime.replay_device)
    settings = config.algorithm_config
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    start = time.perf_counter()
    vector = make_vector(config)
    count = config.runtime.num_envs
    global_step = collection_steps = completed = 0
    latest = {}
    update_s = collection_wait_s = 0.0
    next_metrics_at = 0
    episode_seeds = np.arange(count) + config.runtime.seed
    next_seed = config.runtime.seed + count
    episode_returns, episode_lengths = np.zeros(count), np.zeros(count, dtype=int)
    try:
        obs, _ = vector.reset(seed=episode_seeds.tolist())
        learner = make_learner(config, observation_dim(config), state_dim(config), device)
        if config.runtime.compile_models:
            compile_models(learner)
        replay = TensorReplayBuffer(settings.replay_capacity, obs["observation"].shape[1:],
                                    state_dim(config), config.runtime.seed, replay_device,
                                    pin_memory=device == "cuda" and replay_device == "cpu")
        rng = np.random.default_rng(config.runtime.seed + 1)
        startup_s = time.perf_counter() - start

        def learn(force_metrics=False):
            nonlocal latest, update_s, next_metrics_at
            if global_step < settings.learning_starts or len(replay) < settings.batch_size:
                return
            checked = force_metrics or global_step >= next_metrics_at
            begin = time.perf_counter()
            for update in range(settings.updates_per_step):
                metrics = learner.update(replay.sample(settings.batch_size),
                                         collect_metrics=checked and update == settings.updates_per_step - 1)
                if metrics:
                    latest = metrics
            update_s += time.perf_counter() - begin
            if checked:
                next_metrics_at = global_step + config.runtime.log_interval_steps

        with (output / "episodes.jsonl").open("w") as stream:
            while completed < config.runtime.episodes:
                action = (random_actions(obs["observation"], rng, config.world.num_uavs + 1)
                          if global_step < settings.learning_starts else
                          learner.act_batch(obs["observation"], step=global_step))
                actions = {"continuous": action.continuous, "destination": action.destination}
                overlap = count > 1 and config.runtime.collector_overlap
                if overlap:
                    vector.step_async(actions)
                    learn()  # Use previously stored transitions while CPU workers advance.
                begin = time.perf_counter()
                next_obs, reward, terminated, truncated, infos = (vector.step_wait() if overlap else vector.step(actions))
                collection_wait_s += time.perf_counter() - begin
                # DISABLED autoreset guarantees these are terminal observations/state.
                replay.add_batch(obs["observation"], obs["state"], action, reward,
                                 next_obs["observation"], next_obs["state"], terminated)
                global_step += count
                collection_steps += 1
                episode_returns += reward
                episode_lengths += 1
                done = terminated | truncated
                final_batch = completed + int(done.sum()) >= config.runtime.episodes
                if not overlap or final_batch:
                    # The last collected batch has no following CPU step in
                    # which to execute its deferred updates; drain them here.
                    learn(force_metrics=final_batch)
                for i in np.flatnonzero(done):
                    completed += 1
                    info = unpack_info(infos, i)
                    record = {"episode": completed, "worker": int(i), "seed": int(episode_seeds[i]),
                              "global_step": global_step, "return": float(episode_returns[i]),
                              "episode_length": int(episode_lengths[i]), "terminated": bool(terminated[i]),
                              "truncated": bool(truncated[i]), **info, "learner": latest}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({"algorithm": learner.algorithm, "episode": completed,
                                      "global_step": global_step, "return": round(record["return"], 3),
                                      "search_rate_percent": info["search_rate_percent"],
                                      "delivery_rate_percent": info["delivery_rate_percent"],
                                      "updates": learner.updates}), flush=True)
                    if completed % config.runtime.checkpoint_interval == 0:
                        learner.save(output / f"episode_{completed:06d}.pt")
                obs = next_obs
                if done.any() and completed < config.runtime.episodes:
                    seeds = [None] * count
                    for i in np.flatnonzero(done):
                        seeds[i] = next_seed
                        episode_seeds[i] = next_seed
                        next_seed += 1
                        episode_returns[i] = episode_lengths[i] = 0
                    obs, _ = vector.reset(seed=seeds, options={"reset_mask": done})
        checkpoint = output / "final.pt"
        learner.save(checkpoint)
        elapsed = time.perf_counter() - start
        result = {"checkpoint": str(checkpoint), "algorithm": learner.algorithm,
                  "requested_episodes": config.runtime.episodes, "episodes": completed,
                  "global_step": global_step, "collection_steps": collection_steps,
                  "num_envs": count, "updates": learner.updates, "device": device,
                  "replay_device": replay_device, "replay_bytes": replay.allocated_bytes,
                  "sample_updates_per_transition": learner.updates * settings.batch_size / global_step,
                  "collector_overlap": count > 1 and config.runtime.collector_overlap,
                  "startup_s": startup_s, "collection_wait_s": collection_wait_s, "learner_s": update_s,
                  "elapsed_s": elapsed, "joint_transitions_per_s": global_step / elapsed,
                  "final_episode": record}
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        vector.close()
