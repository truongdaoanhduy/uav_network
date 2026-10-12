"""One-environment MASAC baseline; streaming episode metrics and checkpoints."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from ..algorithms.masac import MASAC
from ..algorithms.replay import ReplayBuffer
from ..environment.env import Action, SwarmSearchEnv


def seed_runtime(config):
    seed = config.runtime.seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.runtime.torch_threads)
    torch.use_deterministic_algorithms(True)


def choose_device(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def train(config):
    config.validate()
    seed_runtime(config)
    device = choose_device(config.runtime.device)
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    env = SwarmSearchEnv(config)
    global_step = 0
    latest = {}
    start = time.perf_counter()
    try:
        obs, _ = env.reset(seed=config.runtime.seed)
        learner = MASAC(config, env.observation_dim, env.state_dim, device)
        replay = ReplayBuffer(config.masac.replay_capacity, obs.shape, env.state_dim,
                              seed=config.runtime.seed)
        rng = np.random.default_rng(config.runtime.seed + 1)
        with (output / "episodes.jsonl").open("w") as stream:
            for episode in range(config.runtime.episodes):
                if episode:
                    obs, _ = env.reset(seed=config.runtime.seed + episode)
                episode_return = 0.0
                while True:
                    state = env.state()
                    if global_step < config.masac.learning_starts:
                        action = Action(rng.uniform(-1, 1, (config.world.num_uavs, 3)),
                                        rng.integers(0, config.world.num_uavs + 1, config.world.num_uavs))
                    else:
                        action = learner.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    replay.add(obs, state, action, reward, next_obs, env.state(), terminated)
                    obs = next_obs
                    global_step += 1
                    episode_return += reward
                    if global_step >= config.masac.learning_starts and len(replay) >= config.masac.batch_size:
                        for _ in range(config.masac.updates_per_step):
                            latest = learner.update(replay.sample(config.masac.batch_size))
                    if terminated or truncated:
                        record = {"episode": episode + 1, "global_step": global_step,
                                  "return": episode_return, "episode_length": env.steps,
                                  "terminated": terminated, "truncated": truncated,
                                  **info, "learner": latest}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                        print(json.dumps({"episode": episode + 1, "global_step": global_step,
                                          "return": round(episode_return, 3),
                                          "search_rate_percent": info["search_rate_percent"],
                                          "delivery_rate_percent": info["delivery_rate_percent"],
                                          "updates": learner.updates}), flush=True)
                        break
                if (episode + 1) % config.runtime.checkpoint_interval == 0:
                    learner.save(output / f"episode_{episode + 1:06d}.pt")
        checkpoint = output / "final.pt"
        learner.save(checkpoint)
        result = {"checkpoint": str(checkpoint), "episodes": config.runtime.episodes,
                  "global_step": global_step, "updates": learner.updates, "device": device,
                  "elapsed_s": time.perf_counter() - start, "final_episode": record}
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        env.close()
