"""CPU checkpoint evaluation using the scenario config saved with that policy."""
import json
from pathlib import Path
import numpy as np
import torch

from ..algorithms.checkpoints import load_policy
from ..environment.env import SwarmSearchEnv
from .rendering import render_episode


def evaluate(checkpoint, episodes=None, output_dir=None, seed=None):
    learner = load_policy(checkpoint, device="cpu")
    config = learner.config
    torch.set_num_threads(config.runtime.torch_threads)
    episodes = config.runtime.eval_episodes if episodes is None else episodes
    seed = config.runtime.eval_seed if seed is None else seed
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Evaluation episodes must be a positive integer")
    output = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = SwarmSearchEnv(config)
    records = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            paths = [env.positions.copy()]
            episode_return = 0.0
            while True:
                obs, reward, terminated, truncated, info = env.step(learner.act(obs, deterministic=True))
                episode_return += reward
                paths.append(env.positions.copy())
                if terminated or truncated:
                    records.append({"episode": episode + 1, "seed": seed + episode,
                                    "return": episode_return, "episode_length": env.steps, **info})
                    break
        preview = render_episode(env, paths, output / "final_episode.png")
        result = {"algorithm": learner.algorithm, "episodes": episodes, "device": "cpu", "render_path": preview,
                  "mean_return": float(np.mean([r["return"] for r in records])),
                  "success_rate": float(np.mean([r["success"] for r in records])),
                  "mean_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
                  "mean_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
                  "records": records}
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        env.close()
