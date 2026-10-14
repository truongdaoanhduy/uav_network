"""CPU checkpoint evaluation using the scenario config saved with that policy."""
import json
from pathlib import Path
import numpy as np
import torch

from ..algorithms.checkpoints import load_policy
from ..environment.env import SwarmSearchEnv
from .rendering import animation_frame, render_animation, render_episode


def evaluate(checkpoint, episodes=None, output_dir=None, seed=None, animate=False,
             frame_stride=5, fps=10, video_format="auto"):
    if type(frame_stride) is not int or frame_stride < 1 or type(fps) is not int or fps < 1:
        raise ValueError("frame_stride and fps must be positive integers")
    if video_format not in {"auto", "mp4", "gif"}:
        raise ValueError("video_format must be auto, mp4 or gif")
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
    if config.runtime.environment_backend == "warp_packet":
        from ..environment.gpu.vector import WarpMissionEnv
        env = WarpMissionEnv(config)
    else:
        env = SwarmSearchEnv(config)
    records = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            paths = [env.positions.copy()]
            frames = [animation_frame(env)] if animate and episode == episodes - 1 else []
            episode_return = 0.0
            while True:
                obs, reward, terminated, truncated, info = env.step(learner.act(obs, deterministic=True))
                episode_return += reward
                paths.append(env.positions.copy())
                if animate and episode == episodes - 1 and (env.steps % frame_stride == 0 or terminated or truncated):
                    frames.append(animation_frame(env))
                if terminated or truncated:
                    records.append({"episode": episode + 1, "seed": seed + episode,
                                    "return": episode_return, "episode_length": env.steps, **info})
                    break
        preview = render_episode(env, paths, output / "final_episode.png")
        result = {"algorithm": learner.algorithm, "episodes": episodes, "device": "cpu", "render_path": preview,
                  "environment_backend": config.runtime.environment_backend,
                  "mean_return": float(np.mean([r["return"] for r in records])),
                  "success_rate": float(np.mean([r["success"] for r in records])),
                  "mean_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
                  "mean_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
                  "records": records}
        if animate:
            result["animation_path"] = render_animation(env, paths, frames, output / "final_episode", fps, video_format)
            result["animation"] = {"sampled_steps": [frame["step"] for frame in frames],
                                   "frame_stride": frame_stride, "fps": fps, "episode": episodes,
                                   "seed": seed + episodes - 1,
                                   "playback_speed": frame_stride * fps * config.world.dt_s}
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        env.close()
