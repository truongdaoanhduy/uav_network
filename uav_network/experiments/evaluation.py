"""CPU checkpoint evaluation using the scenario config saved with that policy."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from ..algorithms.checkpoints import load_policy
from ..environment.env import SwarmSearchEnv
from .rendering import COMMUNICATION_WINDOW_S, animation_frame, render_animation, render_episode
from .comparison_metrics import MissionMetrics, evaluation_protocol, summarize_metrics
from .evaluation_snapshots import load_evaluation_snapshot
from .telemetry import OutcomePool


def make_evaluation_env(config):
    """Use a single CPU world without changing the checkpoint configuration."""
    if config.runtime.environment_backend == "warp_packet":
        from ..environment.gpu.vector import WarpMissionEnv
        return WarpMissionEnv(config)
    return SwarmSearchEnv(config)


def load_evaluation_policy(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    learner = (load_evaluation_snapshot(saved) if saved.get("purpose") == "evaluation_only"
               else load_policy(checkpoint, device="cpu", payload=saved))
    return learner, dict(saved.get("progress", {})), saved.get("source")


def run_episode(env, learner, *, seed, episode=1, action_control=None, trace=False,
                animate=False, frame_stride=5, capture_paths=False, on_progress=None):
    """Shared observed episode loop; traces contain successful whole-report hops."""
    if trace or animate:
        env.enable_communication_trace()
    obs, _ = env.reset(seed=seed)
    layout = {"positions": env.positions.tolist(), "targets": env.targets.tolist(),
              "obstacles": env.obstacles.tolist(), "gcs": env.gcs.tolist()}
    layout_hash = hashlib.sha256(json.dumps(layout, sort_keys=True, allow_nan=False).encode()).hexdigest()
    paths = [env.positions.copy()] if capture_paths or animate else []
    mission = MissionMetrics(env.config, env.positions, env.obstacles,
                             env.reports.known.any(axis=0), env.reports.delivered)
    frames = [animation_frame(env)] if animate else []
    events = []
    episode_return = 0.0
    if on_progress is not None:
        on_progress({"status": "episode_started", "seed": seed, "step": 0})
    while True:
        action = learner.act(obs, deterministic=True)
        if action_control is not None:
            action = action_control(action, obs, env.config)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        if capture_paths or animate:
            paths.append(env.positions.copy())
        mission.observe(env.positions, env.reports.known.any(axis=0), env.reports.delivered, env.steps)
        if animate:
            if env.steps % frame_stride == 0 or terminated or truncated:
                frames.append(animation_frame(env))
                events.extend(frames[-1]["communication_events"])
        elif trace:
            events.extend(env.drain_communication_events())
        if on_progress is not None:
            on_progress({"status": "episode_complete" if terminated or truncated else "running",
                         "seed": seed, "step": env.steps, "max_steps": env.config.world.max_steps,
                         "targets_found": info.get("targets_found"),
                         "targets_delivered": info.get("targets_delivered")})
        if terminated or truncated:
            record = {"episode": episode, "seed": seed, "return": episode_return,
                      "environment_backend": env.config.runtime.environment_backend,
                      "scene_generator": env.config.runtime.environment_backend,
                      "layout_hash": layout_hash,
                      "episode_length": env.steps, **info,
                      "comparison_metrics": mission.finish(info, episode_return)}
            return record, paths, frames, events


class EvaluationPaused(RuntimeError):
    """A bounded CPU session stopped between deterministic episodes."""
    def __init__(self, records):
        super().__init__("Evaluation session paused between episodes")
        self.records = list(records)


def evaluate(checkpoint, episodes=None, output_dir=None, seed=None, animate=False,
             frame_stride=5, fps=10, video_format="auto", on_episode=None,
             completed_records=None, should_pause=None, on_checkpoint=None, on_progress=None):
    if type(frame_stride) is not int or frame_stride < 1 or type(fps) is not int or fps < 1:
        raise ValueError("frame_stride and fps must be positive integers")
    if video_format not in {"auto", "mp4", "gif"}:
        raise ValueError("video_format must be auto, mp4 or gif")
    learner, progress, source = load_evaluation_policy(checkpoint)
    config = learner.config
    torch.set_num_threads(config.runtime.torch_threads)
    episodes = config.runtime.eval_episodes if episodes is None else episodes
    seed = config.runtime.eval_seed if seed is None else seed
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Evaluation episodes must be a positive integer")
    records = list(completed_records or [])
    if (len(records) > episodes or any(not isinstance(record, dict)
            or record.get("episode") != i + 1 or record.get("seed") != seed + i
            for i, record in enumerate(records))):
        raise ValueError("Evaluation resume needs the exact episode/seed prefix")
    # The final scene is recreated if a session ended during media rendering.
    if len(records) == episodes:
        records.pop()
    output = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evaluation folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = make_evaluation_env(config)
    try:
        for episode in range(len(records), episodes):
            if should_pause is not None and should_pause():
                raise EvaluationPaused(records)
            record, paths, frames, events = run_episode(
                env, learner, seed=seed + episode, episode=episode + 1,
                animate=animate and episode == episodes - 1, frame_stride=frame_stride,
                capture_paths=True, on_progress=on_progress)
            records.append(record)
            if on_checkpoint is not None:
                on_checkpoint(records)
            if on_episode is not None:
                on_episode(record)
        preview = render_episode(env, paths, output / "final_episode.png")
        result = {"algorithm": learner.algorithm, "episodes": episodes, "device": "cpu", "render_path": preview,
                  "environment_backend": config.runtime.environment_backend,
                  "mean_return": float(np.mean([r["return"] for r in records])),
                  "success_rate": float(np.mean([r["success"] for r in records])),
                  "mean_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
                  "mean_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
                  "protocol": evaluation_protocol(config, [r["seed"] for r in records], progress),
                  "metric_summary": summarize_metrics(records),
                  "records": records}
        pool = OutcomePool()
        pool.add(records)
        result["pooled"] = pool.metrics()
        if source is not None:
            result["source"] = source
        if animate:
            ledger = {"schema": 1, "seed": seed + episodes - 1,
                      "semantics": "Successful whole-report custody transfers after complete reassembly; destination num_uavs is GCS; power_w is last received fragment TX power; positions are at reception time.",
                      "num_uavs": config.world.num_uavs, "events": events}
            event_path = output / "communication_events.json"
            event_path.write_text(json.dumps(ledger, indent=2, allow_nan=False))
            result["communication_events_path"] = str(event_path)
            result["animation_path"] = render_animation(env, paths, frames, output / "final_episode", fps, video_format)
            result["animation"] = {"sampled_steps": [frame["step"] for frame in frames],
                                   "frame_stride": frame_stride, "fps": fps, "episode": episodes,
                                   "seed": seed + episodes - 1,
                                   "completed_report_hops": len(events),
                                   "communication_window_s": COMMUNICATION_WINDOW_S,
                                   "playback_speed": frame_stride * fps * config.world.dt_s}
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        return result
    finally:
        env.close()
