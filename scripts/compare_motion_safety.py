"""Compare one checkpoint on one map seed; no multiseed averaging or retraining."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from uav_network.algorithms.checkpoints import load_policy
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.geometry import segment_distance
from uav_network.experiments.rendering import render_episode
from uav_network.experiments.training import seed_runtime


def compare(checkpoint, output, seed=10044, receiver_admission=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    learner = load_policy(checkpoint, device="cpu")
    seed_runtime(learner.config)
    records = []
    for mode, repeat in (("reject", 1), ("dynamic_window", 1), ("dynamic_window", 2)):
        config = deepcopy(learner.config)
        config.world.motion_safety = mode
        if receiver_admission is not None:
            config.network.receiver_admission_mask = receiver_admission
        env = SwarmSearchEnv(config)
        paths, episode_return, maximum_accel, violations = [], 0., 0., 0
        began = time.perf_counter()
        try:
            obs, _ = env.reset(seed=seed)
            paths.append(env.positions.copy())
            while True:
                previous_position, previous_velocity = env.positions.copy(), env.velocities.copy()
                obs, reward, terminated, truncated, info = env.step(learner.act(obs, deterministic=True))
                episode_return += reward
                paths.append(env.positions.copy())
                maximum_accel = max(maximum_accel, float(np.abs(
                    env.velocities - previous_velocity).max() / config.world.dt_s))
                for i in range(config.world.num_uavs):
                    for obstacle in env.obstacles:
                        violations += int(segment_distance(previous_position[i], env.positions[i],
                            obstacle[:2]) < obstacle[2] + config.world.obstacle_clearance_m - 1e-7)
                    for j in range(i):
                        violations += int(segment_distance(previous_position[i] - previous_position[j],
                            env.positions[i] - env.positions[j], [0, 0])
                            < config.world.safety_distance_m - 1e-7)
                if env.steps % 200 == 0:
                    print(json.dumps({"mode": mode, "repeat": repeat, "step": env.steps,
                        "search": info["search_rate_percent"], "delivery": info["delivery_rate_percent"]}),
                        flush=True)
                if terminated or truncated:
                    break
            scientific = {"seed": seed, "return": episode_return, **info,
                "final_positions": env.positions.tolist(),
                "undelivered_target_ids": np.flatnonzero(~env.reports.delivered).tolist(),
                "held_reports": {str(i): list(buffer) for i, buffer in enumerate(env.reports.buffers)},
                "gcs_distances_m": np.linalg.norm(env.positions - env.gcs, axis=1).tolist()}
            record = {"mode": mode, "repeat": repeat, "elapsed_s": time.perf_counter() - began,
                      "maximum_realized_axis_acceleration": maximum_accel,
                      "swept_clearance_violations": violations,
                      "episode_hash": sha256(json.dumps(scientific, sort_keys=True,
                          allow_nan=False).encode()).hexdigest(),
                      "trajectory_hash": sha256(np.asarray(paths).tobytes()).hexdigest(), **scientific}
            render_episode(env, paths, output / f"{mode}-{repeat}.png")
            records.append(record)
            print(json.dumps({k: record[k] for k in ("mode", "repeat", "return", "search_rate_percent",
                "delivery_rate_percent", "success", "blocked_moves", "boundary_clips",
                "maximum_realized_axis_acceleration", "swept_clearance_violations", "motion_counts")}),
                flush=True)
        finally:
            env.close()
    repeat_match = all(records[1][k] == records[2][k] for k in ("episode_hash", "trajectory_hash"))
    result = {"checkpoint": str(checkpoint), "algorithm": learner.algorithm,
              "training_seed": learner.config.runtime.seed, "evaluation_seed": seed,
              "receiver_admission_mask": config.network.receiver_admission_mask,
              "scope": "Single frozen policy, single evaluation map, change only motion mode",
              "repeat_exact": repeat_match, "records": records}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if not repeat_match:
        raise AssertionError("Identical evaluation was not exactly repeatable")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=10044)
    parser.add_argument("--receiver-admission-mask", choices=("checkpoint", "true", "false"),
                        default="checkpoint")
    args = parser.parse_args()
    override = None if args.receiver_admission_mask == "checkpoint" else args.receiver_admission_mask == "true"
    compare(args.checkpoint, args.output, args.seed, override)
