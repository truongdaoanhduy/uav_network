"""Paired timing driver: use the same script with PYTHONPATH at each source ref.

Repeat root seed44 for timing medians; these are not independent research seeds.
The pipeline probe is a short benchmark with an explicitly smaller warmup and
deadline, never a convergence result or a change to the training profiles.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from uav_network.configuration import load_config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.vector import random_actions
from uav_network.experiments.benchmark import benchmark
from uav_network.experiments.hardware import hardware_info
from uav_network.experiments.training import seed_runtime, train


def collector_probe(config, steps, repetitions):
    results = []
    for _ in range(repetitions):
        c = deepcopy(config)
        seed_runtime(c)
        env = SwarmSearchEnv(c)
        digest = hashlib.sha256()
        try:
            obs, _ = env.reset(seed=c.runtime.seed)
            rng = np.random.default_rng(c.runtime.seed + 1)
            started = time.perf_counter()
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(
                    random_actions(obs, rng, c.world.num_uavs + 1))
                digest.update(obs.tobytes())
                digest.update(env.state().tobytes())
                digest.update(json.dumps([reward, terminated, truncated, info], sort_keys=True).encode())
                if terminated or truncated:
                    obs, _ = env.reset(seed=c.runtime.seed)
            elapsed = time.perf_counter() - started
            events = env.network.simulator.event_bus.since(0)
            digest.update(json.dumps(events, sort_keys=True).encode())
            results.append({"steps": steps, "seconds": elapsed, "steps_per_s": steps / elapsed,
                            "trajectory_hash": digest.hexdigest(), "network": info["network"]})
        finally:
            env.close()
    return {"repeats": results, "median_seconds": statistics.median(x["seconds"] for x in results),
            "same_seed_repeats_match": len({x["trajectory_hash"] for x in results}) == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion-study.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coalesce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collector-only", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This paired benchmark explicitly requires real CUDA")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.runtime.device = config.runtime.replay_device = args.device
    # setattr also lets this driver run against the historical source revision.
    config.runtime.coalesce_idle_polls = args.coalesce
    config.runtime.fused_optimizers = args.fused
    config.tracking.enabled = False
    result = {"label": args.label, "hardware": hardware_info(), "root_seed": config.runtime.seed,
              "coalesce_idle_polls": args.coalesce, "fused_optimizers_requested": args.fused,
              "config": config.to_dict(), "collector": collector_probe(config, 160, args.repetitions)}
    if not args.collector_only:
        result["learners"] = {}
        for algorithm in ("masac", "matd3"):
            c = deepcopy(config)
            c.runtime.algorithm = algorithm
            trials = [benchmark(c, env_counts=[2], steps=32, learner_updates=50)["results"][0]
                      for _ in range(args.repetitions)]
            pipeline = deepcopy(c)
            pipeline.world.max_steps = 384
            pipeline.runtime.episodes = 2
            pipeline.runtime.max_train_seconds = 0
            pipeline.runtime.output_dir = str(output.parent / f"{args.label}-{algorithm}-pipeline")
            pipeline.algorithm_config.learning_starts = pipeline.algorithm_config.batch_size
            measured = train(pipeline)
            active_s = measured["elapsed_s"] - measured["startup_s"]
            measured["active_joint_transitions_per_s"] = measured["global_step"] / active_s
            result["learners"][algorithm] = {
                "trials": trials,
                "median_update_ms": 1000 * statistics.median(x["learner_s"] / x["learner_updates"] for x in trials),
                "median_collection_ms": 1000 * statistics.median(x["collector_s"] / x["vector_steps"] for x in trials),
                "pipeline_config": pipeline.to_dict(), "pipeline": measured}
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "label": args.label,
                      "collector_median_s": result["collector"]["median_seconds"],
                      "learners": {k: {"update_ms": v["median_update_ms"],
                                       "collection_ms": v["median_collection_ms"],
                                       "pipeline_transitions_per_s": v["pipeline"]["active_joint_transitions_per_s"]}
                                   for k, v in result.get("learners", {}).items()}}, indent=2))


if __name__ == "__main__":
    main()
