"""Restore policy/targets and real progress; rebuild Adam, replay and simulators."""
from copy import deepcopy
import torch

from ..algorithms.checkpoints import load_policy, make_learner
from ..environment.observations import observation_dim, state_dim


def restore_learner(config, device):
    if config.runtime.warm_start_checkpoint == "none":
        return make_learner(config, observation_dim(config), state_dim(config), device), {
            "episodes": 0, "global_step": 0, "collection_steps": 0,
            "next_seed": config.runtime.seed, "records": [],
        }
    path = config.runtime.warm_start_checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True)
    progress = saved.get("progress")
    required = {"episodes", "global_step", "collection_steps", "next_seed", "records"}
    if not isinstance(progress, dict) or not required <= progress.keys():
        raise ValueError("Checkpoint needs verified progress metadata for warm-start")
    for key in required - {"records"}:
        if type(progress[key]) is not int or progress[key] < 0:
            raise ValueError(f"Invalid checkpoint progress: {key}")
    records = progress["records"]
    if (not isinstance(records, list) or len(records) != progress["episodes"]
            or any(not isinstance(row, dict) or row.get("episode") != i + 1
                   for i, row in enumerate(records))):
        raise ValueError("Checkpoint progress must contain the real completed episode prefix")
    if records and (progress["global_step"] < max(row["global_step"] for row in records)
                    or progress["next_seed"] <= max(row["seed"] for row in records)):
        raise ValueError("Checkpoint progress counters/seeds precede its history")
    if progress["episodes"] >= config.runtime.episodes:
        raise ValueError("Requested episode budget must exceed the checkpoint's completed episodes")
    learner = load_policy(path, device, expected_algorithm=config.runtime.algorithm)
    before, current = learner.config.to_dict(), config.to_dict()
    for group in ("world", "sensor", "network", "energy", "reward", config.runtime.algorithm):
        if before[group] != current[group]:
            raise ValueError(f"Warm-start {group} differs from the checkpoint")
    if learner.config.runtime.environment_backend != config.runtime.environment_backend:
        raise ValueError("Warm-start environment_backend must match the checkpoint")
    if learner.config.runtime.seed != config.runtime.seed:
        raise ValueError("Warm-start runtime.seed must match the checkpoint")
    if type(learner.updates) is not int or learner.updates < 0:
        raise ValueError("Invalid checkpoint update counter")
    if not all(torch.isfinite(p).all() for p in learner.state_dict().values()):
        raise ValueError("Warm-start checkpoint contains non-finite model tensors")
    learner.config = deepcopy(config)
    return learner, progress
