"""Restore verified policies/progress and optional full learner state."""
from copy import deepcopy
import torch

from ..algorithms.checkpoints import load_policy, make_learner
from ..environment.observations import observation_dim, state_dim


def validate_progress(progress, episode_budget=None):
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
    if episode_budget is not None and progress["episodes"] >= episode_budget:
        raise ValueError("Requested episode budget must exceed the checkpoint's completed episodes")
    return progress


def restore_learner(config, device):
    if config.runtime.warm_start_checkpoint == "none":
        return make_learner(config, observation_dim(config), state_dim(config), device), {
            "episodes": 0, "global_step": 0, "collection_steps": 0,
            "next_seed": config.runtime.seed, "records": [],
        }
    path = config.runtime.warm_start_checkpoint
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    progress = saved.get("progress")
    if saved.get("version") == 3:
        from .fixed_checkpoints import validate_fixed_checkpoint
        validate_fixed_checkpoint(saved, config)
    else:
        validate_progress(progress, config.runtime.episodes)
    learner = load_policy(path, device, expected_algorithm=config.runtime.algorithm, payload=saved)
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
    progress = dict(progress)
    if saved.get("version") == 3:
        from .fixed_checkpoints import restore_fixed_learner
        restore_fixed_learner(saved, learner)
        progress["_training"] = saved["training"]
        progress["state_restore"] = "training_state_environment_reset"
    else:
        # A policy-only checkpoint supplies weights and counters, not execution
        # or optimizer state. Build these from the requested runtime profile.
        previous = learner
        learner = make_learner(config, observation_dim(config), state_dim(config), device)
        learner.load_state_dict(previous.state_dict())
        learner.updates, learner.actor_updates = previous.updates, previous.actor_updates
        progress["state_restore"] = "policy_warm_start"
    return learner, progress
