"""Evaluation-only actors: immutable on disk, no parent learner construction or RNG."""
from contextlib import suppress
from pathlib import Path

import torch

from ..configuration import Config, _apply
from ..environment.observations import OBSERVATION_VERSION, observation_version, observation_dim, state_dim
from ..algorithms.checkpoints import make_learner, validate_observation_version


def save_evaluation_snapshot(config, folder, actors, version, progress):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"eval_step_{progress['global_step']:012d}_policy_{version}"
    path = folder / f"{stem}.pt"
    serial = 0
    while path.exists():
        serial += 1
        path = folder / f"{stem}_{serial}.pt"
    # PQL callers pass the immutable mailbox value, never its mutable live module.
    frozen = {name: value.detach().cpu().clone() for name, value in actors.items()}
    if not frozen or not all(torch.isfinite(value).all() for value in frozen.values()):
        raise ValueError("Evaluation actors must be finite and nonempty")
    saved = {"version": 1, "purpose": "evaluation_only", "algorithm": config.runtime.algorithm,
             "config": config.to_dict(), "observation_dim": observation_dim(config),
             "state_dim": state_dim(config), "observation_version": observation_version(config), "actors": frozen,
             "progress": {key: progress[key] for key in
                          ("episodes", "global_step", "collection_steps", "next_seed") if key in progress},
             "source": {"policy_version": int(version), "global_step": int(progress["global_step"]),
                        "episodes": int(progress["episodes"]),
                        "policy_version_unit": "completed_actor_updates"}}
    temporary = path.with_suffix(".pt.tmp")
    try:
        torch.save(saved, temporary)
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return path


def load_evaluation_snapshot(saved):
    """Called in the CPU evaluator process; critics are fresh and unused for inference."""
    if saved.get("purpose") != "evaluation_only" or saved.get("version") != 1:
        raise ValueError("Unsupported evaluation snapshot")
    validate_observation_version(saved)
    config = Config()
    _apply(config, saved["config"])
    config.validate()
    if (saved["algorithm"] != config.runtime.algorithm or
            saved["observation_dim"] != observation_dim(config) or saved["state_dim"] != state_dim(config)):
        raise ValueError("Evaluation snapshot configuration/dimensions differ")
    learner = make_learner(config, saved["observation_dim"], saved["state_dim"], device="cpu")
    learner.actors.load_state_dict(saved["actors"], strict=True)
    learner.actor_updates = saved["source"]["policy_version"]
    learner.actors.eval()
    return learner
