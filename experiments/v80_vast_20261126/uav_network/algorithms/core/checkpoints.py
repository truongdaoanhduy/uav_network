"""Checkpoint version 4: algorithm-specific, no previous UAV world config."""
from contextlib import suppress
from pathlib import Path
import torch
from uav_network.algorithms.configuration.schema import LearnerConfig

CHECKPOINT_VERSION=4
SCENARIO="zhang-xue-7radio-v60"

def make_learner(config, observation_dim, state_dim, device="cpu"):
    from uav_network.algorithms.matd3.learner import MATD3
    from uav_network.algorithms.masac.learner import MASAC
    return {"matd3":MATD3,"masac":MASAC}[config.runtime.algorithm](
        config, observation_dim, state_dim, device
    )

def validate_observation_version(saved):
    if saved.get("version") != CHECKPOINT_VERSION or saved.get("scenario_id")!=SCENARIO:
        raise ValueError("Historical checkpoint belongs to a different scenario: restore its pinned Git version")

def save_policy(learner, path, progress=None, *, training=None):
    floating=[v for v in learner.state_dict().values() if v.is_floating_point()]
    if any(not torch.isfinite(v).all() for v in floating):
        raise FloatingPointError("Cannot save nonfinite model weights")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    state={
        "version":CHECKPOINT_VERSION,
        "scenario_id":SCENARIO,
        "algorithm":learner.algorithm,
        "config":learner.config.to_dict(),
        "observation_dim":learner.observation_dim,
        "state_dim":learner.state_dim,
        "model":learner.state_dict(),
        "updates":learner.updates,
        "actor_updates":learner.actor_updates
    }
    if progress is not None:state["progress"]=progress
    if training is not None:state["training"]=training
    try:
        torch.save(state,temp)
        temp.replace(path)
    finally:
        with suppress(OSError):temp.unlink(missing_ok=True)

def load_policy(path, device="cpu", expected_algorithm=None, *, payload=None):
    saved=payload if payload is not None else torch.load(path,map_location=device,weights_only=True)
    validate_observation_version(saved)
    algorithm=saved.get("algorithm")
    if algorithm not in ("matd3","masac") or (expected_algorithm and algorithm!=expected_algorithm):
        raise ValueError(f"Checkpoint algorithm mismatch: {algorithm!r}, expected {expected_algorithm!r}")
    config=LearnerConfig.from_dict(saved["config"])
    config.runtime.algorithm=algorithm
    learner=make_learner(config,saved["observation_dim"],saved["state_dim"],device)
    learner.load_state_dict(saved["model"])
    learner.updates=saved["updates"]
    learner.actor_updates=saved.get("actor_updates",0)
    return learner
