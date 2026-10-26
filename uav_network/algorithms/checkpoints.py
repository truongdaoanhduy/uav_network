"""Versioned policies and optional progress; Adam/replay are not serialized."""
from contextlib import suppress
from pathlib import Path
import torch

from ..configuration import Config, WorldConfig, _apply
from ..environment.observations import OBSERVATION_VERSION, observation_dim, state_dim


def make_learner(config, obs_dim, critic_dim, device="cpu"):
    from .masac import MASAC
    from .matd3 import MATD3
    cls = {"masac": MASAC, "matd3": MATD3}[config.runtime.algorithm]
    return cls(config, obs_dim, critic_dim, device)


def validate_observation_version(saved):
    """Guard observation and seeded scene-generator semantics at every load path."""
    version = saved.get("observation_version")
    if type(version) is not int or version != OBSERVATION_VERSION:
        raise ValueError("Checkpoint observation semantics differ; use its pinned historical source "
                         "or retrain from episode 0")
    generation = saved.get("config", {}).get("world", {}).get("generation_version")
    if type(generation) is not int or generation != WorldConfig().generation_version:
        raise ValueError("Checkpoint scenario generation semantics differ; use its pinned historical source "
                         "or retrain from episode 0")


def save_policy(learner, path, progress=None, *, training=None):
    checks = [torch.isfinite(value).all() for value in learner.state_dict().values() if value.is_floating_point()]
    if not torch.stack(checks).all():
        raise FloatingPointError("Cannot save a policy with non-finite parameters")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    saved = {"version": 3 if training is not None else 2, "algorithm": learner.algorithm,
                "config": learner.config.to_dict(), "observation_dim": learner.observation_dim,
                "state_dim": learner.state_dim, "observation_version": OBSERVATION_VERSION,
                "model": learner.state_dict(),
                "updates": learner.updates}
    if training is not None:
        saved["training"] = training
        saved["actor_updates"] = learner.actor_updates
    if progress is not None:
        saved["progress"] = progress
    try:
        torch.save(saved, temporary)
        temporary.replace(path)
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def load_policy(path, device="cpu", expected_algorithm=None, *, payload=None):
    saved = payload if payload is not None else torch.load(path, map_location=device, weights_only=True)
    if saved.get("version") not in {2, 3}:
        raise ValueError("Battery/mask observations require a version-2 checkpoint; retrain old policies")
    validate_observation_version(saved)
    algorithm = saved.get("algorithm")
    if algorithm not in {"masac", "matd3"} or (expected_algorithm and algorithm != expected_algorithm):
        raise ValueError(f"Checkpoint algorithm {algorithm!r} does not match {expected_algorithm!r}")
    config = Config()
    _apply(config, saved["config"])
    # Legacy v2 policies must keep their original range-only mask when evaluated.
    if "receiver_admission_mask" not in saved["config"].get("network", {}):
        config.network.receiver_admission_mask = False
    if "motion_safety" not in saved["config"].get("world", {}):
        config.world.motion_safety = "reject"
    if "rediscovery_recovery" not in saved["config"].get("network", {}):
        config.network.rediscovery_recovery = False
    if "ack_uses_data_power" not in saved["config"].get("network", {}):
        config.network.ack_uses_data_power = False
    config.runtime.algorithm = algorithm
    config.validate()
    if saved["observation_dim"] != observation_dim(config) or saved["state_dim"] != state_dim(config):
        raise ValueError("Checkpoint observation/state dimensions do not match its configuration")
    learner = make_learner(config, saved["observation_dim"], saved["state_dim"], device)
    learner.load_state_dict(saved["model"])
    learner.updates = saved["updates"]
    learner.actor_updates = saved.get("actor_updates", learner.updates if algorithm == "masac"
                                              else learner.updates // config.matd3.policy_delay)
    return learner
