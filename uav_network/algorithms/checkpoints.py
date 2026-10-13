"""Versioned policies and optional progress; Adam/replay are not serialized."""
from pathlib import Path
import torch

from ..configuration import Config, _apply
from ..environment.observations import observation_dim, state_dim


def make_learner(config, obs_dim, critic_dim, device="cpu"):
    from .masac import MASAC
    from .matd3 import MATD3
    cls = {"masac": MASAC, "matd3": MATD3}[config.runtime.algorithm]
    return cls(config, obs_dim, critic_dim, device)


def save_policy(learner, path, progress=None):
    checks = [torch.isfinite(value).all() for value in learner.state_dict().values() if value.is_floating_point()]
    if not torch.stack(checks).all():
        raise FloatingPointError("Cannot save a policy with non-finite parameters")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    saved = {"version": 2, "algorithm": learner.algorithm,
                "config": learner.config.to_dict(), "observation_dim": learner.observation_dim,
                "state_dim": learner.state_dim, "model": learner.state_dict(),
                "updates": learner.updates}
    if progress is not None:
        saved["progress"] = progress
    torch.save(saved, temporary)
    temporary.replace(path)


def load_policy(path, device="cpu", expected_algorithm=None):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("version") != 2:
        raise ValueError("Battery/mask observations require a version-2 checkpoint; retrain old policies")
    algorithm = saved.get("algorithm")
    if algorithm not in {"masac", "matd3"} or (expected_algorithm and algorithm != expected_algorithm):
        raise ValueError(f"Checkpoint algorithm {algorithm!r} does not match {expected_algorithm!r}")
    config = Config()
    _apply(config, saved["config"])
    config.runtime.algorithm = algorithm
    config.validate()
    if saved["observation_dim"] != observation_dim(config) or saved["state_dim"] != state_dim(config):
        raise ValueError("Checkpoint observation/state dimensions do not match its configuration")
    learner = make_learner(config, saved["observation_dim"], saved["state_dim"], device)
    learner.load_state_dict(saved["model"])
    learner.updates = saved["updates"]
    return learner
