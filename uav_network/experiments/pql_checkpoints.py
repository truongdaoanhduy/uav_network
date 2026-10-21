"""Reject inconsistent PQL resume metadata before allocating replay or workers."""
import math

import torch

from .continuation import validate_progress
from ..algorithms.precision import validate_precision_state


def validate_pql_checkpoint(saved, config):
    if saved.get("version") != 3 or not isinstance(saved.get("training"), dict):
        raise ValueError("PQL resume requires a complete version-3 training checkpoint")
    progress = validate_progress(saved.get("progress"), config.runtime.episodes)
    training = saved["training"]
    before, current = saved["config"]["runtime"], config.to_dict()["runtime"]
    for name in ("amp_mode", "compile_models", "fused_optimizers", "vectorize_critics", "deterministic"):
        if before.get(name, "off" if name == "amp_mode" else None) != current[name]:
            raise ValueError(f"PQL resume runtime.{name} differs from checkpoint")
    precision = training.get("precision")
    if precision is not None and (not isinstance(precision, dict) or precision.keys() != {"critic", "policy"}):
        raise ValueError("PQL precision checkpoint must preserve both worker states")
    groups = ("critic", "actor", "alpha") if config.runtime.algorithm == "masac" else ("critic", "actor")
    for worker in ("critic", "policy"):
        validate_precision_state(precision.get(worker) if precision is not None else None,
                                 config.runtime.amp_mode, groups)

    required = {"collections", "eligible_collections", "credit", "warmup_origin", "optimizers",
                "collector_policy_version", "critic_policy_version", "policy_critic_version", "rng", "replay"}
    if not required <= training.keys():
        raise ValueError("PQL checkpoint is missing controller/optimizer/RNG state")
    for name in ("collections", "eligible_collections", "warmup_origin",
                 "collector_policy_version", "critic_policy_version", "policy_critic_version"):
        if type(training[name]) is not int or training[name] < 0:
            raise ValueError(f"Invalid PQL checkpoint counter: {name}")
    for name in ("updates", "actor_updates"):
        if type(saved.get(name)) is not int or saved[name] < 0:
            raise ValueError(f"Invalid PQL checkpoint update counter: {name}")
    if (training["collections"] != progress["collection_steps"]
            or training["eligible_collections"] > training["collections"]
            or training["warmup_origin"] > training["collections"]):
        raise ValueError("PQL controller collections differ from verified progress")
    original_count = saved["config"]["runtime"]["num_envs"]
    if config.runtime.num_envs != original_count:
        raise ValueError("PQL resume num_envs differs from checkpoint")
    if progress["global_step"] != original_count * progress["collection_steps"]:
        raise ValueError("PQL checkpoint joint-transition count is inconsistent")
    credit = training["credit"]
    if type(credit) not in (float, int) or not math.isfinite(credit) or not 0 <= credit < 1:
        raise ValueError("PQL checkpoint credit must be finite and fractional")
    rate = saved["config"]["pql"]["critic_sample_ratio"]
    expected = training["eligible_collections"] * rate
    if (saved["updates"] != math.floor(expected + 1e-8)
            or not math.isclose(credit, max(0., expected - saved["updates"]), rel_tol=0, abs_tol=1e-8)):
        raise ValueError("PQL checkpoint has unearned critic update credit")
    delay = saved["config"]["matd3"]["policy_delay"] if saved["algorithm"] == "matd3" else 1
    if saved["actor_updates"] != saved["updates"] // delay:
        raise ValueError("PQL checkpoint does not preserve the policy delay clock")
    if (max(training["collector_policy_version"], training["critic_policy_version"]) > saved["actor_updates"]
            or training["policy_critic_version"] > saved["updates"]):
        raise ValueError("PQL checkpoint snapshot version exceeds completed updates")
    if not all(torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("PQL checkpoint contains non-finite model tensors")
    return progress
