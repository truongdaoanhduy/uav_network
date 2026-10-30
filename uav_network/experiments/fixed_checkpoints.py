"""Complete fixed-schedule learner state with optional in-flight mission state."""
from ..configuration import with_legacy_feature_defaults
import random

import numpy as np
import torch

from ..algorithms.checkpoints import save_policy, validate_observation_version
from ..algorithms.precision import validate_precision_state
from ..algorithms.replay import transition_shapes
from .pql_workers import optimizer_states, restore_optimizers


def replay_state(replay, *, copy=True):
    """Freeze by default; borrow full CPU storage only during synchronous save."""
    from ..algorithms.hybrid_replay import HybridMeasuredReplay
    if isinstance(replay, HybridMeasuredReplay):
        return replay.state_dict(copy=copy)
    def snapshot(value):
        result = value.detach().cpu()
        # Compact partial views so torch.save never writes unused ring storage.
        if copy or result.untyped_storage().nbytes() != result.numel() * result.element_size():
            result = result.clone()
        return result

    result = {"capacity": replay.capacity, "index": replay.index, "size": replay.size,
              "device_type": replay.device.type, "rng": replay.rng.get_state(),
              "data": {name: snapshot(value[:replay.size]) for name, value in replay.data.items()}}
    if hasattr(replay, "measurement_state"):
        result["measurement"] = {key: snapshot(value) if isinstance(value, torch.Tensor) else value
                                 for key, value in replay.measurement_state().items()}
    return result


def restore_replay(replay, saved):
    from ..algorithms.hybrid_replay import HybridMeasuredReplay
    if isinstance(replay, HybridMeasuredReplay):
        return replay.load_state_dict(saved)
    if (saved["capacity"] != replay.capacity or not 0 <= saved["size"] <= replay.capacity
            or not 0 <= saved["index"] < replay.capacity):
        raise ValueError("Fixed-schedule replay checkpoint capacity/size/cursor mismatch")
    if saved["device_type"] != replay.device.type:
        raise ValueError("Fixed-schedule replay RNG resume requires the original device type")
    if saved["data"].keys() != replay.data.keys():
        raise ValueError("Fixed-schedule replay checkpoint transition fields mismatch")
    for name, value in replay.data.items():
        incoming = saved["data"][name]
        if incoming.shape != value[:saved["size"]].shape or incoming.dtype != value.dtype:
            raise ValueError("Fixed-schedule replay checkpoint transition shape/dtype mismatch")
        value[:saved["size"]].copy_(incoming)
    replay.rng.set_state(saved["rng"].cpu())
    replay.index, replay.size = saved["index"], saved["size"]
    if hasattr(replay, "load_measurement_state") and "measurement" in saved:
        replay.load_measurement_state(saved["measurement"])


def runtime_rng_state(rng, *, action_generator=None):
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "collector": rng.bit_generator.state,
            "collector_torch": action_generator.get_state() if action_generator is not None else None}


def restore_runtime_rng(saved, rng, *, action_generator=None):
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"]:
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Fixed-schedule learner RNG resume requires the original CUDA device count")
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda"]])
    random.setstate(saved["python"])
    numpy_state = saved["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    rng.bit_generator.state = saved["collector"]
    if (saved["collector_torch"] is None) != (action_generator is None):
        raise ValueError("Fixed-schedule collector RNG checkpoint does not match its environment backend")
    if action_generator is not None:
        action_generator.set_state(saved["collector_torch"].cpu())


def validate_fixed_checkpoint(saved, config):
    """Validate v3 payload for fixed-schedule continuation; return verified progress."""
    from .continuation import validate_progress
    if saved.get("version") != 3 or not isinstance(saved.get("training"), dict):
        raise ValueError("Fixed-schedule resume requires a complete version-3 training checkpoint")
    validate_observation_version(saved)
    training = saved["training"]
    required = {"trainer", "optimizers", "replay", "rng", "collections", "warmup_origin",
                "random_warmup", "next_metrics_at", "latest", "last_actor_metrics", "learner_device_type", "pending_learning"}
    if not required <= training.keys() or training["trainer"] != "fixed_schedule":
        raise ValueError("Fixed-schedule checkpoint is missing optimizer/replay/RNG/schedule state")
    if saved.get("algorithm") != config.runtime.algorithm:
        raise ValueError("Fixed-schedule resume algorithm differs from checkpoint")
    before, current = with_legacy_feature_defaults(saved["config"]), config.to_dict()
    if before["runtime"].get("replay_cpu_fraction", 0.0) != config.runtime.replay_cpu_fraction:
        raise ValueError("Fixed-schedule resume runtime.replay_cpu_fraction differs from checkpoint")
    if before["pql"]["enabled"] or config.pql.enabled:
        raise ValueError("Fixed-schedule resume cannot change PQL scheduling")
    for group in ("world", "sensor", "network", "energy", "reward", config.runtime.algorithm):
        if before[group] != current[group]:
            raise ValueError(f"Fixed-schedule resume {group} differs from checkpoint")
    for name in ("seed", "num_envs", "environment_backend", "collector_overlap", "deterministic",
                 "vectorize_critics", "compile_models", "fused_optimizers", "torch_threads", "amp_mode"):
        if before["runtime"].get(name, "off" if name == "amp_mode" else None) != current["runtime"][name]:
            raise ValueError(f"Fixed-schedule resume runtime.{name} differs from checkpoint")
    groups = ("critic", "actor", "alpha") if config.runtime.algorithm == "masac" else ("critic", "actor")
    validate_precision_state(training.get("precision"), config.runtime.amp_mode, groups)
    progress = validate_progress(saved.get("progress"), config.runtime.episodes)
    for name in ("updates", "actor_updates"):
        if type(saved.get(name)) is not int or saved[name] < 0:
            raise ValueError(f"Invalid fixed-schedule checkpoint counter: {name}")
    for name in ("collections", "warmup_origin", "next_metrics_at"):
        if type(training[name]) is not int or training[name] < 0:
            raise ValueError(f"Invalid fixed-schedule checkpoint schedule: {name}")
    if (training["collections"] != progress["collection_steps"]
            or progress["global_step"] != config.runtime.num_envs * training["collections"]
            or training["warmup_origin"] > progress["global_step"]):
        raise ValueError("Fixed-schedule checkpoint counters differ from verified progress")
    delay = config.matd3.policy_delay if config.runtime.algorithm == "matd3" else 1
    if saved["actor_updates"] != saved["updates"] // delay:
        raise ValueError("Fixed-schedule checkpoint does not preserve the policy delay clock")
    if (type(training["random_warmup"]) is not bool or type(training["pending_learning"]) is not bool
            or not isinstance(training["latest"], dict)
            or not isinstance(training["last_actor_metrics"], dict)):
        raise ValueError("Invalid fixed-schedule checkpoint warmup/metric schedule")
    optimizers = training["optimizers"]
    names = {"critic", "actors", "alpha"} if saved["algorithm"] == "masac" else {"critic", "actors"}
    if (not isinstance(optimizers, dict) or optimizers.keys() != names
            or not isinstance(optimizers["actors"], list)
            or len(optimizers["actors"]) != config.world.num_uavs):
        raise ValueError("Fixed-schedule checkpoint optimizer state is incomplete")
    for state in [optimizers["critic"], *optimizers["actors"], *([optimizers["alpha"]] if "alpha" in names else [])]:
        if not isinstance(state, dict) or not {"state", "param_groups"} <= state.keys():
            raise ValueError("Fixed-schedule checkpoint Adam state is incomplete")
        for parameter in state["state"].values():
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                   for value in parameter.values()):
                raise ValueError("Fixed-schedule checkpoint contains non-finite Adam tensors")
    replay = training["replay"]
    hybrid = config.runtime.replay_cpu_fraction > 0
    replay_required = {"capacity", "size", "index", "rng"} | (
        {"layout", "cpu_data", "hot_data"} if hybrid else {"device_type", "data"})
    if not isinstance(replay, dict) or not replay_required <= replay.keys():
        raise ValueError("Fixed-schedule checkpoint requires complete replay state")
    inserted = progress["global_step"] - training["warmup_origin"]
    capacity = config.algorithm_config.replay_capacity
    if (replay["capacity"] != capacity or type(replay["size"]) is not int
            or replay["size"] != min(capacity, inserted) or type(replay["index"]) is not int
            or replay["index"] != inserted % capacity):
        raise ValueError("Fixed-schedule checkpoint replay cursor differs from verified progress")
    shapes = transition_shapes((config.world.num_uavs, saved["observation_dim"]), saved["state_dim"], config.continuous_dim)
    if hybrid:
        from ..algorithms.hybrid_replay import validate_hybrid_state
        validate_hybrid_state(replay, capacity, (config.world.num_uavs, saved["observation_dim"]),
                              saved["state_dim"], config.runtime.replay_cpu_fraction, continuous_dim=config.continuous_dim)
    elif not isinstance(replay["data"], dict) or replay["data"].keys() != shapes.keys():
        raise ValueError("Fixed-schedule checkpoint replay fields are incomplete")
    for name, shape in (() if hybrid else shapes.items()):
        value = replay["data"][name]
        dtype = torch.int64 if name == "destination" else torch.float32
        if (not isinstance(value, torch.Tensor) or value.shape != (replay["size"], *shape)
                or value.dtype != dtype or not torch.isfinite(value).all()):
            raise ValueError("Fixed-schedule checkpoint replay transitions are invalid")
    rng = training["rng"]
    if not isinstance(rng, dict) or not {"torch", "cuda", "python", "numpy", "collector", "collector_torch"} <= rng.keys():
        raise ValueError("Fixed-schedule checkpoint RNG state is incomplete")
    if (rng["collector_torch"] is None) != (config.runtime.environment_backend != "warp_packet"):
        raise ValueError("Fixed-schedule checkpoint collector RNG differs from environment backend")
    if not all(torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("Fixed-schedule checkpoint contains non-finite model tensors")
    if "environment" in training:
        from .environment_checkpoints import validate_environment
        validate_environment(training["environment"], config, progress)
    return progress


def restore_fixed_learner(saved, learner):
    training = saved["training"]
    if training["learner_device_type"] != learner.device.type:
        raise ValueError("Fixed-schedule learner RNG resume requires the original device type")
    restore_optimizers(learner, learner, training["optimizers"])
    if training.get("precision") is not None:
        learner.precision.load_state_dict(training["precision"])
    if learner.algorithm == "matd3":
        learner._last_actor_metrics = {name: value.to(learner.device).clone() for name, value in training["last_actor_metrics"].items()}


def save_fixed_checkpoint(learner, path, progress, replay, rng, *, warmup_origin, random_warmup,
                          next_metrics_at, latest, pending_learning, action_generator=None, environment=None, selection_state=None):
    training = {"trainer": "fixed_schedule", "optimizers": optimizer_states(learner, learner),
                "precision": learner.precision.state_dict(),
                "replay": replay_state(replay, copy=False), "rng": runtime_rng_state(rng, action_generator=action_generator),
                "collections": progress["collection_steps"], "warmup_origin": warmup_origin,
                "random_warmup": random_warmup, "pending_learning": pending_learning, "next_metrics_at": next_metrics_at,
                "latest": latest, "last_actor_metrics": getattr(learner, "_last_actor_metrics", {}),
                "learner_device_type": learner.device.type}
    if environment is not None:
        training["environment"] = environment
    if selection_state is not None:
        training["best_checkpoint"] = selection_state
    save_policy(learner, path, progress, training=training)
