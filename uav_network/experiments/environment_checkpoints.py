"""In-flight mission and collector state, optional for legacy v3 checkpoints."""
from ..configuration import with_legacy_feature_defaults
import numpy as np
import torch


def snapshot_environment(vector, returns, lengths, seeds, next_seed):
    if not hasattr(vector, "snapshot_state"):
        return None
    return {"version": 1, "world": vector.snapshot_state(),
            "returns": torch.as_tensor(returns).detach().cpu().clone(),
            "lengths": torch.as_tensor(lengths).detach().cpu().clone(),
            "seeds": torch.as_tensor(seeds).detach().cpu().clone(), "next_seed": int(next_seed)}


def validate_environment(saved, config, progress):
    """Validate collector counters before restoring any simulator arrays."""
    count = config.runtime.num_envs
    if (not isinstance(saved, dict) or saved.get("version") != 1
            or config.runtime.environment_backend != "warp_packet"):
        raise ValueError("Invalid environment checkpoint version/backend")
    for name, dtype in (("returns", torch.float64), ("lengths", torch.int64), ("seeds", torch.int64)):
        value = saved.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (count,)
                or value.dtype != dtype or value.device.type != "cpu"):
            raise ValueError(f"Invalid environment collector {name}")
    world = saved.get("world")
    current = config.to_dict()
    expected = {key: current[key] for key in ("world", "sensor", "network", "energy", "reward")} | {
        "num_envs": count, "backend": "warp_packet"}
    if (not isinstance(world, dict) or world.get("version") != 1 or with_legacy_feature_defaults(world.get("config", {})) != expected
            or not isinstance(world.get("arrays"), dict)):
        raise ValueError("Invalid environment world/config")
    arrays = world["arrays"]
    if (not torch.isfinite(saved["returns"]).all() or bool((saved["lengths"] < 0).any())
            or bool((saved["lengths"] > progress["collection_steps"]).any())
            or type(saved.get("next_seed")) is not int or saved["next_seed"] != progress["next_seed"]
            or saved["next_seed"] <= int(saved["seeds"].max())
            or not torch.equal(saved["seeds"], arrays.get("seeds", torch.empty(0)))
            or not torch.equal(saved["lengths"], arrays.get("steps", torch.empty(0)).to(torch.int64))):
        raise ValueError("Environment collector counters/seeds differ from world/progress")


def initialize_environment(vector, initial, saved=None):
    """Restore retained worlds; only already terminated worlds receive new seeds."""
    count = vector.num_envs
    device_env = hasattr(vector, "snapshot_state")
    device = vector.device if device_env else "cpu"
    if saved is not None:
        validate_environment(saved, vector.config, initial)
        obs = vector.restore_state(saved["world"])
        returns = saved["returns"].to(device).clone()
        lengths = saved["lengths"].to(device).clone()
        seeds = saved["seeds"].numpy().copy()
        next_seed = saved["next_seed"]
        done = saved["world"]["arrays"]["terminated"].bool().numpy()
        if done.any():
            replacements = [None] * count
            for i in np.flatnonzero(done):
                replacements[i] = seeds[i] = next_seed
                next_seed += 1
                returns[i] = lengths[i] = 0
            obs, _ = vector.reset(seed=replacements, options={"reset_mask": done})
        return obs, returns, lengths, seeds, next_seed
    seeds = np.arange(count) + initial["next_seed"]
    obs, _ = vector.reset(seed=seeds.tolist())
    returns, lengths = (torch.zeros(count, device=device, dtype=torch.float64),
                        torch.zeros(count, device=device, dtype=torch.int64)) if device_env else (
                            np.zeros(count), np.zeros(count, dtype=np.int64))
    return obs, returns, lengths, seeds, initial["next_seed"] + count
