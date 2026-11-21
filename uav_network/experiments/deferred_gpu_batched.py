"""Batched distinct MATD3 shared Actor checkpoints over Warp CUDA vector worlds.

Uses torch.func.functional_call/vmap to evaluate K independently trained
Actor weight sets in one GPU forward pass. Each Actor controls 5 independent
fixed-seed worlds; K*5 worlds advance together in one Warp launch. No Critics,
replay or optimizers are instantiated. Strictly opt-in, with sequential
per-checkpoint fallback in the parent evaluator.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from ..algorithms.action_utils import canonical_continuous, valid_destinations
from ..algorithms.models import DeterministicActor, HybridActor
from ..configuration import Config, _apply, config_fingerprint
from ..environment.observations import observation_dim
from ..environment.gpu.vector import WarpVectorEnv
from .gpu_mission_metrics import gpu_episode_record


class BatchedSharedActor:
    def __init__(self, config, payloads, device):
        if config.runtime.algorithm not in {"masac", "matd3"} or not config.algorithm_config.share_actor_parameters:
            raise ValueError("Multi-policy vmap requires shared MASAC or MATD3 Actors")
        if not payloads:
            raise ValueError("At least one trained policy must be supplied")
        self.config = config
        self.device = torch.device(device)
        self.n = config.world.num_uavs
        self.k = self.n + 1
        self.count = len(payloads)
        a = config.algorithm_config
        if config.runtime.algorithm == "masac":
            self.single = HybridActor(
                observation_dim(config)+self.n, self.k, a.hidden_sizes,
                config.continuous_dim, layer_norm=a.actor_layer_norm).to(self.device).eval()
        else:
            self.single = DeterministicActor(
                observation_dim(config)+self.n, self.k, a.hidden_sizes,
                config.continuous_dim, config.matd3.output_normalization, config.motion_dim,
                layer_norm=a.actor_layer_norm,
                batch_norm=config.matd3.actor_batch_norm).to(self.device).eval()
        weights = []
        norms = []
        expected = config_fingerprint(config)
        for saved in payloads:
            check = Config()
            _apply(check, saved["config"])
            check.validate()
            if (check.runtime.algorithm != config.runtime.algorithm or
                    config_fingerprint(check) != expected or
                    saved.get("purpose") != "evaluation_only"):
                raise ValueError("Batched Actor checkpoints must have identical scientific configs")
            tensors = saved["actors"]
            fields = {}
            for name, parameter in self.single.named_parameters():
                key = "0."+name
                if key not in tensors or tensors[key].shape != parameter.shape:
                    raise ValueError("Checkpoint Actor parameter differs: " + key)
                if not torch.isfinite(tensors[key]).all():
                    raise ValueError("Nonfinite Actor parameter in checkpoint")
                fields[name] = tensors[key]
            buffers = {}
            for name, template in self.single.named_buffers():
                key = "0."+name
                if key not in tensors or tensors[key].shape != template.shape:
                    raise ValueError("Checkpoint BatchNorm buffer differs: " + key)
                if not torch.isfinite(tensors[key]).all():
                    raise ValueError("Nonfinite Actor BatchNorm buffer: " + key)
                buffers[name] = tensors[key]
            if set(tensors) != {"0."+key for key in fields} | {"0."+key for key in buffers}:
                raise ValueError("Unexpected/missing Actor weight/buffer keys in snapshot")
            weights.append(fields)
            norms.append(buffers)
        self.params = {
            key: torch.stack([row[key] for row in weights]).to(self.device)
            for key in weights[0]}
        self.buffers = {
            key: torch.stack([row[key] for row in norms]).to(self.device)
            for key in norms[0]}

    @torch.no_grad()
    def act_batch(self, observations):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        seeds = obs.shape[0] // self.count
        if obs.ndim != 3 or obs.shape[0] != self.count*seeds or obs.shape[1] != self.n:
            raise ValueError("Observation batch does not match the grouped Actors")
        # [K, 5, agent, features], then one flattened Actor batch per policy.
        grouped = obs.reshape(self.count, seeds, self.n, -1)
        flat = grouped.reshape(self.count, seeds*self.n, -1)
        indices = (torch.arange(self.n, device=self.device)
                   .expand(seeds, self.n).reshape(1, -1).expand(self.count, -1))
        identity = F.one_hot(indices, num_classes=self.n).to(flat.dtype)
        network_in = torch.cat((flat, identity), -1)
        masks = valid_destinations(flat, self.k)
        # vmap over K state_dicts rather than replicating 8 Python actor modules.
        def infer_one(params, buffers, x, mask):
            return functional_call(self.single, (params, buffers), (x, mask))
        continuous, probs = torch.vmap(infer_one, in_dims=(0, 0, 0, 0))(
            self.params, self.buffers, network_in, masks)
        continuous = continuous.reshape(self.count, seeds, self.n, -1)
        destinations = probs.reshape(self.count, seeds, self.n, -1).argmax(-1)
        continuous = canonical_continuous(
            continuous, grouped, destinations,
            learn_power=self.config.network.power_control == "learned")
        return {"continuous":continuous.reshape(self.count*seeds, self.n, -1).contiguous(),
                "destination":destinations.reshape(self.count*seeds, self.n).contiguous()}


def evaluate_group(config, payloads, *, device, seed, episodes, vector=None):
    """Vectorized numerical evaluation records, one list per policy."""
    if episodes < 1 or not payloads:
        raise ValueError("Invalid policy group size/seed count")
    model = BatchedSharedActor(config, payloads, device)
    total_envs = model.count*episodes
    env_config = deepcopy(config)
    env_config.runtime.num_envs = total_envs
    env_config.runtime.device = "cuda" if model.device.type == "cuda" else "cpu"
    owned_vector = vector is None
    if vector is None:
        vector = WarpVectorEnv(env_config, device=device)
    elif (vector.num_envs != total_envs or vector.device != model.device):
        raise ValueError("Preallocated Warp pool has incompatible size/device")
    seeds = [seed+i for _ in payloads for i in range(episodes)]
    done = torch.zeros(total_envs, dtype=torch.bool, device=model.device)
    returns = torch.zeros(total_envs, dtype=torch.float64, device=model.device)
    info_by_world = [None]*total_envs
    try:
        obs, _ = vector.reset(seed=seeds)
        for step in range(config.world.max_steps):
            actions = model.act_batch(obs["observation"])
            actions["continuous"][done] = 0
            actions["destination"][done] = 0
            obs, reward, terminated, truncated, info = vector.step(actions)
            returns += reward.to(dtype=torch.float64) * (~done)
            just_done = (terminated | truncated) & (~done)
            for index in torch.nonzero(just_done).flatten().cpu().tolist():
                detail = info["device_episode_infos"].get(index)
                if detail is None:
                    raise RuntimeError("Warp GPU terminal metrics missing for policy world")
                info_by_world[index] = detail
            done |= terminated | truncated
            if bool(done.all()):
                break
        if not bool(done.all()) or any(entry is None for entry in info_by_world):
            raise RuntimeError("Grouped GPU Evaluation left an episode incomplete")
        sums = returns.cpu().tolist()
        grouped_records = []
        for policy_index in range(model.count):
            records = []
            for offset in range(episodes):
                i = policy_index*episodes+offset
                info = info_by_world[i]
                records.append(gpu_episode_record(info, config, seed=seeds[i],
                                                  episode=offset+1, episode_return=sums[i]))
            grouped_records.append(records)
        return grouped_records
    finally:
        if owned_vector:
            vector.close()
