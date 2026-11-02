"""Diagnostic collector-only TX-power coverage for MASAC causal experiments."""
import numpy as np
import torch

from ..environment.env import Action


_POWER_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
_BUCKETS = 10_000


def apply_power_support(action, observation, *, step, config):
    fraction = float(config.runtime.power_support_fraction)
    if (fraction <= 0 or config.runtime.algorithm != "masac"
            or config.network.power_control != "learned"):
        return action
    continuous, destination = action.continuous, action.destination
    if isinstance(continuous, torch.Tensor):
        obs = torch.as_tensor(observation, device=continuous.device)
        dest = torch.as_tensor(destination, device=continuous.device)
        flat = torch.arange(dest.numel(), device=continuous.device).reshape(dest.shape)
        threshold = int(round(fraction * _BUCKETS))
        selected = (obs[..., 9] > .5) & (dest != 0) & (((flat + int(step)) % _BUCKETS) < threshold)
        grid = torch.tensor(_POWER_GRID, dtype=continuous.dtype, device=continuous.device)
        values = grid[(flat + int(step)) % len(_POWER_GRID)]
        result = continuous.clone()
        result[..., -1] = torch.where(selected, values, result[..., -1])
        return Action(result, destination)
    obs = np.asarray(observation)
    dest = np.asarray(destination)
    flat = np.arange(dest.size, dtype=np.int64).reshape(dest.shape)
    threshold = int(round(fraction * _BUCKETS))
    selected = (obs[..., 9] > .5) & (dest != 0) & (((flat + int(step)) % _BUCKETS) < threshold)
    values = np.asarray(_POWER_GRID, dtype=np.asarray(continuous).dtype)[(flat + int(step)) % len(_POWER_GRID)]
    result = np.asarray(continuous).copy()
    result[..., -1] = np.where(selected, values, result[..., -1])
    return Action(result, destination)
