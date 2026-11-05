"""Bound peak critic activations while preserving a full logical batch."""
import torch
from torch.utils.checkpoint import checkpoint


def chunked_call(function, state, joint, continuous, chunk_size):
    if not chunk_size or len(state) <= chunk_size:
        return function(state, joint, continuous)
    values = []
    for start in range(0, len(state), chunk_size):
        inputs = (state[start:start + chunk_size], joint[start:start + chunk_size],
                  continuous[start:start + chunk_size])
        # Sampling happens before this pure critic call. Recompute activations at
        # backward instead of retaining every chunk's large enumeration graph.
        value = (checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=False)
                 if torch.is_grad_enabled() else function(*inputs))
        values.append(value)
    return torch.cat(values, dim=0)


def chunked_critic_values(function, state, joint, chunk_size, per_critic=False):
    """Bound a pure twin-critic call by original replay rows.

    Q outputs always have batch on axis0. Per-critic joint inputs have shape
    [critic, batch, agent, action], so their batch must be sliced on axis1.
    Policy/noise sampling and loss reduction belong outside this function.
    """
    if not chunk_size or len(state) <= chunk_size:
        return function(state, joint)
    values = []
    for start in range(0, len(state), chunk_size):
        rows = slice(start, start + chunk_size)
        inputs = (state[rows], joint[:, rows] if per_critic else joint[rows])
        value = (checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=False)
                 if torch.is_grad_enabled() else function(*inputs))
        values.append(value)
    return tuple(torch.cat([value[i] for value in values], dim=0) for i in range(2))
