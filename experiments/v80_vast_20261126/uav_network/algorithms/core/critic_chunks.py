"""Bound peak critic activations while preserving a full logical batch."""
import torch
from torch.utils.checkpoint import checkpoint


def chunked_call(function, state, joint, continuous, chunk_size, *batch_inputs):
    if not chunk_size or len(state) <= chunk_size:
        return function(state, joint, continuous, *batch_inputs)
    values = []
    for start in range(0, len(state), chunk_size):
        rows = slice(start, start + chunk_size)
        inputs = (state[rows], joint[rows], continuous[rows],
                  *(value[rows] for value in batch_inputs))
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

def chunked_single_q_values(function, state, joints, chunk_size):
    """Single-head actor Q with checkpointing over the replay batch axis.

    Joints are [candidate, batch, uav, action]. This helper is not used
    for the twin-Q target or critic training objective.
    """
    if not chunk_size or len(state) <= chunk_size:
        return function(state, joints)
    values = []
    for start in range(0, len(state), chunk_size):
        inputs = (state[start:start + chunk_size], joints[:, start:start + chunk_size])
        value = (checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=False)
                 if torch.is_grad_enabled() else function(*inputs))
        values.append(value)
    return torch.cat(values, dim=0)
