"""Isolated research candidate: first-layer affine projection reuse.

Only candidate evaluation changes. No changes to losses, network parameters,
Actor observation paths, or replay sampling.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _verify(critic, state, base_joint):
    assert state.ndim == 2 and base_joint.ndim == 3
    n, action_dim = base_joint.shape[1:]
    state_dim = state.shape[-1]
    for tower in (critic.q1, critic.q2):
        layer = tower[0]
        assert isinstance(layer, nn.Linear), "Requires first nn.Linear"
        assert layer.in_features == state_dim + n * action_dim
    return n, action_dim, state_dim


def _shared_tower(tower, base, delta, state_dim, action_dim):
    n, b, k, d = delta.shape
    assert d == action_dim
    layer = tower[0]
    zbase = F.linear(base, layer.weight, layer.bias)
    blocks = torch.stack([
        layer.weight[:, state_dim + i * d:state_dim + (i + 1) * d]
        for i in range(n)
    ])
    dz = torch.bmm(delta.reshape(n, b * k, d), blocks.transpose(1, 2))
    z = zbase[None, :, None] + dz.reshape(n, b, k, -1)
    return tower[1:](z.reshape(n * b * k, -1)).reshape(n, b, k)


def _individual_tower(tower, base, delta, state_dim, action_dim, index):
    b, k, d = delta.shape
    assert d == action_dim
    layer = tower[0]
    zbase = F.linear(base, layer.weight, layer.bias)
    weight = layer.weight[:, state_dim + index * d:state_dim + (index + 1) * d]
    dz = F.linear(delta, weight)
    return tower[1:]((zbase[:, None] + dz).reshape(b * k, -1)).reshape(b, k)


def masac_shared_q(critic, state, joint, continuous):
    """Return [batch, agent, destination] clipped twin Q."""
    b, n = state.shape[0], joint.shape[1]
    nc, action_dim, state_dim = _verify(critic, state, joint)
    k = joint.shape[-1] - continuous.shape[-1]
    assert nc == n and k == n + 1 and action_dim == continuous.shape[-1] + k
    c = continuous.shape[-1]
    base_joint = joint.detach()
    choices = continuous.transpose(0, 1)[:, :, None].expand(n, b, k, c).clone()
    choices[:, :, 0, -1] = 0
    discrete = torch.eye(k, dtype=joint.dtype, device=joint.device)[None, None].expand(n, b, -1, -1)
    candidates = torch.cat((choices, discrete), -1)
    delta = candidates - base_joint.transpose(0, 1)[:, :, None]
    base = torch.cat((state, base_joint.reshape(b, -1)), -1)
    q1 = _shared_tower(critic.q1, base, delta, state_dim, action_dim)
    q2 = _shared_tower(critic.q2, base, delta, state_dim, action_dim)
    return torch.minimum(q1, q2).permute(1, 0, 2).contiguous()


def masac_individual_q(critic, state, joint, continuous, index):
    """Return [batch, destination] clipped twin Q for one independent critic."""
    b, n, action_dim = joint.shape
    _, width, state_dim = _verify(critic, state, joint)
    c = continuous.shape[-1]
    k = action_dim - c
    assert k == n + 1
    choices = continuous[:, None].expand(b, k, c).clone()
    choices[:, 0, -1] = 0
    discrete = torch.eye(k, device=joint.device, dtype=joint.dtype)[None].expand(b, -1, -1)
    candidate = torch.cat((choices, discrete), -1)
    base_joint = joint.detach()
    base = torch.cat((state, base_joint.reshape(b, -1)), -1)
    delta = candidate - base_joint[:, index, None]
    q1 = _individual_tower(critic.q1, base, delta, state_dim, width, index)
    q2 = _individual_tower(critic.q2, base, delta, state_dim, width, index)
    return torch.minimum(q1, q2)


def matd3_shared_twin_q(critic, state, joints):
    """Return both twin heads [batch, agent], preserving existing call interface.

    Each input candidate changes exactly one agent action; recover the base
    from the peer actions of a different candidate. Base remains detached.
    """
    n, b, n2, d = joints.shape
    assert n == n2 and n >= 2
    base_joint = torch.stack([joints[(i + 1) % n, :, i] for i in range(n)], dim=1).detach()
    _, width, state_dim = _verify(critic, state, base_joint)
    assert width == d
    own_candidates = torch.stack([joints[i, :, i] for i in range(n)], dim=0)
    delta = (own_candidates - base_joint.transpose(0, 1))[:, :, None, :]
    base = torch.cat((state, base_joint.reshape(b, -1)), -1)
    q1 = _shared_tower(critic.q1, base, delta, state_dim, width)
    q2 = _shared_tower(critic.q2, base, delta, state_dim, width)
    return (
        q1[..., 0].transpose(0, 1).contiguous(),
        q2[..., 0].transpose(0, 1).contiguous(),
    )


def matd3_shared_actor_q1(critic, state, joints):
    """Return (Q1 [B,N], None) for MATD3 shared-critic *actor* objective.

    Q2 is intentionally not evaluated here: TD3 actor maximizes Q1 only.
    The Bellman critic update still computes BOTH independent twin Q networks.
    This function is permitted only in the unchunked shared-critic actor path.
    """
    n, b, other_n, action_width = joints.shape
    assert n == other_n and n >= 2
    base_joint = torch.stack(
        [joints[(i + 1) % n, :, i] for i in range(n)], dim=1
    ).detach()
    _, width, state_dim = _verify(critic, state, base_joint)
    assert width == action_width
    own_candidates = torch.stack(
        [joints[i, :, i] for i in range(n)], dim=0
    )
    delta = (
        own_candidates - base_joint.transpose(0, 1)
    )[:, :, None, :]
    base = torch.cat((state, base_joint.reshape(b, -1)), -1)
    q1 = _shared_tower(critic.q1, base, delta, state_dim, width)
    return q1[..., 0].transpose(0, 1).contiguous(), None
