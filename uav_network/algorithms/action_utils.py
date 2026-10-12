"""Shared policy constraints encoded in the local observation tail."""
import torch
from torch.nn import functional as F


def valid_destinations(observation, destinations):
    mask = observation[..., -destinations:] > .5
    mask = mask.clone()
    mask[..., 0] = True
    return mask


def masked_log_probabilities(logits, mask=None):
    if mask is not None:
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return F.log_softmax(logits, dim=-1)


def canonical_continuous(continuous, observation, destination):
    active = (observation[..., 9] > .5).to(continuous.dtype)
    return torch.cat((continuous[..., :2] * active[..., None],
                      continuous[..., 2:3] * (active * (destination != 0))[..., None]), -1)


def joint_action(continuous, destination, destinations):
    return torch.cat((continuous, F.one_hot(destination, destinations)), -1)
