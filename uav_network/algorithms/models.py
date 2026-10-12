"""Plain ReLU MLPs and numerically stable hybrid action distributions."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    def __init__(self, observation_dim, destinations, hidden_sizes):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.mean = nn.Linear(hidden_sizes[-1], 3)
        self.log_std = nn.Linear(hidden_sizes[-1], 3)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)

    def sample(self, observations, deterministic=False):
        hidden = F.relu(self.encoder(observations))
        mean = self.mean(hidden)
        std = self.log_std(hidden).clamp(-5, 2).exp()
        distribution = Normal(mean, std)
        raw = mean if deterministic else distribution.rsample()
        continuous = raw.tanh()
        # Stable tanh Jacobian, even when the resulting action rounds to +/-1.
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_continuous = (distribution.log_prob(raw) - log_jacobian).sum(-1)
        log_discrete = F.log_softmax(self.destination(hidden), dim=-1)
        return continuous, log_continuous, log_discrete.exp(), log_discrete


class TwinCritic(nn.Module):
    def __init__(self, state_dim, num_uavs, destinations, hidden_sizes):
        super().__init__()
        input_dim = state_dim + num_uavs * (3 + destinations)
        self.q1 = mlp(input_dim, hidden_sizes, 1)
        self.q2 = mlp(input_dim, hidden_sizes, 1)

    def forward(self, state, joint_actions):
        inputs = torch.cat([state, joint_actions.flatten(1)], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)
