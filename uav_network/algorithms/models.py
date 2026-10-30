"""Plain ReLU MLPs and numerically stable hybrid action distributions."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from .action_utils import masked_log_probabilities


def mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    def __init__(self, observation_dim, destinations, hidden_sizes, continuous_dim=3):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.mean = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)

    def sample(self, observations, deterministic=False, mask=None, return_parts=False):
        hidden = F.relu(self.encoder(observations))
        mean = self.mean(hidden)
        std = self.log_std(hidden).clamp(-5, 2).exp()
        distribution = Normal(mean, std, validate_args=False)
        raw = mean if deterministic else distribution.rsample()
        continuous = raw.tanh()
        # Stable tanh Jacobian, even when the resulting action rounds to +/-1.
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        parts = distribution.log_prob(raw) - log_jacobian
        log_continuous = parts if return_parts else parts.sum(-1)
        log_discrete = masked_log_probabilities(self.destination(hidden), mask)
        return continuous, log_continuous, log_discrete.exp(), log_discrete


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, destinations, hidden_sizes, continuous_dim=3):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.continuous = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)

    def forward(self, observations, mask):
        hidden = F.relu(self.encoder(observations))
        return self.continuous(hidden).tanh(), masked_log_probabilities(self.destination(hidden), mask).exp()


class TwinCritic(nn.Module):
    def __init__(self, state_dim, num_uavs, destinations, hidden_sizes, continuous_dim=3):
        super().__init__()
        input_dim = state_dim + num_uavs * (continuous_dim + destinations)
        self.q1 = mlp(input_dim, hidden_sizes, 1)
        self.q2 = mlp(input_dim, hidden_sizes, 1)

    def forward(self, state, joint_actions):
        inputs = torch.cat([state, joint_actions.flatten(1)], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)
