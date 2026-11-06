"""Plain ReLU MLPs and numerically stable hybrid action distributions."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from .action_utils import masked_log_probabilities


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(input_dim, hidden_sizes, output_dim, layer_norm=False):
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    if layer_norm:
        blocks = []
        for layer in layers:
            blocks.append(layer)
            if isinstance(layer, nn.Linear):
                blocks.append(nn.LayerNorm(layer.out_features))
        layers = blocks
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    def __init__(self, observation_dim, destinations, hidden_sizes, continuous_dim=3, layer_norm=False):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1], layer_norm=layer_norm)
        self.mean = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)

    def sample(self, observations, deterministic=False, mask=None, return_parts=False):
        hidden = F.relu(self.encoder(observations))
        mean = self.mean(hidden)
        std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
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
    def __init__(self, observation_dim, destinations, hidden_sizes, continuous_dim=3,
                 output_normalization=False, normalization_dims=None, layer_norm=False):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1], layer_norm=layer_norm)
        self.continuous = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)
        self.output_normalization = output_normalization
        self.normalization_dims = continuous_dim if normalization_dims is None else normalization_dims

    def forward(self, observations, mask):
        hidden = F.relu(self.encoder(observations))
        raw = self.continuous(hidden)
        if self.output_normalization:
            # Wang et al. (ICML 2020): when the mean pre-tanh magnitude is >1,
            # divide by that mean. Normalize motion only; learned RF power is a
            # separate control and should not rescale XYZ flight commands.
            motion = raw[..., :self.normalization_dims]
            scale = motion.abs().mean(-1, keepdim=True).clamp_min(1.0)
            motion = motion / scale
            raw = torch.cat((motion, raw[..., self.normalization_dims:]), -1)
        return raw.tanh(), masked_log_probabilities(self.destination(hidden), mask).exp()


class TwinCritic(nn.Module):
    def __init__(self, state_dim, num_uavs, destinations, hidden_sizes, continuous_dim=3):
        super().__init__()
        input_dim = state_dim + num_uavs * (continuous_dim + destinations)
        self.q1 = mlp(input_dim, hidden_sizes, 1)
        self.q2 = mlp(input_dim, hidden_sizes, 1)

    def forward(self, state, joint_actions):
        inputs = torch.cat([state, joint_actions.flatten(1)], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


def initialize_matd3_orthogonal(actors, critics):
    """Opt-in MATD3 ablation: semi-orthogonal Linear weights, zero biases.

    Encoder/critic ReLU layers use gain sqrt(2), actor output heads 0.01,
    and scalar Q heads 1.0. LayerNorm affine parameters stay at (1, 0).
    Apply to ONLINE networks before the target deepcopy and Adam creation.
    """
    def linear(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    for actor in actors:
        for layer in actor.encoder.modules():
            if isinstance(layer, nn.Linear):
                linear(layer, math.sqrt(2.0))
        linear(actor.continuous, 0.01)
        linear(actor.destination, 0.01)
    for critic in critics:
        for tower in (critic.q1, critic.q2):
            dense = [layer for layer in tower.modules() if isinstance(layer, nn.Linear)]
            for layer in dense[:-1]:
                linear(layer, math.sqrt(2.0))
            linear(dense[-1], 1.0)


def initialize_matd3_kaiming(actors, critics):
    """Opt-in He/Kaiming normal init for *all* MATD3 Linear weights.

    Hidden layers feed ReLU and use fan_in/ReLU gain; actor's tanh/logit
    and both scalar Q output heads are linear preactivations and use
    fan_in/linear gain. All Linear biases are zero. LayerNorm affine
    parameters are unchanged. Apply before target deepcopy and Adam.
    """
    def linear(layer, nonlinearity):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    for actor in actors:
        for layer in actor.encoder.modules():
            if isinstance(layer, nn.Linear):
                linear(layer, "relu")
        linear(actor.continuous, "linear")
        linear(actor.destination, "linear")
    for critic in critics:
        for tower in (critic.q1, critic.q2):
            dense = [layer for layer in tower.modules() if isinstance(layer, nn.Linear)]
            for layer in dense[:-1]:
                linear(layer, "relu")
            linear(dense[-1], "linear")
