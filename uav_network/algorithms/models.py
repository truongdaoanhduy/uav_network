"""Plain ReLU MLPs and numerically stable hybrid action distributions."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from .action_utils import masked_log_probabilities


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class RolloutBatchNorm1d(nn.BatchNorm1d):
    """Ordinary BN during actor updates, frozen running-stat BN during rollout.

    In off-policy RL, `@torch.no_grad` runs do NOT normally prevent BN buffer
    mutation. Target actor bootstrapping, rollout, and forensic logging must
    use stored running statistics, not overwrite them using an unrelated batch
    of states. This explicit policy is particular to this BN-only ablation.
    """
    def forward(self, input):
        if self.track_running_stats and not torch.is_grad_enabled():
            return F.batch_norm(input, self.running_mean, self.running_var,
                                self.weight, self.bias, training=False,
                                momentum=self.momentum, eps=self.eps)
        return super().forward(input)


def mlp(input_dim, hidden_sizes, output_dim, layer_norm=False, batch_norm=False, norm_last_linear=True):
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    if layer_norm or batch_norm:
        blocks = []
        for layer in layers:
            blocks.append(layer)
            if isinstance(layer, nn.Linear) and (norm_last_linear or layer is not layers[-1]):
                blocks.append(RolloutBatchNorm1d(layer.out_features) if batch_norm
                              else nn.LayerNorm(layer.out_features))
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

    def forward(self, observations, mask=None):
        """Deterministic mean+argmax validation for hybrid MASAC.

        Does not change the stochastic SAC update or training rollout;
        permits torch.func.functional_call/vmap across immutable checkpoints.
        """
        continuous, _, probabilities, _ = self.sample(
            observations, deterministic=True, mask=mask)
        return continuous, probabilities


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, destinations, hidden_sizes, continuous_dim=3,
                 output_normalization=False, normalization_dims=None, layer_norm=False,
                 batch_norm=False):
        super().__init__()
        self.encoder = mlp(observation_dim, hidden_sizes[:-1], hidden_sizes[-1],
                           layer_norm=layer_norm, batch_norm=batch_norm)
        self.continuous = nn.Linear(hidden_sizes[-1], continuous_dim)
        self.destination = nn.Linear(hidden_sizes[-1], destinations)
        self.output_normalization = output_normalization
        self.normalization_dims = continuous_dim if normalization_dims is None else normalization_dims

    def output_regularization(self, observations):
        """Squared raw policy parameters, before tanh or destination masking.

        Ackermann's original MATD3 p_train uses mean(flatparam**2). For the
        UAV hybrid adapter, flat parameters are XYZ/RF and receiver logits.
        This path retains a gradient when float32 tanh has saturated.
        """
        hidden = F.relu(self.encoder(observations))
        parameters = torch.cat((self.continuous(hidden), self.destination(hidden)), -1)
        return parameters.float().square().mean(-1)


    def normalize_pretanh(self, raw):
        if not self.output_normalization:
            return raw
        # Wang et al. (ICML 2020). RF is a separate control from XYZ.
        motion = raw[..., :self.normalization_dims]
        scale = motion.abs().mean(-1, keepdim=True).clamp_min(1.0)
        return torch.cat((motion / scale, raw[..., self.normalization_dims:]), -1)

    def forward(self, observations, mask):
        hidden = F.relu(self.encoder(observations))
        raw = self.normalize_pretanh(self.continuous(hidden))
        return raw.tanh(), masked_log_probabilities(self.destination(hidden), mask).exp()


class TwinCritic(nn.Module):
    def __init__(self, state_dim, num_uavs, destinations, hidden_sizes, continuous_dim=3,
                 batch_norm=False):
        super().__init__()
        input_dim = state_dim + num_uavs * (continuous_dim + destinations)
        # Never normalize scalar Q outputs: BN belongs to hidden feature layers.
        self.q1 = mlp(input_dim, hidden_sizes, 1, batch_norm=batch_norm,
                      norm_last_linear=False)
        self.q2 = mlp(input_dim, hidden_sizes, 1, batch_norm=batch_norm,
                      norm_last_linear=False)
        self.batch_norm = bool(batch_norm)
        self.state_dim = state_dim

    def forward(self, state, joint_actions):
        inputs = torch.cat([state, joint_actions.flatten(1)], dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def forward_shared_state(self, state, joint_actions, *, q1_only=False):
        """Evaluate broadcasted alternatives without repeating the global state.

        state: [batch, state_dim]; joint_actions: [candidate, batch,
        (optional route), num_uavs, action_dim]. Decompose the first
        affine map as W[s, a] + b = W_s s + W_a a + b.
        This is a computational optimization, NOT a factorized-Q critic.
        """
        if joint_actions.ndim not in (4, 5) or joint_actions.shape[1] != state.shape[0]:
            raise ValueError("joint_actions must be [candidate, batch, (route), uav, action]")
        action_inputs = joint_actions.flatten(-2)
        if self.batch_norm:
            # BN expects [samples, features]; candidate x replay x route tensors
            # cannot be forwarded as [candidate, batch, features] to BatchNorm1d.
            # Use running stats (Critic is eval() during Actor updates), preserving
            # gradients w.r.t. candidate actions and exact baseline Q semantics.
            candidates, batch = action_inputs.shape[:2]
            extra = action_inputs.shape[2:-1]
            expanded_state = state.reshape(1, batch, *([1] * len(extra)), self.state_dim)
            expanded_state = expanded_state.expand(candidates, batch, *extra, self.state_dim)
            features = torch.cat((expanded_state, action_inputs), -1).reshape(-1,
                self.state_dim + action_inputs.shape[-1])
            shape = (candidates, batch, *extra)
            q1 = self.q1(features).reshape(shape)
            return q1 if q1_only else (q1, self.q2(features).reshape(shape))

        def run(head):
            first = head[0]
            shared_projection = F.linear(state, first.weight[:, :self.state_dim])
            action_projection = F.linear(
                action_inputs, first.weight[:, self.state_dim:], first.bias)
            broadcast_shape = (1, len(state), *([1] * (action_inputs.ndim - 3)),
                               first.out_features)
            hidden = action_projection + shared_projection.reshape(broadcast_shape)
            for layer in head[1:]:
                hidden = layer(hidden)
            return hidden.squeeze(-1)

        q1 = run(self.q1)
        return q1 if q1_only else (q1, run(self.q2))


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
