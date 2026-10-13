"""Hybrid MASAC with independent actors and centralized twin soft critics."""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from ..environment.env import Action
from .action_utils import canonical_continuous, joint_action, valid_destinations
from .checkpoints import load_policy, save_policy
from .models import HybridActor, TwinCritic
from .optimization import scalar_metrics, step_optimizer, step_actor_optimizers, tensor_batch, critic_metrics


def soft_bellman_target(reward, terminated, soft_value, gamma):
    return reward + gamma * (1 - terminated) * soft_value


class MASAC(nn.Module):
    algorithm = "masac"

    def __init__(self, config, observation_dim, state_dim, device="cpu"):
        super().__init__()
        self.config = deepcopy(config)
        self.config.runtime.algorithm = self.algorithm
        self.observation_dim, self.state_dim = observation_dim, state_dim
        self.device = torch.device(device)
        self.num_uavs = n = config.world.num_uavs
        self.destinations = k = n + 1
        a = config.masac
        self.actors = nn.ModuleList(HybridActor(observation_dim, k, a.hidden_sizes) for _ in range(n))
        self.critics = nn.ModuleList(TwinCritic(state_dim, n, k, a.hidden_sizes) for _ in range(n))
        self.target_critics = deepcopy(self.critics)
        self.target_critics.requires_grad_(False)
        self.log_alpha_continuous = nn.Parameter(torch.full((n,), math.log(a.initial_alpha)))
        self.log_alpha_discrete = nn.Parameter(torch.full((n,), math.log(a.initial_alpha)))
        self.to(self.device)
        adam = dict(lr=a.learning_rate, fused=config.runtime.fused_optimizers and self.device.type == "cuda")
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), **adam) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), **adam)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha_continuous, self.log_alpha_discrete], **adam)
        self.updates = 0

    def _sample(self, actor, observation, deterministic=False):
        mask = valid_destinations(observation, self.destinations)
        continuous, parts, probs, logs = actor.sample(observation, deterministic, mask, return_parts=True)
        # 0*log(0) contributes zero. Clear it before alpha multiplication:
        # float32.min * alpha>1 overflows, even though the destination has p=0.
        logs = logs.masked_fill(probs == 0, 0)
        active = (observation[:, 9] > .5).to(continuous.dtype)
        continuous = continuous * active[:, None]
        # Power is a control only when the categorical destination is non-silent.
        conditional_logp = (parts[:, :2].sum(-1) + (1 - probs[:, 0]) * parts[:, 2]) * active
        return continuous, conditional_logp, probs, logs

    def _joint_sample(self, observations, deterministic=False):
        actions, logs, probabilities, log_probabilities, raw_actions = [], [], [], [], []
        for i, actor in enumerate(self.actors):
            obs = observations[:, i]
            continuous, logp, probs, log_probs = self._sample(actor, obs, deterministic)
            destination = probs.argmax(-1) if deterministic else Categorical(probs=probs, validate_args=False).sample()
            actions.append(joint_action(canonical_continuous(continuous, obs, destination),
                                        destination, self.destinations))
            logs.append(logp)
            probabilities.append(probs)
            log_probabilities.append(log_probs)
            raw_actions.append(continuous)
        return tuple(torch.stack(values, 1) for values in
                     (actions, logs, probabilities, log_probabilities, raw_actions))

    @torch.no_grad()
    def act_batch(self, observations, deterministic=False, step=0):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        joint, *_ = self._joint_sample(obs, deterministic)
        values = joint.cpu().numpy()
        return Action(values[..., :3], values[..., 3:].argmax(-1).astype(np.int64))

    def act(self, observations, deterministic=False, step=0):
        action = self.act_batch(np.asarray(observations)[None], deterministic, step)
        return Action(action.continuous[0], action.destination[0])

    def _expected_q(self, critic, state, joint, continuous, index):
        """Enumerate own destination; sample and detach all other agents."""
        b, k = len(state), self.destinations
        alternatives = joint.detach()[:, None].expand(-1, k, -1, -1).clone()
        choices = continuous[:, None].expand(-1, k, -1).clone()
        choices[:, 0, 2] = 0
        alternatives[:, :, index, :3] = choices
        alternatives[:, :, index, 3:] = torch.eye(k, device=self.device)[None]
        repeated_state = state[:, None].expand(-1, k, -1).reshape(b * k, -1)
        q1, q2 = critic(repeated_state, alternatives.reshape(b * k, self.num_uavs, -1))
        return torch.minimum(q1, q2).reshape(b, k)

    def update(self, batch, collect_metrics=True):
        data = tensor_batch(batch, self.device)
        a = self.config.masac
        alpha_c, alpha_d = self.log_alpha_continuous.exp().detach(), self.log_alpha_discrete.exp().detach()
        with torch.no_grad():
            joint, logs, probs, log_probs, continuous = self._joint_sample(data["next_observation"])
            targets = []
            for i, critic in enumerate(self.target_critics):
                q = self._expected_q(critic, data["next_state"], joint, continuous[:, i], i)
                soft = (probs[:, i] * (q - alpha_d[i] * log_probs[:, i])).sum(-1) - alpha_c[i] * logs[:, i]
                targets.append(soft_bellman_target(data["reward"], data["terminated"], soft, a.gamma))
            target = torch.stack(targets, 1)
        replay_joint = joint_action(data["continuous"], data["destination"], self.destinations)
        values = [critic(data["state"], replay_joint) for critic in self.critics]
        q1, q2 = (torch.stack([q[k] for q in values], 1) for k in (0, 1))
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_grad = step_optimizer(critic_loss, self.critic_optimizer, self.critics.parameters(), a.grad_clip_norm, collect_metrics)
        with torch.no_grad():
            base_joint, *_ = self._joint_sample(data["observation"])
        self.critics.requires_grad_(False)
        actor_grads = []
        actor_losses, entropies_c, entropies_d, targets_c, targets_d = [], [], [], [], []
        try:
            for i, actor in enumerate(self.actors):
                obs = data["observation"][:, i]
                continuous, logp, probs, log_probs = self._sample(actor, obs)
                q = self._expected_q(self.critics[i], data["state"], base_joint, continuous, i)
                loss = (alpha_c[i] * logp + (probs * (alpha_d[i] * log_probs - q)).sum(-1)).mean()
                actor_losses.append(loss)
                entropies_c.append(-logp.detach().mean())
                entropies_d.append(-(probs.detach() * log_probs.detach()).sum(-1).mean())
                active = (obs[:, 9] > .5).float()
                targets_c.append((a.target_continuous_entropy / 3 * (3 - probs[:, 0].detach()) * active).mean())
                choices = valid_destinations(obs, self.destinations).sum(-1).float()
                targets_d.append((a.target_discrete_entropy_ratio * choices.log()).mean())
            actor_grads = step_actor_optimizers(actor_losses, self.actor_optimizers,
                                               [actor.parameters() for actor in self.actors],
                                               a.grad_clip_norm, collect_metrics)
            actor_losses = [loss.detach() for loss in actor_losses]
        finally:
            self.critics.requires_grad_(True)
        hc, hd, tc, td = (torch.stack(v) for v in (entropies_c, entropies_d, targets_c, targets_d))
        before_c, before_d = self.log_alpha_continuous.detach().clone(), self.log_alpha_discrete.detach().clone()
        alpha_loss = (self.log_alpha_continuous * (hc - tc) + self.log_alpha_discrete * (hd - td)).mean()
        alpha_grad = step_optimizer(alpha_loss, self.alpha_optimizer,
                       [self.log_alpha_continuous, self.log_alpha_discrete], a.grad_clip_norm, collect_metrics)
        with torch.no_grad():
            available_c = (data["observation"][:, :, 9] > .5).any(0)
            available_d = (valid_destinations(data["observation"], self.destinations).sum(-1) > 1).any(0)
            self.log_alpha_continuous.copy_(torch.where(available_c, self.log_alpha_continuous, before_c))
            self.log_alpha_discrete.copy_(torch.where(available_d, self.log_alpha_discrete, before_d))
            self.log_alpha_continuous.clamp_(-20, 5)
            self.log_alpha_discrete.clamp_(-20, 5)
            for target_parameter, parameter in zip(self.target_critics.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, a.tau)
        self.updates += 1
        if not collect_metrics:
            return {}
        return scalar_metrics({**critic_metrics(q1, q2, target, data),
                               "critic_grad_norm": critic_grad, "actor_grad_norm": torch.stack(actor_grads).mean(),
                               "alpha_grad_norm": alpha_grad,
                               "critic_gradient_clipped": (critic_grad > a.grad_clip_norm).float(),
                               "critic_loss": critic_loss, "actor_loss": torch.stack(actor_losses).mean(),
                               "continuous_entropy": hc.mean(), "discrete_entropy": hd.mean(),
                               "alpha_continuous": self.log_alpha_continuous.exp().mean(),
                               "alpha_discrete": self.log_alpha_discrete.exp().mean()})

    def save(self, path, progress=None):
        save_policy(self, path, progress)

    @classmethod
    def load(cls, path, device="cpu"):
        return load_policy(path, device, expected_algorithm=cls.algorithm)
