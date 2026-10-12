"""Hybrid MASAC: separate actors and per-UAV centralized twin soft critics."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from ..configuration import Config, _apply
from ..environment.env import Action
from .models import HybridActor, TwinCritic


def soft_bellman_target(reward, terminated, soft_value, gamma):
    return reward + gamma * (1 - terminated) * soft_value


class MASAC(nn.Module):
    def __init__(self, config, observation_dim, state_dim, device="cpu"):
        super().__init__()
        self.config = deepcopy(config)
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
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=a.learning_rate) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=a.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha_continuous, self.log_alpha_discrete], lr=a.learning_rate)
        self.updates = 0

    def _joint_sample(self, observations, deterministic=False):
        actions, logs, probabilities, log_probabilities = [], [], [], []
        for i, actor in enumerate(self.actors):
            continuous, logp, probs, log_probs = actor.sample(observations[:, i], deterministic)
            destination = probs.argmax(-1) if deterministic else Categorical(probs=probs).sample()
            actions.append(torch.cat([continuous, F.one_hot(destination, self.destinations)], -1))
            logs.append(logp)
            probabilities.append(probs)
            log_probabilities.append(log_probs)
        return (torch.stack(actions, 1), torch.stack(logs, 1),
                torch.stack(probabilities, 1), torch.stack(log_probabilities, 1))

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        joint, _, _, _ = self._joint_sample(obs, deterministic)
        values = joint[0].cpu().numpy()
        return Action(values[:, :3], values[:, 3:].argmax(-1).astype(np.int64))

    def _expected_q(self, critic, state, joint, continuous, index):
        """Enumerate own categorical choice; other agents remain sampled."""
        b, k = len(state), self.destinations
        alternatives = joint.detach()[:, None].expand(-1, k, -1, -1).clone()
        alternatives[:, :, index, :3] = continuous[:, None].expand(-1, k, -1)
        alternatives[:, :, index, 3:] = torch.eye(k, device=self.device)[None]
        repeated_state = state[:, None].expand(-1, k, -1).reshape(b * k, -1)
        q1, q2 = critic(repeated_state, alternatives.reshape(b * k, self.num_uavs, -1))
        return torch.minimum(q1, q2).reshape(b, k)

    def _step_optimizer(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite MASAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters), self.config.masac.grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()

    def update(self, batch):
        data = {name: torch.as_tensor(value, device=self.device,
                                     dtype=torch.long if name == "destination" else torch.float32)
                for name, value in batch.items()}
        a = self.config.masac
        alpha_c, alpha_d = self.log_alpha_continuous.exp().detach(), self.log_alpha_discrete.exp().detach()
        with torch.no_grad():
            next_joint, next_logs, next_probs, next_log_probs = self._joint_sample(data["next_observation"])
            targets = []
            for i, critic in enumerate(self.target_critics):
                q = self._expected_q(critic, data["next_state"], next_joint, next_joint[:, i, :3], i)
                soft = (next_probs[:, i] * (q - alpha_d[i] * next_log_probs[:, i])).sum(-1)
                soft -= alpha_c[i] * next_logs[:, i]
                targets.append(soft_bellman_target(data["reward"], data["terminated"], soft, a.gamma))
            target = torch.stack(targets, 1)
        replay_joint = torch.cat([data["continuous"], F.one_hot(data["destination"], self.destinations)], -1)
        values = [critic(data["state"], replay_joint) for critic in self.critics]
        q1 = torch.stack([q[0] for q in values], 1)
        q2 = torch.stack([q[1] for q in values], 1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self._step_optimizer(critic_loss, self.critic_optimizer, self.critics.parameters())
        with torch.no_grad():
            base_joint, _, _, _ = self._joint_sample(data["observation"])
        self.critics.requires_grad_(False)
        actor_losses, continuous_entropy, discrete_entropy = [], [], []
        try:
            for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                continuous, logp, probs, log_probs = actor.sample(data["observation"][:, i])
                q = self._expected_q(self.critics[i], data["state"], base_joint, continuous, i)
                loss = (alpha_c[i] * logp + (probs * (alpha_d[i] * log_probs - q)).sum(-1)).mean()
                self._step_optimizer(loss, optimizer, actor.parameters())
                actor_losses.append(float(loss.detach()))
                continuous_entropy.append(-logp.detach().mean())
                discrete_entropy.append(-(probs.detach() * log_probs.detach()).sum(-1).mean())
        finally:
            self.critics.requires_grad_(True)
        hc, hd = torch.stack(continuous_entropy), torch.stack(discrete_entropy)
        target_d = a.target_discrete_entropy_ratio * math.log(self.destinations)
        alpha_loss = (self.log_alpha_continuous * (hc - a.target_continuous_entropy)
                      + self.log_alpha_discrete * (hd - target_d)).mean()
        self._step_optimizer(alpha_loss, self.alpha_optimizer,
                             [self.log_alpha_continuous, self.log_alpha_discrete])
        with torch.no_grad():
            self.log_alpha_continuous.clamp_(-20, 5)
            self.log_alpha_discrete.clamp_(-20, 5)
            for target_parameter, parameter in zip(self.target_critics.parameters(), self.critics.parameters()):
                target_parameter.lerp_(parameter, a.tau)
        self.updates += 1
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(np.mean(actor_losses)),
                "continuous_entropy": float(hc.mean()), "discrete_entropy": float(hd.mean()),
                "alpha_continuous": float(self.log_alpha_continuous.exp().detach().mean()),
                "alpha_discrete": float(self.log_alpha_discrete.exp().detach().mean())}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"version": 1, "config": self.config.to_dict(), "observation_dim": self.observation_dim,
                    "state_dim": self.state_dim, "model": self.state_dict(), "updates": self.updates}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        config = Config()
        _apply(config, saved["config"])
        config.validate()
        learner = cls(config, saved["observation_dim"], saved["state_dim"], device)
        learner.load_state_dict(saved["model"])
        learner.updates = saved["updates"]
        return learner
