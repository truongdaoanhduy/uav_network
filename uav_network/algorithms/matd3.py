"""MATD3 with a straight-through categorical next-hop branch."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..environment.env import Action
from .action_utils import canonical_continuous, joint_action, valid_destinations
from .checkpoints import load_policy, save_policy
from .models import DeterministicActor, TwinCritic
from .optimization import scalar_metrics, step_optimizer, tensor_batch, critic_metrics


def td3_target(reward, terminated, q1, q2, gamma):
    return reward + gamma * (1 - terminated) * torch.minimum(q1, q2)


class MATD3(nn.Module):
    algorithm = "matd3"

    def __init__(self, config, observation_dim, state_dim, device="cpu"):
        super().__init__()
        self.config = deepcopy(config)
        self.config.runtime.algorithm = self.algorithm
        self.observation_dim, self.state_dim = observation_dim, state_dim
        self.device = torch.device(device)
        self.num_uavs = n = config.world.num_uavs
        self.destinations = k = n + 1
        a = config.matd3
        self.actors = nn.ModuleList(DeterministicActor(observation_dim, k, a.hidden_sizes) for _ in range(n))
        self.critics = nn.ModuleList(TwinCritic(state_dim, n, k, a.hidden_sizes) for _ in range(n))
        self.target_actors, self.target_critics = deepcopy(self.actors), deepcopy(self.critics)
        self.target_actors.requires_grad_(False)
        self.target_critics.requires_grad_(False)
        self.to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=a.learning_rate) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=a.learning_rate)
        self.updates = 0
        self._last_actor_metrics = {}

    def _actor_action(self, actor, observation, straight_through=False):
        continuous, probs = actor(observation, valid_destinations(observation, self.destinations))
        destination = probs.argmax(-1)
        hard = F.one_hot(destination, self.destinations).to(probs.dtype)
        discrete = hard + probs - probs.detach() if straight_through else hard
        continuous = canonical_continuous(continuous, observation, destination)
        return torch.cat((continuous, discrete), -1)

    @torch.no_grad()
    def target_actions(self, observations):
        a = self.config.matd3
        actions = []
        for i, actor in enumerate(self.target_actors):
            obs = observations[:, i]
            continuous, probs = actor(obs, valid_destinations(obs, self.destinations))
            noise = (torch.randn_like(continuous) * a.target_noise_std).clamp(-a.target_noise_clip, a.target_noise_clip)
            destination = probs.argmax(-1)
            continuous = canonical_continuous((continuous + noise).clamp(-1, 1), obs, destination)
            actions.append(joint_action(continuous, destination, self.destinations))
        return torch.stack(actions, 1)

    @torch.no_grad()
    def act_batch(self, observations, deterministic=False, step=0):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        a = self.config.matd3
        fraction = min(1, max(0, step) / a.epsilon_decay_steps)
        epsilon = a.initial_epsilon + fraction * (a.final_epsilon - a.initial_epsilon)
        actions = []
        for i, actor in enumerate(self.actors):
            local = obs[:, i]
            mask = valid_destinations(local, self.destinations)
            continuous, probs = actor(local, mask)
            destination = probs.argmax(-1)
            if not deterministic:
                continuous = (continuous + torch.randn_like(continuous) * a.exploration_noise_std).clamp(-1, 1)
                random_destination = torch.multinomial(mask.float(), 1).squeeze(-1)
                explore = torch.rand(len(local), device=self.device) < epsilon
                destination = torch.where(explore, random_destination, destination)
            continuous = canonical_continuous(continuous, local, destination)
            actions.append(joint_action(continuous, destination, self.destinations))
        values = torch.stack(actions, 1).cpu().numpy()
        return Action(values[..., :3], values[..., 3:].argmax(-1).astype(np.int64))

    def act(self, observations, deterministic=False, step=0):
        action = self.act_batch(np.asarray(observations)[None], deterministic, step)
        return Action(action.continuous[0], action.destination[0])

    def update(self, batch, collect_metrics=True):
        data, a = tensor_batch(batch, self.device), self.config.matd3
        with torch.no_grad():
            next_actions = self.target_actions(data["next_observation"])
            target = torch.stack([td3_target(data["reward"], data["terminated"],
                                            *critic(data["next_state"], next_actions), a.gamma)
                                  for critic in self.target_critics], 1)
        replay_actions = joint_action(data["continuous"], data["destination"], self.destinations)
        values = [critic(data["state"], replay_actions) for critic in self.critics]
        q1, q2 = (torch.stack([q[k] for q in values], 1) for k in (0, 1))
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_grad = step_optimizer(loss, self.critic_optimizer, self.critics.parameters(), a.grad_clip_norm, collect_metrics)
        batch_metrics = critic_metrics(q1, q2, target, data) if collect_metrics else {}
        self.updates += 1
        policy_updated = self.updates % a.policy_delay == 0
        actor_losses, actor_grads = [], []
        if policy_updated:
            with torch.no_grad():
                base = torch.stack([self._actor_action(actor, data["observation"][:, i])
                                    for i, actor in enumerate(self.actors)], 1)
            self.critics.requires_grad_(False)
            try:
                for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                    joint = base.clone()
                    joint[:, i] = self._actor_action(actor, data["observation"][:, i], straight_through=True)
                    q1, _ = self.critics[i](data["state"], joint)
                    actor_loss = -q1.mean()
                    actor_grads.append(step_optimizer(actor_loss, optimizer, actor.parameters(), a.grad_clip_norm, collect_metrics))
                    actor_losses.append(actor_loss.detach())
            finally:
                self.critics.requires_grad_(True)
            self._last_actor_metrics = {"actor_loss": torch.stack(actor_losses).mean(),
                                        "actor_grad_norm": torch.stack(actor_grads).mean(),
                                        "actor_update_step": torch.tensor(float(self.updates), device=self.device)}
            with torch.no_grad():
                for target_collection, collection in ((self.target_actors, self.actors),
                                                       (self.target_critics, self.critics)):
                    for target_p, p in zip(target_collection.parameters(), collection.parameters()):
                        target_p.lerp_(p, a.tau)
        if not collect_metrics:
            return {}
        return scalar_metrics({**batch_metrics, **self._last_actor_metrics,
                               "critic_loss": loss, "critic_grad_norm": critic_grad,
                               "critic_gradient_clipped": (critic_grad > a.grad_clip_norm).float(),
                               "policy_updated": torch.tensor(float(policy_updated), device=self.device)})

    def save(self, path, progress=None):
        save_policy(self, path, progress)

    @classmethod
    def load(cls, path, device="cpu"):
        return load_policy(path, device, expected_algorithm=cls.algorithm)
