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
from .parallel_critics import critic_values
from .precision import MixedPrecision, autocast_update
from .optimization import scalar_metrics, step_optimizer, step_actor_optimizers, tensor_batch, critic_metrics


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
        self.share_actor_parameters = a.share_actor_parameters
        self.share_critic_parameters = a.share_critic_parameters
        actor_input_dim = observation_dim + n if self.share_actor_parameters else observation_dim
        actor_count = 1 if self.share_actor_parameters else n
        critic_count = 1 if self.share_critic_parameters else n
        self.actors = nn.ModuleList(DeterministicActor(actor_input_dim, k, a.hidden_sizes) for _ in range(actor_count))
        self.critics = nn.ModuleList(TwinCritic(state_dim, n, k, a.hidden_sizes) for _ in range(critic_count))
        self.target_actors, self.target_critics = deepcopy(self.actors), deepcopy(self.critics)
        self.target_actors.requires_grad_(False)
        self.target_critics.requires_grad_(False)
        self.to(self.device)
        adam = dict(lr=a.learning_rate)
        if config.runtime.fused_optimizers and self.device.type == "cuda":
            adam["fused"] = True
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), **adam) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), **adam)
        self.precision = MixedPrecision(config.runtime.amp_mode, self.device,
                                        ("critic", "actor", "alpha") if self.algorithm == "masac" else ("critic", "actor"))
        self.updates = self.actor_updates = 0
        self._last_actor_metrics = {}

    def _actor_at(self, index, *, target=False):
        actors = self.target_actors if target else self.actors
        return actors[0] if self.share_actor_parameters else actors[index]

    def _actor_input(self, observation, index):
        if not self.share_actor_parameters:
            return observation
        identity = F.one_hot(torch.full((len(observation),), index, device=observation.device, dtype=torch.long),
                             num_classes=self.num_uavs).to(observation.dtype)
        return torch.cat((observation, identity), dim=-1)

    def _actor_action(self, actor, observation, straight_through=False, agent_index=0):
        actor_observation = self._actor_input(observation, agent_index)
        continuous, probs = actor(actor_observation, valid_destinations(observation, self.destinations))
        destination = probs.argmax(-1)
        hard = F.one_hot(destination, self.destinations).to(probs.dtype)
        discrete = hard + probs - probs.detach() if straight_through else hard
        continuous = canonical_continuous(continuous, observation, destination)
        return torch.cat((continuous, discrete), -1)

    @torch.no_grad()
    def target_actions(self, observations):
        a = self.config.matd3
        actions = []
        for i in range(self.num_uavs):
            actor = self._actor_at(i, target=True)
            obs = observations[:, i]
            actor_observation = self._actor_input(obs, i)
            continuous, probs = actor(actor_observation, valid_destinations(obs, self.destinations))
            noise = (torch.randn_like(continuous) * a.target_noise_std).clamp(-a.target_noise_clip, a.target_noise_clip)
            destination = probs.argmax(-1)
            continuous = canonical_continuous((continuous + noise).clamp(-1, 1), obs, destination)
            actions.append(joint_action(continuous, destination, self.destinations))
        return torch.stack(actions, 1)

    @torch.no_grad()
    def act_batch(self, observations, deterministic=False, step=0, *, return_tensors=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        a = self.config.matd3
        fraction = min(1, max(0, step) / a.epsilon_decay_steps)
        epsilon = a.initial_epsilon + fraction * (a.final_epsilon - a.initial_epsilon)
        actions = []
        for i in range(self.num_uavs):
            actor = self._actor_at(i)
            local = obs[:, i]
            mask = valid_destinations(local, self.destinations)
            continuous, probs = actor(self._actor_input(local, i), mask)
            destination = probs.argmax(-1)
            if not deterministic:
                continuous = (continuous + torch.randn_like(continuous) * a.exploration_noise_std).clamp(-1, 1)
                random_destination = torch.multinomial(mask.float(), 1).squeeze(-1)
                explore = torch.rand(len(local), device=self.device) < epsilon
                destination = torch.where(explore, random_destination, destination)
            continuous = canonical_continuous(continuous, local, destination)
            actions.append(joint_action(continuous, destination, self.destinations))
        joint = torch.stack(actions, 1)
        if return_tensors:
            return Action(joint[..., :3].contiguous(), joint[..., 3:].argmax(-1))
        values = joint.cpu().numpy()
        return Action(values[..., :3], values[..., 3:].argmax(-1).astype(np.int64))

    def act(self, observations, deterministic=False, step=0):
        action = self.act_batch(np.asarray(observations)[None], deterministic, step)
        return Action(action.continuous[0], action.destination[0])

    def _critic_values(self, *args):
        return getattr(self, "_compiled_critic_values", critic_values)(*args)

    @autocast_update
    def update_critic(self, batch, collect_metrics=True):
        data, a = tensor_batch(batch, self.device), self.config.matd3
        vectorized = self.config.runtime.vectorize_critics
        with torch.no_grad():
            next_actions = self.target_actions(data["next_observation"])
            if vectorized:
                target = td3_target(data["reward"][:, None], data["terminated"][:, None],
                                    *self._critic_values(self.target_critics, data["next_state"], next_actions, True), a.gamma)
            else:
                target = torch.stack([td3_target(data["reward"], data["terminated"],
                                                *critic(data["next_state"], next_actions), a.gamma)
                                      for critic in self.target_critics], 1)
        replay_actions = joint_action(data["continuous"], data["destination"], self.destinations)
        q1, q2 = self._critic_values(self.critics, data["state"], replay_actions, vectorized)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_step = (lambda *args: self.precision.step("critic", *args)) if self.precision.enabled else step_optimizer
        critic_grad = critic_step(loss, self.critic_optimizer, self.critics.parameters(), a.grad_clip_norm, collect_metrics)
        batch_metrics = critic_metrics(q1, q2, target, data) if collect_metrics else {}
        self.updates += 1
        if not collect_metrics:
            return {}
        return scalar_metrics({**batch_metrics, **self._last_actor_metrics,
                               "critic_loss": loss, "critic_grad_norm": critic_grad,
                               "critic_gradient_clipped": (critic_grad > a.grad_clip_norm).float(),
                               "policy_updated": torch.tensor(0., device=self.device)})

    @autocast_update
    def update_actor(self, batch, collect_metrics=True):
        data, a = tensor_batch(batch, self.device), self.config.matd3
        vectorized = self.config.runtime.vectorize_critics
        actor_losses, actor_grads = [], []
        with torch.no_grad():
            base = torch.stack([self._actor_action(self._actor_at(i), data["observation"][:, i], agent_index=i)
                                for i in range(self.num_uavs)], 1)
        self.critics.requires_grad_(False)
        try:
            batched_q = None
            if vectorized:
                joints = base[None].expand(self.num_uavs, -1, -1, -1).clone()
                indices = torch.arange(self.num_uavs, device=self.device)
                joints[indices, :, indices] = torch.stack([
                    self._actor_action(self._actor_at(i), data["observation"][:, i],
                                       straight_through=True, agent_index=i)
                    for i in range(self.num_uavs)])
                batched_q, _ = self._critic_values(self.critics, data["state"], joints, True, True)
            for i in range(self.num_uavs):
                actor = self._actor_at(i)
                if vectorized:
                    q1 = batched_q[:, i]
                else:
                    joint = base.clone()
                    joint[:, i] = self._actor_action(actor, data["observation"][:, i],
                                                     straight_through=True, agent_index=i)
                    q1, _ = self.critics[i](data["state"], joint)
                actor_loss = -q1.mean()
                actor_losses.append(actor_loss)
            actor_step = self.precision.step_actors if self.precision.enabled else step_actor_optimizers
            if self.share_actor_parameters:
                step_losses = [torch.stack(actor_losses).mean()]
                parameter_groups = [self.actors[0].parameters()]
            else:
                step_losses = actor_losses
                parameter_groups = [actor.parameters() for actor in self.actors]
            actor_grads = actor_step(step_losses, self.actor_optimizers, parameter_groups,
                                     a.grad_clip_norm, collect_metrics)
            actor_losses = [loss.detach() for loss in actor_losses]
        finally:
            self.critics.requires_grad_(True)
        self._last_actor_metrics = {"actor_loss": torch.stack(actor_losses).mean(),
                                    "actor_grad_norm": torch.stack(actor_grads).mean(),
                                    "actor_update_step": torch.tensor(float(self.updates), device=self.device)}
        self.actor_updates += 1
        if not collect_metrics:
            return {}
        return scalar_metrics({**self._last_actor_metrics,
                               "policy_updated": torch.tensor(1., device=self.device)})

    @torch.no_grad()
    def update_targets(self, *, actor=True, critic=True):
        collections = []
        if actor:
            collections.append((self.target_actors, self.actors))
        if critic:
            collections.append((self.target_critics, self.critics))
        for target_collection, collection in collections:
            for target_p, p in zip(target_collection.parameters(), collection.parameters()):
                target_p.lerp_(p, self.config.matd3.tau)

    def update(self, batch, collect_metrics=True):
        result = self.update_critic(batch, collect_metrics)
        if self.updates % self.config.matd3.policy_delay == 0:
            result.update(self.update_actor(batch, collect_metrics))
            self.update_targets()
        return result

    def save(self, path, progress=None):
        save_policy(self, path, progress)

    @classmethod
    def load(cls, path, device="cpu"):
        return load_policy(path, device, expected_algorithm=cls.algorithm)
