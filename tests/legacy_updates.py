"""Frozen afeaa3d update equations: oracle for the split refactor, not production code."""
import torch
from torch.nn import functional as F
from uav_network.algorithms.action_utils import joint_action, valid_destinations
from uav_network.algorithms.parallel_critics import critic_values
from uav_network.algorithms.optimization import tensor_batch, step_optimizer, step_actor_optimizers, scalar_metrics, critic_metrics
from uav_network.algorithms.masac import soft_bellman_target
from uav_network.algorithms.matd3 import td3_target

def legacy_masac_update(self, batch, collect_metrics=True):
    data = tensor_batch(batch, self.device)
    a = self.config.masac
    vectorized = self.config.runtime.vectorize_critics
    alpha_c, alpha_d = self.log_alpha_continuous.exp().detach(), self.log_alpha_discrete.exp().detach()
    with torch.no_grad():
        joint, logs, probs, log_probs, continuous = self._joint_sample(data["next_observation"])
        targets = []
        qs = (self._expected_q_all(self.target_critics, data["next_state"], joint, continuous)
              if vectorized else None)
        for i, critic in enumerate(self.target_critics):
            q = qs[:, i] if vectorized else self._expected_q(critic, data["next_state"], joint, continuous[:, i], i)
            soft = (probs[:, i] * (q - alpha_d[i] * log_probs[:, i])).sum(-1) - alpha_c[i] * logs[:, i]
            targets.append(soft_bellman_target(data["reward"], data["terminated"], soft, a.gamma))
        target = torch.stack(targets, 1)
    replay_joint = joint_action(data["continuous"], data["destination"], self.destinations)
    q1, q2 = critic_values(self.critics, data["state"], replay_joint, vectorized)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    critic_grad = step_optimizer(critic_loss, self.critic_optimizer, self.critics.parameters(), a.grad_clip_norm, collect_metrics)
    with torch.no_grad():
        base_joint, *_ = self._joint_sample(data["observation"])
    self.critics.requires_grad_(False)
    actor_grads = []
    actor_losses, entropies_c, entropies_d, targets_c, targets_d = [], [], [], [], []
    try:
        samples = ([self._sample(actor, data["observation"][:, i]) for i, actor in enumerate(self.actors)]
                   if vectorized else None)
        qs = (self._expected_q_all(self.critics, data["state"], base_joint,
                                  torch.stack([s[0] for s in samples], 1)) if vectorized else None)
        for i, actor in enumerate(self.actors):
            obs = data["observation"][:, i]
            continuous, logp, probs, log_probs = samples[i] if vectorized else self._sample(actor, obs)
            q = qs[:, i] if vectorized else self._expected_q(self.critics[i], data["state"], base_joint, continuous, i)
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

def legacy_matd3_update(self, batch, collect_metrics=True):
    data, a = tensor_batch(batch, self.device), self.config.matd3
    vectorized = self.config.runtime.vectorize_critics
    with torch.no_grad():
        next_actions = self.target_actions(data["next_observation"])
        if vectorized:
            target = td3_target(data["reward"][:, None], data["terminated"][:, None],
                                *critic_values(self.target_critics, data["next_state"], next_actions, True), a.gamma)
        else:
            target = torch.stack([td3_target(data["reward"], data["terminated"],
                                            *critic(data["next_state"], next_actions), a.gamma)
                                  for critic in self.target_critics], 1)
    replay_actions = joint_action(data["continuous"], data["destination"], self.destinations)
    q1, q2 = critic_values(self.critics, data["state"], replay_actions, vectorized)
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
            batched_q = None
            if vectorized:
                joints = base[None].expand(self.num_uavs, -1, -1, -1).clone()
                indices = torch.arange(self.num_uavs, device=self.device)
                joints[indices, :, indices] = torch.stack([
                    self._actor_action(actor, data["observation"][:, i], straight_through=True)
                    for i, actor in enumerate(self.actors)])
                batched_q, _ = critic_values(self.critics, data["state"], joints, True, True)
            for i, actor in enumerate(self.actors):
                if vectorized:
                    q1 = batched_q[:, i]
                else:
                    joint = base.clone()
                    joint[:, i] = self._actor_action(actor, data["observation"][:, i], straight_through=True)
                    q1, _ = self.critics[i](data["state"], joint)
                actor_loss = -q1.mean()
                actor_losses.append(actor_loss)
            actor_grads = step_actor_optimizers(actor_losses, self.actor_optimizers,
                                               [actor.parameters() for actor in self.actors],
                                               a.grad_clip_norm, collect_metrics)
            actor_losses = [loss.detach() for loss in actor_losses]
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

