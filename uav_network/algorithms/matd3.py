"""MATD3 with a soft Gumbel-Softmax categorical next-hop branch."""
from copy import deepcopy

import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F

from .action_types import Action
from .action_utils import canonical_continuous, joint_action, valid_destinations
from .checkpoints import load_policy, save_policy
from .models import DeterministicActor, TwinCritic, initialize_matd3_orthogonal, initialize_matd3_kaiming
from .parallel_critics import critic_values
from .critic_chunks import chunked_critic_values, chunked_single_q_values
from .precision import MixedPrecision, autocast_update
from .optimization import scalar_metrics, step_optimizer, step_actor_optimizers, tensor_batch, critic_metrics
from .matd3_diagnostics import MATD3DiagnosticsMixin
from .matd3_gradient_probe import MATD3GradientProbe
from .layerwise_gradient_probe import LayerwiseGradientProbe


def td3_target(reward, terminated, q1, q2, gamma):
    """TD3 clipped-double-Q target. Do not bootstrap from true terminations."""
    nonterminal = 1.0 - terminated.to(dtype=reward.dtype)
    return reward + gamma * nonterminal * torch.minimum(q1, q2)


class MATD3(MATD3DiagnosticsMixin, nn.Module):
    algorithm = "matd3"

    def __init__(self, config, observation_dim, state_dim, device="cpu"):
        super().__init__()
        self.config = deepcopy(config)
        self.config.runtime.algorithm = self.algorithm
        self.observation_dim, self.state_dim = observation_dim, state_dim
        self.device = torch.device(device)
        self.num_uavs = n = config.world.num_uavs
        self.destinations = k = n + 1
        self.continuous_dim = config.continuous_dim
        self.learn_power = config.network.power_control == "learned"
        a = config.matd3
        self.share_actor_parameters = a.share_actor_parameters
        self.share_critic_parameters = a.share_critic_parameters
        self._factorized_critic_enabled = os.environ.get("UAV_FACTORIZED_CRITIC", "0").lower() in {"1", "true", "yes"}
        actor_input_dim = observation_dim + n if self.share_actor_parameters else observation_dim
        actor_count = 1 if self.share_actor_parameters else n
        critic_count = 1 if self.share_critic_parameters else n
        self.actors = nn.ModuleList(DeterministicActor(actor_input_dim, k, a.hidden_sizes, config.continuous_dim,
                                                       a.output_normalization, config.motion_dim,
                                                       layer_norm=a.actor_layer_norm,
                                                       batch_norm=a.actor_batch_norm)
                                    for _ in range(actor_count))
        self.critics = nn.ModuleList(TwinCritic(state_dim, n, k, a.hidden_sizes, config.continuous_dim,
                                                       batch_norm=a.critic_batch_norm) for _ in range(critic_count))
        if a.initialization in {"orthogonal", "kaiming"}:
            # Preserve the CPU RNG state after the standard Module creation so
            # the ablation does not shift subsequent exploration/replay draws.
            with torch.random.fork_rng(devices=[]):
                initializer = (initialize_matd3_orthogonal if a.initialization == "orthogonal"
                               else initialize_matd3_kaiming)
                initializer(self.actors, self.critics)
        self.target_actors, self.target_critics = deepcopy(self.actors), deepcopy(self.critics)
        self.target_actors.requires_grad_(False)
        self.target_critics.requires_grad_(False)
        if a.actor_batch_norm:
            # Fixed target running statistics are required for valid Bellman targets.
            self.target_actors.eval()
        if a.critic_batch_norm:
            self.target_critics.eval()
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
        if isinstance(index, int):
            index = torch.full((len(observation),), index, device=observation.device, dtype=torch.long)
        else:
            index = torch.as_tensor(index, device=observation.device, dtype=torch.long)
        identity = F.one_hot(index, num_classes=self.num_uavs).to(observation.dtype)
        return torch.cat((observation, identity), dim=-1)

    @staticmethod
    def _relaxed_destination(probs):
        # Ackermann MATD3 uses a soft categorical (Gumbel-Softmax) sample for
        # differentiable discrete policies. Invalid masked choices stay outside
        # the support by receiving the minimum finite logit.
        tiny = torch.finfo(probs.dtype).tiny
        logits = probs.clamp_min(tiny).log().masked_fill(probs == 0, torch.finfo(probs.dtype).min)
        return F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)

    def _actor_action(self, actor, observation, relaxed_discrete=False, agent_index=0):
        actor_observation = self._actor_input(observation, agent_index)
        continuous, probs = actor(actor_observation, valid_destinations(observation, self.destinations))
        discrete = (self._relaxed_destination(probs) if relaxed_discrete
                    else F.one_hot(probs.argmax(-1), self.destinations).to(probs.dtype))
        destination = discrete.argmax(-1)
        continuous = canonical_continuous(continuous, observation, destination, learn_power=self.learn_power)
        return torch.cat((continuous, discrete), -1)

    def _policy_outputs(self, observations, *, target=False):
        """Return continuous actions/probabilities for every UAV.

        A shared actor is evaluated once on a flattened B*N batch so sharing
        reduces kernel-launch and graph overhead in addition to parameter count.
        """
        if self.share_actor_parameters:
            b, n = observations.shape[:2]
            flat = observations.reshape(b * n, -1)
            indices = torch.arange(n, device=observations.device).expand(b, n).reshape(-1)
            actor = self._actor_at(0, target=target)
            continuous, probs = actor(self._actor_input(flat, indices),
                                      valid_destinations(flat, self.destinations))
            return continuous.reshape(b, n, self.continuous_dim), probs.reshape(b, n, self.destinations)
        actors = self.target_actors if target else self.actors
        values = [actor(observations[:, i], valid_destinations(observations[:, i], self.destinations))
                  for i, actor in enumerate(actors)]
        return tuple(torch.stack([value[j] for value in values], 1) for j in range(2))

    def _actor_actions(self, observations, *, relaxed_discrete=False, target=False):
        continuous, probs = self._policy_outputs(observations, target=target)
        discrete = (self._relaxed_destination(probs) if relaxed_discrete
                    else F.one_hot(probs.argmax(-1), self.destinations).to(probs.dtype))
        destination = discrete.argmax(-1)
        continuous = canonical_continuous(continuous, observations, destination, learn_power=self.learn_power)
        return torch.cat((continuous, discrete), -1)

    @torch.no_grad()
    def target_actions(self, observations):
        """Smooth every agent's continuous target action (Ackermann 2019, Eq. 5).

        The original author's TF2 source smooths the current learner's action
        only in a per-agent update; the published equation specifies all
        agents. The categorical branch remains a soft Gumbel sample as declared
        by this UAV hybrid-action adapter.
        """
        a = self.config.matd3
        if not self.share_actor_parameters:
            actions = []
            for i, actor in enumerate(self.target_actors):
                obs = observations[:, i]
                continuous, probs = actor(obs, valid_destinations(obs, self.destinations))
                discrete = self._relaxed_destination(probs)
                destination = discrete.argmax(-1)
                noise = (torch.randn_like(continuous) * a.target_noise_std).clamp(
                    -a.target_noise_clip, a.target_noise_clip)
                continuous = canonical_continuous((continuous + noise).clamp(-1, 1), obs, destination,
                                                  learn_power=self.learn_power)
                actions.append(torch.cat((continuous, discrete), -1))
            return torch.stack(actions, 1)
        continuous, probs = self._policy_outputs(observations, target=True)
        discrete = self._relaxed_destination(probs)
        destination = discrete.argmax(-1)
        noise = (torch.randn_like(continuous) * a.target_noise_std).clamp(-a.target_noise_clip, a.target_noise_clip)
        continuous = canonical_continuous((continuous + noise).clamp(-1, 1), observations, destination,
                                          learn_power=self.learn_power)
        return torch.cat((continuous, discrete), -1)

    @torch.no_grad()
    def act_batch(self, observations, deterministic=False, step=0, *, return_tensors=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        a = self.config.matd3
        if not self.share_actor_parameters:
            actions = []
            for i, actor in enumerate(self.actors):
                local = obs[:, i]
                continuous, probs = actor(local, valid_destinations(local, self.destinations))
                if deterministic:
                    destination = probs.argmax(-1)
                else:
                    continuous = (continuous + torch.randn_like(continuous) * a.exploration_noise_std).clamp(-1, 1)
                    destination = self._relaxed_destination(probs).argmax(-1)
                continuous = canonical_continuous(continuous, local, destination, learn_power=self.learn_power)
                actions.append(joint_action(continuous, destination, self.destinations))
            joint = torch.stack(actions, 1)
        else:
            continuous, probs = self._policy_outputs(obs)
            if deterministic:
                destination = probs.argmax(-1)
            else:
                continuous = (continuous + torch.randn_like(continuous) * a.exploration_noise_std).clamp(-1, 1)
                destination = self._relaxed_destination(probs).argmax(-1)
            continuous = canonical_continuous(continuous, obs, destination, learn_power=self.learn_power)
            joint = joint_action(continuous, destination, self.destinations)
        if return_tensors:
            return Action(joint[..., :self.continuous_dim].contiguous(), joint[..., self.continuous_dim:].argmax(-1))
        values = joint.cpu().numpy()
        return Action(values[..., :self.continuous_dim], values[..., self.continuous_dim:].argmax(-1).astype(np.int64))

    def act(self, observations, deterministic=False, step=0):
        action = self.act_batch(np.asarray(observations)[None], deterministic, step)
        return Action(action.continuous[0], action.destination[0])

    def _critic_values(self, critics, state, actions, vectorized=False, per_critic=False):
        evaluate = getattr(self, "_compiled_critic_values", critic_values)
        return chunked_critic_values(
            lambda s, j: evaluate(critics, s, j, vectorized, per_critic),
            state, actions, self.config.runtime.critic_chunk_size, per_critic)

    def _shared_critic_actor_values(self, state, joints):
        return chunked_critic_values(self._shared_critic_actor_values_full, state, joints,
                                       self.config.runtime.critic_chunk_size, per_critic=True)

    def _shared_critic_actor_values_full(self, state, joints):
        if (getattr(self, "_factorized_critic_enabled", False) and
                not self.config.matd3.critic_batch_norm and self.config.runtime.critic_chunk_size == 0):
            # TD3 actor loss consumes Q1 only. Twin Q remains enabled for Critic Bellman update.
            from .factorized_candidate import matd3_shared_actor_q1
            return matd3_shared_actor_q1(self.critics[0], state, joints)
        """Evaluate one team critic for each agent-specific joint action.

        `joints` is [agent, batch, agent, action]. The single shared critic is
        evaluated in one flattened batch and returned as [batch, agent].
        """
        n, b = joints.shape[:2]
        repeated_state = state[None].expand(n, -1, -1).reshape(n * b, -1)
        q1, q2 = self.critics[0](repeated_state, joints.reshape(n * b, self.num_uavs, -1))
        return (q1.reshape(n, b).transpose(0, 1).contiguous(),
                q2.reshape(n, b).transpose(0, 1).contiguous())

    def _shared_critic_actor_q1_values(self, state, joints):
        # Preserve notebook's OPTIONAL factorized branch if explicitly requested.
        if (getattr(self, "_factorized_critic_enabled", False)
                and not self.config.matd3.critic_batch_norm
                and self.config.runtime.critic_chunk_size == 0):
            return self._shared_critic_actor_values(state, joints)[0]
        return chunked_single_q_values(
            lambda s, j: self.critics[0].forward_shared_state(
                s, j, q1_only=True).transpose(0, 1).contiguous(),
            state, joints, self.config.runtime.critic_chunk_size)

    @autocast_update
    def update_critic(self, batch, collect_metrics=True):
        data, a = tensor_batch(batch, self.device), self.config.matd3
        vectorized = self.config.runtime.vectorize_critics
        with torch.no_grad():
            next_actions = self.target_actions(data["next_observation"])
            target = td3_target(data["reward"][:, None], data["terminated"][:, None],
                                *self._critic_values(self.target_critics, data["next_state"],
                                                     next_actions, vectorized), a.gamma)
        with LayerwiseGradientProbe(
                self.critics, "critic", enabled=(collect_metrics and not self.precision.enabled and
                     os.environ.get("UAV_FAST_TRAINING", "false").lower() not in ("1", "true"))) as layer_probe:
            # Register forward/output/parameter gradient hooks BEFORE the
            # existing online Q1/Q2 forward, then observe the same backward.
            replay_actions = joint_action(data["continuous"], data["destination"], self.destinations,
                                          learn_power=self.learn_power)
            q1, q2 = self._critic_values(self.critics, data["state"], replay_actions, vectorized)
            loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            if a.grad_clip_mode == "per_parameter":
                # Preserve the baseline loss scale and update semantics.
                loss = loss * len(self.critics)
            critic_step = (lambda *args: self.precision.step("critic", *args, grad_clip_mode=a.grad_clip_mode)) if self.precision.enabled else (lambda *args: step_optimizer(*args, grad_clip_mode=a.grad_clip_mode))
            critic_grad = critic_step(loss, self.critic_optimizer,
                                      self.critics.parameters(), a.grad_clip_norm, collect_metrics)
            layer_metrics = layer_probe.metrics()
        batch_metrics = critic_metrics(q1, q2, target, data) if collect_metrics else {}
        self.updates += 1
        # Passive: read already-computed optimizer scalars, no gradients/RNG/Adam changes.
        dense_sink = getattr(self, "_dense_loss_sink", None)
        if dense_sink is not None:
            dense_sink.record("critic_loss", loss)
            dense_sink.record("critic_grad_norm", critic_grad)
            dense_sink.record("td_error_abs_mean", (q1.detach()-target.detach()).abs().mean())
        if not collect_metrics:
            return {}
        return scalar_metrics({**batch_metrics, **layer_metrics, **self._last_actor_metrics,
                               "critic_loss": loss, "critic_grad_norm": critic_grad,
                               "critic_gradient_clipped": (critic_grad > a.grad_clip_norm).float() * (a.grad_clip_mode != "none"),
                               "policy_updated": torch.tensor(0., device=self.device)})

    def _actor_output_regularization(self, observations):
        """One batch-mean author output penalty per decentralized agent."""
        if self.share_actor_parameters:
            b, n = observations.shape[:2]
            flat = observations.reshape(b * n, -1)
            ids = torch.arange(n, device=flat.device).expand(b, n).reshape(-1)
            penalties = self.actors[0].output_regularization(self._actor_input(flat, ids))
            return penalties.reshape(b, n).mean(0)
        return torch.stack([actor.output_regularization(observations[:, i]).mean()
                            for i, actor in enumerate(self.actors)])


    @autocast_update
    def update_actor(self, batch, collect_metrics=True):
        data, a = tensor_batch(batch, self.device), self.config.matd3
        vectorized = self.config.runtime.vectorize_critics
        actor_losses, actor_grads = [], []
        grad_probe = MATD3GradientProbe(
            self.actors, batch_size=len(data["observation"]),
            num_agents=self.num_uavs, motion_dim=self.config.motion_dim,
            continuous_dim=self.continuous_dim, share_actor=self.share_actor_parameters,
            grad_clip=a.grad_clip_norm, grad_clip_mode=a.grad_clip_mode,
            enabled=(collect_metrics and not self.precision.enabled and
                     os.environ.get("UAV_FAST_TRAINING", "false").lower() not in ("1", "true")))
        probe_metrics = {}
        layer_probe = LayerwiseGradientProbe(
            self.actors,"actor",enabled=(collect_metrics and not self.precision.enabled and
                     os.environ.get("UAV_FAST_TRAINING", "false").lower() not in ("1", "true")))
        base = joint_action(data["continuous"], data["destination"], self.destinations,
                            learn_power=self.learn_power).detach()
        self.critics.requires_grad_(False)
        critic_was_training = self.critics.training
        if a.critic_batch_norm:
            # Actor Q-gradient flows through BN in eval mode (affine, fixed
            # running stats) without corrupting replay-critic BN moments.
            self.critics.eval()
        try:
            grad_probe.__enter__()
            layer_probe.__enter__()
            regularization = (self._actor_output_regularization(data["observation"])
                              if a.actor_output_regularization > 0 else None)
            batched_q = None
            candidate_actions = None
            if vectorized or self.share_critic_parameters or self.share_actor_parameters:
                if self.share_actor_parameters:
                    candidate_actions = self._actor_actions(data["observation"], relaxed_discrete=True)
                else:
                    candidate_actions = torch.stack([
                        self._actor_action(actor, data["observation"][:, i],
                                           relaxed_discrete=True, agent_index=i)
                        for i, actor in enumerate(self.actors)], 1)
            if candidate_actions is not None:
                grad_probe.watch_action(candidate_actions)
            if vectorized or self.share_critic_parameters:
                joints = base[None].expand(self.num_uavs, -1, -1, -1).clone()
                indices = torch.arange(self.num_uavs, device=self.device)
                joints[indices, :, indices] = candidate_actions.transpose(0, 1).to(joints.dtype)
                if self.share_critic_parameters:
                    batched_q = self._shared_critic_actor_q1_values(data["state"], joints)
                else:
                    batched_q, _ = self._critic_values(self.critics, data["state"], joints, True, True)
            for i in range(self.num_uavs):
                if batched_q is not None:
                    q1 = batched_q[:, i]
                else:
                    joint = base.clone()
                    own_action = (candidate_actions[:, i] if candidate_actions is not None
                                  else self._actor_action(self.actors[i], data["observation"][:, i],
                                                          relaxed_discrete=True, agent_index=i))
                    grad_probe.watch_action(own_action)
                    joint[:, i] = own_action.to(joint.dtype)
                    q1, _ = chunked_critic_values(self.critics[i], data["state"], joint,
                                                   self.config.runtime.critic_chunk_size)
                actor_loss = -q1.mean()
                if regularization is not None:
                    actor_loss = actor_loss + a.actor_output_regularization * regularization[i]
                actor_losses.append(actor_loss)
            actor_step = (lambda *args: self.precision.step_actors(*args, grad_clip_mode=a.grad_clip_mode)) if self.precision.enabled else (lambda *args: step_actor_optimizers(*args, grad_clip_mode=a.grad_clip_mode))
            if self.share_actor_parameters:
                step_losses = [torch.stack(actor_losses).mean()]
                parameter_groups = [self.actors[0].parameters()]
            else:
                step_losses = actor_losses
                parameter_groups = [actor.parameters() for actor in self.actors]
            actor_grads = actor_step(step_losses, self.actor_optimizers, parameter_groups,
                                     a.grad_clip_norm, collect_metrics)
            actor_losses = [loss.detach() for loss in actor_losses]
            probe_metrics = {**grad_probe.metrics(), **layer_probe.metrics()}
        finally:
            layer_probe.__exit__(None, None, None)
            grad_probe.__exit__(None, None, None)
            self.critics.requires_grad_(True)
            if a.critic_batch_norm and critic_was_training:
                self.critics.train()
        self._last_actor_metrics = {"actor_loss": torch.stack(actor_losses).mean(),
                                    "actor_output_penalty": (a.actor_output_regularization * regularization.detach().mean()
                                                             if regularization is not None else torch.zeros((), device=self.device)),
                                    "actor_grad_norm": torch.stack(actor_grads).mean(),
                                    "actor_update_step": torch.tensor(float(self.updates), device=self.device)}
        if (collect_metrics and os.environ.get("UAV_FAST_TRAINING", "false").lower()
                not in ("1", "true")):
            # These are diagnostics only; do not change the TD3 objective,
            # Actor/target gradient paths, or random sampling.
            self._last_actor_metrics.update(self._actor_output_diagnostics(data["observation"]))
            self._last_actor_metrics.update(self._actor_head_gradient_diagnostics())
            self._last_actor_metrics.update(probe_metrics)
        self.actor_updates += 1
        # Only an ACTUAL delayed MATD3 Actor update contributes to the sample.
        dense_sink = getattr(self, "_dense_loss_sink", None)
        if dense_sink is not None:
            dense_sink.record("actor_loss", self._last_actor_metrics["actor_loss"])
            dense_sink.record("actor_grad_norm", self._last_actor_metrics["actor_grad_norm"])
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
        if actor and self.config.matd3.actor_batch_norm:
            # Unlike LN, BN stores non-parameter running statistics. The old
            # MATD3 Polyak routine only synchronized parameters; without these
            # buffers, target networks use permanently stale BN states.
            for target_actor, source_actor in zip(self.target_actors, self.actors):
                source_buffers = dict(source_actor.named_buffers())
                for name, target_buffer in target_actor.named_buffers():
                    target_buffer.copy_(source_buffers[name])
        if critic and self.config.matd3.critic_batch_norm:
            for target_critic, source_critic in zip(self.target_critics, self.critics):
                source_buffers = dict(source_critic.named_buffers())
                for name, target_buffer in target_critic.named_buffers():
                    target_buffer.copy_(source_buffers[name])

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
