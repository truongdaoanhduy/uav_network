"""Canonical Pure-Continuous MATD3 & MASAC Baselines + Zero-Oracle Event-Driven MAC Trigger.

Strictly implements textbook off-policy CTDE baselines without custom architectural or
hybrid-action modifications:
- CanonicalContinuousMATD3: Fujimoto et al. (ICML 2018) + Ackermann et al. (2019).
  Plain ReLU MLPs, default PyTorch init, pure MSE Bellman loss, target policy smoothing
  (sigma=0.2, clip=0.5), delayed policy update (d=2), Polyak tau=0.005, Gaussian
  exploration noise (sigma=0.1). No LayerNorm, SiLU, Huber, pre_tanh_reg, or Gumbel-Softmax.
- CanonicalContinuousMASAC: Haarnoja et al. (ICML 2018 / arXiv:1812.05905 SAC v2) CTDE.
  Plain ReLU MLPs, default PyTorch init, squashed Gaussian actors (tanh-normal with exact
  log-prob Jacobian), twin soft critics, pure MSE Bellman loss, single automatic entropy
  temperature alpha per agent targeting -dim(A_i) = -2.0. No discrete-SAC / dual-alpha hack.
- event_driven_comm_action_*: Zero-oracle rule-based MAC trigger using only the local
  local observation tail mask [silent, gcs_dirty, peer_dirty] and GCS route-valid
  bit at index 10 (1=GCS > 2=PEER > 0=SILENT).
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from uav_network.algorithms.core.action_types import Action


def event_driven_comm_action_np(obs_or_mask) -> np.ndarray:
    """Zero-oracle event-driven compact_3 trigger from local mask [..., (silent, gcs, peer)].

    Priority: 1 (GCS) if dirty & radio alive, else 2 (PEER) if valid peer route & dirty,
    else 0 (SILENT). Full observations also verify route flag at index 10;
    no oracle state is accessed. For a standalone three-bit mask, its caller
    is responsible for providing a route-feasible GCS bit.
    """
    arr = np.asarray(obs_or_mask)
    mask = arr[..., -3:] if arr.shape[-1] > 3 else arr
    gcs_ok = mask[..., 1] > 0.5
    if arr.shape[-1] > 3:
        gcs_ok = gcs_ok & (arr[..., 10] > 0.5)
    peer_ok = mask[..., 2] > 0.5
    dest = np.zeros(mask.shape[:-1], dtype=np.int64)
    dest = np.where(peer_ok, 2, dest)
    dest = np.where(gcs_ok, 1, dest)
    return dest


def event_driven_comm_action_torch(obs_or_mask):
    """Torch counterpart of `event_driven_comm_action_np` (runs on-device, zero sync)."""
    import torch

    mask = obs_or_mask[..., -3:] if obs_or_mask.shape[-1] > 3 else obs_or_mask
    gcs_ok = mask[..., 1] > 0.5
    if obs_or_mask.shape[-1] > 3:
        gcs_ok = gcs_ok & (obs_or_mask[..., 10] > 0.5)
    peer_ok = mask[..., 2] > 0.5
    dest = torch.zeros(mask.shape[:-1], dtype=torch.long, device=mask.device)
    dest = torch.where(peer_ok, torch.full_like(dest, 2), dest)
    dest = torch.where(gcs_ok, torch.ones_like(dest), dest)
    return dest


def _plain_relu_mlp(in_dim: int, hidden_sizes, out_dim: int):
    from torch import nn

    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(nn.ReLU())
        prev = int(h)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _assert_canonical_config(config, alg: str) -> None:
    hp = getattr(config, alg)
    if config.continuous_dim != 2 or config.network.power_control != "fixed":
        raise ValueError("Canonical continuous baseline requires power_control='fixed' and continuous_dim=2")
    if hp.activation != "relu":
        raise ValueError(f"Canonical {alg} baseline requires activation='relu', got {hp.activation!r}")
    if hp.actor_layer_norm or hp.critic_layer_norm or hp.actor_batch_norm or hp.critic_batch_norm:
        raise ValueError(f"Canonical {alg} baseline forbids LayerNorm/BatchNorm")
    if hp.grad_clip_norm != 0.0:
        raise ValueError(f"Canonical {alg} baseline forbids grad_clip_norm (expected 0.0)")
    if hp.critic_huber_delta != 0.0:
        raise ValueError(f"Canonical {alg} baseline requires MSE loss (critic_huber_delta=0.0)")
    if hp.sync_critic_loss_scale:
        raise ValueError(f"Canonical {alg} baseline requires sync_critic_loss_scale=False")
    if alg == "matd3":
        if hp.pre_tanh_reg != 0.0 or hp.actor_output_regularization != 0.0:
            raise ValueError("Canonical MATD3 baseline requires pre_tanh_reg=0.0 and actor_output_regularization=0.0")
        if hp.initialization != "default":
            raise ValueError(f"Canonical MATD3 baseline requires default PyTorch init, got {hp.initialization!r}")


class CanonicalContinuousMATD3:
    """Textbook MATD3 (Fujimoto ICML 2018 + Ackermann 2019) for 2D continuous UAV motion."""

    algorithm = "matd3"
    baseline_mode = "canonical_continuous"

    def __init__(self, config, observation_dim: int, state_dim: int, device: str = "cpu"):
        import torch
        from torch import nn

        _assert_canonical_config(config, "matd3")
        self.config = deepcopy(config)
        self.device = torch.device(device)
        self.num_uavs = n = int(config.world.num_uavs)
        self.obs_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.act_dim = 2
        a = config.matd3
        self.gamma = float(a.gamma)
        self.tau = float(a.tau)
        self.policy_delay = int(a.policy_delay)
        self.target_noise_std = float(a.target_noise_std)
        self.target_noise_clip = float(a.target_noise_clip)
        self.exploration_noise_std = float(a.exploration_noise_std)
        self.grad_clip_norm = float(a.grad_clip_norm)

        self.actors = nn.ModuleList(
            _plain_relu_mlp(self.obs_dim, a.hidden_sizes, self.act_dim) for _ in range(n)
        ).to(self.device)
        q_in = self.state_dim + n * self.act_dim
        self.critics_q1 = nn.ModuleList(
            _plain_relu_mlp(q_in, a.hidden_sizes, 1) for _ in range(n)
        ).to(self.device)
        self.critics_q2 = nn.ModuleList(
            _plain_relu_mlp(q_in, a.hidden_sizes, 1) for _ in range(n)
        ).to(self.device)
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_q1 = deepcopy(self.critics_q1).requires_grad_(False)
        self.target_q2 = deepcopy(self.critics_q2).requires_grad_(False)

        self.actor_opts = [torch.optim.Adam(m.parameters(), lr=a.learning_rate) for m in self.actors]
        self.critic_opts = [
            torch.optim.Adam(list(self.critics_q1[i].parameters()) + list(self.critics_q2[i].parameters()),
                             lr=a.learning_rate)
            for i in range(n)
        ]
        self.updates = 0

    def parameters(self):
        for group in (self.actors, self.critics_q1, self.critics_q2):
            yield from group.parameters()

    def act_batch(self, observations, deterministic: bool = False, step: int = 0, *, return_tensors: bool = False):
        import torch

        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            active = (obs[..., 9] > 0.5).to(obs.dtype).unsqueeze(-1)
            acts = [torch.tanh(self.actors[i](obs[:, i])) for i in range(self.num_uavs)]
            cont = torch.stack(acts, dim=1)
            if not deterministic:
                cont = (cont + torch.randn_like(cont) * self.exploration_noise_std).clamp(-1.0, 1.0)
            cont = (cont * active).contiguous()
            dest = event_driven_comm_action_torch(obs)
            if return_tensors:
                return Action(cont, dest)
            return Action(cont.cpu().numpy(), dest.cpu().numpy().astype(np.int64))

    def update(self, batch):
        import torch
        from torch.nn import functional as F

        self.updates += 1
        obs = torch.as_tensor(batch["observation"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_observation"], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        act_cont = torch.as_tensor(batch["continuous"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        term = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        bsz = state.shape[0]

        with torch.no_grad():
            next_acts = []
            for i in range(self.num_uavs):
                mu = torch.tanh(self.target_actors[i](next_obs[:, i]))
                noise = (torch.randn_like(mu) * self.target_noise_std).clamp(
                    -self.target_noise_clip, self.target_noise_clip)
                alive = (next_obs[:, i, 9:10] > 0.5).to(mu.dtype)
                next_acts.append((mu + noise).clamp(-1.0, 1.0) * alive)
            next_joint = torch.cat(next_acts, dim=-1)
            next_in = torch.cat([next_state, next_joint], dim=-1)

        cur_in = torch.cat([state, act_cont.reshape(bsz, -1)], dim=-1)
        c_losses = []
        for i in range(self.num_uavs):
            with torch.no_grad():
                tq = torch.minimum(self.target_q1[i](next_in), self.target_q2[i](next_in)).squeeze(-1)
                y = rew + self.gamma * (1.0 - term) * tq
            q1 = self.critics_q1[i](cur_in).squeeze(-1)
            q2 = self.critics_q2[i](cur_in).squeeze(-1)
            loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opts[i].zero_grad(set_to_none=True)
            loss_c.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critics_q1[i].parameters()) + list(self.critics_q2[i].parameters()),
                    self.grad_clip_norm,
                )
            self.critic_opts[i].step()
            c_losses.append(float(loss_c.item()))

        metrics = {"critic_loss": float(sum(c_losses) / len(c_losses))}
        if self.updates % self.policy_delay == 0:
            a_losses = []
            for i in range(self.num_uavs):
                self.critics_q1[i].requires_grad_(False)
                joint_i = act_cont.clone()
                joint_i[:, i] = torch.tanh(self.actors[i](obs[:, i]))
                q_pi = self.critics_q1[i](torch.cat([state, joint_i.reshape(bsz, -1)], dim=-1)).squeeze(-1)
                loss_a = -q_pi.mean()
                self.actor_opts[i].zero_grad(set_to_none=True)
                loss_a.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip_norm)
                self.actor_opts[i].step()
                self.critics_q1[i].requires_grad_(True)
                a_losses.append(float(loss_a.item()))
            metrics["actor_loss"] = float(sum(a_losses) / len(a_losses))
            with torch.no_grad():
                for src, dst in ((self.actors, self.target_actors),
                                 (self.critics_q1, self.target_q1),
                                 (self.critics_q2, self.target_q2)):
                    for ps, pt in zip(src.parameters(), dst.parameters()):
                        pt.mul_(1.0 - self.tau).add_(ps, alpha=self.tau)
        return metrics


class CanonicalContinuousMASAC:
    """Textbook CTDE MASAC (Haarnoja ICML 2018 / 2019 SAC v2) for 2D continuous UAV motion."""

    algorithm = "masac"
    baseline_mode = "canonical_continuous"

    def __init__(self, config, observation_dim: int, state_dim: int, device: str = "cpu"):
        import torch
        from torch import nn

        _assert_canonical_config(config, "masac")
        self.config = deepcopy(config)
        self.device = torch.device(device)
        self.num_uavs = n = int(config.world.num_uavs)
        self.obs_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.act_dim = 2
        a = config.masac
        self.gamma = float(a.gamma)
        self.tau = float(a.tau)
        self.grad_clip_norm = float(a.grad_clip_norm)
        self.target_entropy = float(a.target_continuous_entropy)

        self.actors = nn.ModuleList(
            _plain_relu_mlp(self.obs_dim, a.hidden_sizes, 2 * self.act_dim) for _ in range(n)
        ).to(self.device)
        q_in = self.state_dim + n * self.act_dim
        self.critics_q1 = nn.ModuleList(_plain_relu_mlp(q_in, a.hidden_sizes, 1) for _ in range(n)).to(self.device)
        self.critics_q2 = nn.ModuleList(_plain_relu_mlp(q_in, a.hidden_sizes, 1) for _ in range(n)).to(self.device)
        self.target_q1 = deepcopy(self.critics_q1).requires_grad_(False)
        self.target_q2 = deepcopy(self.critics_q2).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.full((n,), math.log(float(a.initial_alpha)), device=self.device))

        self.actor_opts = [torch.optim.Adam(m.parameters(), lr=a.learning_rate) for m in self.actors]
        self.critic_opts = [
            torch.optim.Adam(list(self.critics_q1[i].parameters()) + list(self.critics_q2[i].parameters()),
                             lr=a.learning_rate)
            for i in range(n)
        ]
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=a.learning_rate)
        self.updates = 0

    def parameters(self):
        for group in (self.actors, self.critics_q1, self.critics_q2):
            yield from group.parameters()
        yield self.log_alpha

    def _sample_agent(self, i: int, obs_i, deterministic: bool = False):
        import torch
        from torch.distributions import Normal
        from torch.nn import functional as F

        out = self.actors[i](obs_i)
        mean, log_std = out[..., :self.act_dim], out[..., self.act_dim:].clamp(-20.0, 2.0)
        std = log_std.exp()
        dist = Normal(mean, std, validate_args=False)
        raw = mean if deterministic else dist.rsample()
        act = torch.tanh(raw)
        log_jac = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_prob = (dist.log_prob(raw) - log_jac).sum(dim=-1)
        return act, log_prob

    def act_batch(self, observations, deterministic: bool = False, step: int = 0, *, return_tensors: bool = False):
        import torch

        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            active = (obs[..., 9] > 0.5).to(obs.dtype).unsqueeze(-1)
            acts = [self._sample_agent(i, obs[:, i], deterministic=deterministic)[0] for i in range(self.num_uavs)]
            cont = (torch.stack(acts, dim=1) * active).contiguous()
            dest = event_driven_comm_action_torch(obs)
            if return_tensors:
                return Action(cont, dest)
            return Action(cont.cpu().numpy(), dest.cpu().numpy().astype(np.int64))

    def update(self, batch):
        import torch
        from torch.nn import functional as F

        self.updates += 1
        obs = torch.as_tensor(batch["observation"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_observation"], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        act_cont = torch.as_tensor(batch["continuous"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        term = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        bsz = state.shape[0]
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_acts, next_logps = [], []
            for i in range(self.num_uavs):
                a_i, lp_i = self._sample_agent(i, next_obs[:, i], deterministic=False)
                alive = (next_obs[:, i, 9:10] > 0.5).to(a_i.dtype)
                next_acts.append(a_i * alive)
                next_logps.append(lp_i * alive.squeeze(-1))
            next_in = torch.cat([next_state, torch.cat(next_acts, dim=-1)], dim=-1)

        cur_in = torch.cat([state, act_cont.reshape(bsz, -1)], dim=-1)
        c_losses, a_losses, logp_list = [], [], []
        for i in range(self.num_uavs):
            with torch.no_grad():
                tq = torch.minimum(self.target_q1[i](next_in), self.target_q2[i](next_in)).squeeze(-1)
                y = rew + self.gamma * (1.0 - term) * (tq - alpha[i] * next_logps[i])
            q1 = self.critics_q1[i](cur_in).squeeze(-1)
            q2 = self.critics_q2[i](cur_in).squeeze(-1)
            loss_c = 0.5 * (F.mse_loss(q1, y) + F.mse_loss(q2, y))
            self.critic_opts[i].zero_grad(set_to_none=True)
            loss_c.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.critics_q1[i].parameters()) + list(self.critics_q2[i].parameters()),
                    self.grad_clip_norm,
                )
            self.critic_opts[i].step()
            c_losses.append(float(loss_c.item()))

        for i in range(self.num_uavs):
            self.critics_q1[i].requires_grad_(False)
            self.critics_q2[i].requires_grad_(False)
            a_i, lp_i = self._sample_agent(i, obs[:, i], deterministic=False)
            joint_i = act_cont.clone()
            joint_i[:, i] = a_i
            q_in_i = torch.cat([state, joint_i.reshape(bsz, -1)], dim=-1)
            q_pi = torch.minimum(self.critics_q1[i](q_in_i), self.critics_q2[i](q_in_i)).squeeze(-1)
            loss_a = (alpha[i] * lp_i - q_pi).mean()
            self.actor_opts[i].zero_grad(set_to_none=True)
            loss_a.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.grad_clip_norm)
            self.actor_opts[i].step()
            self.critics_q1[i].requires_grad_(True)
            self.critics_q2[i].requires_grad_(True)
            a_losses.append(float(loss_a.item()))
            logp_list.append(lp_i.detach())

        alpha_loss = sum(-(self.log_alpha[i] * (logp_list[i] + self.target_entropy)).mean()
                         for i in range(self.num_uavs))
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for src, dst in ((self.critics_q1, self.target_q1), (self.critics_q2, self.target_q2)):
                for ps, pt in zip(src.parameters(), dst.parameters()):
                    pt.mul_(1.0 - self.tau).add_(ps, alpha=self.tau)

        return {
            "critic_loss": float(sum(c_losses) / len(c_losses)),
            "actor_loss": float(sum(a_losses) / len(a_losses)),
            "alpha_mean": float(self.log_alpha.exp().mean().item()),
        }
