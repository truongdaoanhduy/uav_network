"""Canonical Vanilla MATD3 & Vanilla MASAC Baselines + GPU-Resident FP16 Replay Buffer.

Paper & Library Provenance:
- Vanilla MATD3: Ackermann et al. (2019, arXiv:1910.01465), Fujimoto et al. (TD3, ICML 2018),
  Lowe et al. (MADDPG, NeurIPS 2017). Plain MLP [256, 256] ReLU, no BatchNorm/LayerNorm,
  twin centralized critics, policy delay d=2, target smoothing on continuous actions only,
  masked Straight-Through Gumbel-Softmax on discrete destination head.
- Vanilla MASAC: Haarnoja et al. (SAC v2, arXiv:1812.05905), Bettini et al. (BenchMARL, JMLR 2024).
  Plain MLP [256, 256] ReLU, TanhNormal (stable log-prob) + MaskedCategorical, dual alpha_c/alpha_d
  with dynamic discrete target entropy 0.5 * ln(max(K_valid, 1)), masked alpha_d loss over
  multi-option states (valid_counts > 1), clamped log_alpha [-10, 2], ST-Gumbel-Softmax.
- Parameter Sharing: Homogeneous Shared Actor + One-Hot Agent ID (BenchMARL / EPyMARL / Terry 2020)
  and Shared Twin Centralized Critic for team-reward 6-UAV cooperative search & relay.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_FILL_VALUE = -1e9
LOG_ALPHA_MIN = -10.0
LOG_ALPHA_MAX = 2.0


def build_plain_mlp(input_dim: int, hidden_sizes: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Canonical plain MLP with ReLU activations — strictly NO BatchNorm or LayerNorm."""
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def append_one_hot_agent_id(obs: torch.Tensor) -> torch.Tensor:
    """Concatenate one-hot agent ID (N dims) to local observations: (B, N, O) -> (B, N, O+N)."""
    b, n, _ = obs.shape
    eye = torch.eye(n, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(b, n, n)
    return torch.cat([obs, eye], dim=-1)


def masked_st_gumbel_softmax(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    tau: float = 1.0,
    add_gumbel_noise: bool = True,
) -> torch.Tensor:
    """FP32 Masked Straight-Through Gumbel-Softmax (mask applied AFTER Gumbel noise)."""
    logits_f32 = logits.float()
    mask_bool = action_mask.bool()
    if add_gumbel_noise:
        u = torch.rand_like(logits_f32).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        scaled = (logits_f32 + gumbel) / max(tau, 1e-4)
    else:
        scaled = logits_f32 / max(tau, 1e-4)
    masked_logits = scaled.masked_fill(~mask_bool, MASK_FILL_VALUE)
    y_soft = F.softmax(masked_logits, dim=-1)
    idx = masked_logits.argmax(dim=-1)
    y_hard = F.one_hot(idx, num_classes=logits_f32.shape[-1]).float()
    return y_hard - y_soft.detach() + y_soft


class GPUHybridReplayBuffer:
    """100% GPU-resident FP16/FP32 Replay Buffer (~3.4 GiB at 1M steps on Tesla T4)."""

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        cont_dim: int,
        disc_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.empty((capacity, num_agents, obs_dim), dtype=torch.float16, device=device)
        self.next_obs = torch.empty((capacity, num_agents, obs_dim), dtype=torch.float16, device=device)
        self.state = torch.empty((capacity, state_dim), dtype=torch.float16, device=device)
        self.next_state = torch.empty((capacity, state_dim), dtype=torch.float16, device=device)
        self.act_cont = torch.empty((capacity, num_agents, cont_dim), dtype=torch.float32, device=device)
        self.act_disc = torch.empty((capacity, num_agents, disc_dim), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.mask = torch.empty((capacity, num_agents, disc_dim), dtype=torch.uint8, device=device)
        self.next_mask = torch.empty((capacity, num_agents, disc_dim), dtype=torch.uint8, device=device)

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        act_cont: torch.Tensor,
        act_disc_onehot: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_state: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
        next_mask: torch.Tensor,
    ) -> None:
        """Contiguous slice-copy insertion (zero scatter-index allocation)."""
        batch = int(obs.shape[0])
        if batch <= 0:
            return
        obs_d = obs.to(self.device, dtype=torch.float16, non_blocking=True)
        next_obs_d = next_obs.to(self.device, dtype=torch.float16, non_blocking=True)
        state_d = state.to(self.device, dtype=torch.float16, non_blocking=True)
        next_state_d = next_state.to(self.device, dtype=torch.float16, non_blocking=True)
        ac_d = act_cont.to(self.device, dtype=torch.float32, non_blocking=True)
        ad_d = act_disc_onehot.to(self.device, dtype=torch.float32, non_blocking=True)
        rew_d = rewards.view(batch, 1).to(self.device, dtype=torch.float32, non_blocking=True)
        done_d = dones.view(batch, 1).to(self.device, dtype=torch.float32, non_blocking=True)
        mask_d = mask.to(self.device, dtype=torch.uint8, non_blocking=True)
        next_mask_d = next_mask.to(self.device, dtype=torch.uint8, non_blocking=True)

        end = self.ptr + batch
        pairs = (
            (self.obs, obs_d),
            (self.next_obs, next_obs_d),
            (self.state, state_d),
            (self.next_state, next_state_d),
            (self.act_cont, ac_d),
            (self.act_disc, ad_d),
            (self.rewards, rew_d),
            (self.dones, done_d),
            (self.mask, mask_d),
            (self.next_mask, next_mask_d),
        )
        if end <= self.capacity:
            s = slice(self.ptr, end)
            for dst, src in pairs:
                dst[s].copy_(src, non_blocking=True)
        else:
            n1 = self.capacity - self.ptr
            n2 = batch - n1
            for dst, src in pairs:
                dst[self.ptr :].copy_(src[:n1], non_blocking=True)
                dst[:n2].copy_(src[n1:], non_blocking=True)

        self.ptr = end % self.capacity
        self.size = min(self.size + batch, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx].float(),
            "next_obs": self.next_obs[idx].float(),
            "state": self.state[idx].float(),
            "next_state": self.next_state[idx].float(),
            "act_cont": self.act_cont[idx],
            "act_disc": self.act_disc[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "mask": self.mask[idx].bool(),
            "next_mask": self.next_mask[idx].bool(),
        }

    def vram_bytes(self) -> int:
        tensors = (
            self.obs, self.next_obs, self.state, self.next_state,
            self.act_cont, self.act_disc, self.rewards, self.dones,
            self.mask, self.next_mask,
        )
        return sum(t.element_size() * t.nelement() for t in tensors)


class SharedTwinCentralizedCritic(nn.Module):
    """Canonical Twin Centralized Critic Q1(s, a_1..N), Q2(s, a_1..N) with plain MLP [256, 256]."""

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        cont_dim: int,
        disc_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        in_dim = state_dim + num_agents * (cont_dim + disc_dim)
        self.q1 = build_plain_mlp(in_dim, hidden_sizes, 1)
        self.q2 = build_plain_mlp(in_dim, hidden_sizes, 1)

    def _pack(self, state: torch.Tensor, act_cont: torch.Tensor, act_disc: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        return torch.cat([state, act_cont.reshape(b, -1), act_disc.reshape(b, -1)], dim=-1)

    def forward(
        self, state: torch.Tensor, act_cont: torch.Tensor, act_disc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._pack(state, act_cont, act_disc)
        return self.q1(x), self.q2(x)

    def q1_only(self, state: torch.Tensor, act_cont: torch.Tensor, act_disc: torch.Tensor) -> torch.Tensor:
        return self.q1(self._pack(state, act_cont, act_disc))


class VanillaDeterministicHybridActor(nn.Module):
    """Canonical MATD3 Shared Actor (Plain MLP [256, 256] ReLU, tanh continuous + logits discrete)."""

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        cont_dim: int,
        disc_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.backbone = build_plain_mlp(obs_dim + num_agents, hidden_sizes[:-1], hidden_sizes[-1])
        self.cont_head = nn.Linear(hidden_sizes[-1], cont_dim)
        self.disc_head = nn.Linear(hidden_sizes[-1], disc_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.backbone(append_one_hot_agent_id(obs)))
        return torch.tanh(self.cont_head(h)), self.disc_head(h)


class VanillaMATD3:
    """Canonical Vanilla MATD3 (Shared Actor + One-Hot ID, Shared Twin Centralized Critic)."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        cont_dim: int = 2,
        disc_dim: int = 7,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        exploration_noise_std: float = 0.1,
        target_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        gumbel_tau: float = 1.0,
        grad_clip_norm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.exploration_noise_std = exploration_noise_std
        self.target_noise_std = target_noise_std
        self.target_noise_clip = target_noise_clip
        self.gumbel_tau = gumbel_tau
        self.grad_clip_norm = grad_clip_norm
        self.total_updates = 0

        self.actor = VanillaDeterministicHybridActor(obs_dim, num_agents, cont_dim, disc_dim, hidden_sizes).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = SharedTwinCentralizedCritic(state_dim, num_agents, cont_dim, disc_dim, hidden_sizes).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, action_mask: torch.Tensor, explore: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_cont, logits_disc = self.actor(obs)
        if explore:
            mu_cont = (mu_cont + torch.randn_like(mu_cont) * self.exploration_noise_std).clamp(-1.0, 1.0)
            disc_oh = masked_st_gumbel_softmax(logits_disc, action_mask, tau=self.gumbel_tau, add_gumbel_noise=True)
        else:
            masked_logits = logits_disc.masked_fill(~action_mask.bool(), MASK_FILL_VALUE)
            disc_oh = F.one_hot(masked_logits.argmax(dim=-1), num_classes=logits_disc.shape[-1]).float()
        return mu_cont, disc_oh

    def update(self, batch: Dict[str, torch.Tensor], return_metrics: bool = True) -> Dict[str, float]:
        self.total_updates += 1
        obs, next_obs = batch["obs"], batch["next_obs"]
        state, next_state = batch["state"], batch["next_state"]
        act_cont, act_disc = batch["act_cont"], batch["act_disc"]
        rewards, dones = batch["rewards"], batch["dones"]
        mask, next_mask = batch["mask"], batch["next_mask"]

        with torch.no_grad():
            next_cont, next_logits = self.target_actor(next_obs)
            noise = (torch.randn_like(next_cont) * self.target_noise_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_cont = (next_cont + noise).clamp(-1.0, 1.0)
            masked_next_logits = next_logits.masked_fill(~next_mask, MASK_FILL_VALUE)
            next_disc = F.one_hot(masked_next_logits.argmax(dim=-1), num_classes=next_logits.shape[-1]).float()
            tq1, tq2 = self.target_critic(next_state, next_cont, next_disc)
            target_q = rewards + self.gamma * (1.0 - dones) * torch.minimum(tq1, tq2)

        q1, q2 = self.critic(state, act_cont, act_disc)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_opt.step()

        actor_loss_t = None
        if self.total_updates % self.policy_delay == 0:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            pred_cont, pred_logits = self.actor(obs)
            pred_disc = masked_st_gumbel_softmax(pred_logits, mask, tau=self.gumbel_tau, add_gumbel_noise=True)
            actor_loss_t = -self.critic.q1_only(state, pred_cont, pred_disc).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss_t.backward()
            if self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_opt.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)

            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic, self.target_critic)

        if not return_metrics:
            return {}
        metrics = {"critic_loss": float(critic_loss.item()), "q1_mean": float(q1.mean().item())}
        if actor_loss_t is not None:
            metrics["actor_loss"] = float(actor_loss_t.item())
        return metrics

    @torch.no_grad()
    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            p_dst.data.mul_(1.0 - self.tau).add_(p_src.data, alpha=self.tau)


class VanillaStochasticHybridActor(nn.Module):
    """Canonical MASAC Shared Actor (Plain MLP [256, 256] ReLU, TanhNormal + MaskedCategorical)."""

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        cont_dim: int,
        disc_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = build_plain_mlp(obs_dim + num_agents, hidden_sizes[:-1], hidden_sizes[-1])
        self.mean_head = nn.Linear(hidden_sizes[-1], cont_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], cont_dim)
        self.disc_head = nn.Linear(hidden_sizes[-1], disc_dim)

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor, gumbel_tau: float = 1.0, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.backbone(append_one_hot_agent_id(obs)))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        logits_disc = self.disc_head(h)

        if deterministic:
            u = mean
            act_cont = torch.tanh(u)
            log_pi_cont = torch.zeros(obs.shape[:2], device=obs.device)
            masked_logits = logits_disc.masked_fill(~action_mask.bool(), MASK_FILL_VALUE)
            act_disc = F.one_hot(masked_logits.argmax(dim=-1), num_classes=logits_disc.shape[-1]).float()
            entropy_disc = torch.zeros(obs.shape[:2], device=obs.device)
            return act_cont, act_disc, log_pi_cont, entropy_disc

        # Reparameterized TanhNormal with numerically stable Jacobian correction
        eps = torch.randn_like(mean)
        u = mean + std * eps
        act_cont = torch.tanh(u)
        log_prob_gauss = -0.5 * (eps.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        squash_corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_pi_cont = (log_prob_gauss - squash_corr).sum(dim=-1)

        # ST-Gumbel-Softmax for Q gradient + Exact Shannon Entropy for low-variance alpha_d
        act_disc = masked_st_gumbel_softmax(logits_disc, action_mask, tau=gumbel_tau, add_gumbel_noise=True)
        masked_logits = logits_disc.float().masked_fill(~action_mask.bool(), MASK_FILL_VALUE)
        probs = F.softmax(masked_logits, dim=-1)
        entropy_disc = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        return act_cont, act_disc, log_pi_cont, entropy_disc


class VanillaMASAC:
    """Canonical Vanilla CTDE MASAC (Shared Actor + One-Hot ID, Shared Twin Soft Critic, dual alpha)."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        cont_dim: int = 2,
        disc_dim: int = 7,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_alpha_cont: float = 0.2,
        initial_alpha_disc: float = 0.2,
        target_entropy_cont: float | None = None,
        target_entropy_disc_ratio: float = 0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        gumbel_tau: float = 1.0,
        grad_clip_norm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.grad_clip_norm = grad_clip_norm
        self.target_entropy_cont = float(-cont_dim if target_entropy_cont is None else target_entropy_cont)
        self.target_entropy_disc_ratio = float(target_entropy_disc_ratio)
        self.total_updates = 0

        self.actor = VanillaStochasticHybridActor(
            obs_dim, num_agents, cont_dim, disc_dim, hidden_sizes,
            log_std_min=log_std_min, log_std_max=log_std_max,
        ).to(self.device)
        self.critic = SharedTwinCentralizedCritic(state_dim, num_agents, cont_dim, disc_dim, hidden_sizes).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.log_alpha_c = nn.Parameter(torch.tensor(math.log(initial_alpha_cont), device=self.device))
        self.log_alpha_d = nn.Parameter(torch.tensor(math.log(initial_alpha_disc), device=self.device))

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha_c, self.log_alpha_d], lr=lr)

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, action_mask: torch.Tensor, explore: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act_cont, act_disc, _, _ = self.actor(obs, action_mask, gumbel_tau=self.gumbel_tau, deterministic=not explore)
        return act_cont, act_disc

    def update(self, batch: Dict[str, torch.Tensor], return_metrics: bool = True) -> Dict[str, float]:
        self.total_updates += 1
        obs, next_obs = batch["obs"], batch["next_obs"]
        state, next_state = batch["state"], batch["next_state"]
        act_cont, act_disc = batch["act_cont"], batch["act_disc"]
        rewards, dones = batch["rewards"], batch["dones"]
        mask, next_mask = batch["mask"], batch["next_mask"]

        alpha_c = self.log_alpha_c.exp().detach()
        alpha_d = self.log_alpha_d.exp().detach()

        with torch.no_grad():
            n_cont, n_disc, n_log_pi_c, n_ent_d = self.actor(next_obs, next_mask, gumbel_tau=self.gumbel_tau)
            tq1, tq2 = self.target_critic(next_state, n_cont, n_disc)
            entropy_term = (alpha_c * n_log_pi_c - alpha_d * n_ent_d).mean(dim=-1, keepdim=True)
            target_q = rewards + self.gamma * (1.0 - dones) * (torch.minimum(tq1, tq2) - entropy_term)

        q1, q2 = self.critic(state, act_cont, act_disc)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        p_cont, p_disc, log_pi_c, ent_d = self.actor(obs, mask, gumbel_tau=self.gumbel_tau)
        q1_pi, q2_pi = self.critic(state, p_cont, p_disc)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_entropy = (alpha_c * log_pi_c - alpha_d * ent_d).mean(dim=-1, keepdim=True)
        actor_loss = (actor_entropy - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # Dual alpha update: mask discrete alpha loss strictly over multi-option states (valid_counts > 1)
        valid_counts = mask.float().sum(dim=-1).clamp_min(1.0)
        target_h_d = self.target_entropy_disc_ratio * torch.log(valid_counts)
        alpha_c_loss = -(self.log_alpha_c * (log_pi_c.detach() + self.target_entropy_cont)).mean()
        multi_opt = (valid_counts > 1.0).float()
        alpha_d_loss = -(self.log_alpha_d * (target_h_d - ent_d.detach()) * multi_opt).sum() / multi_opt.sum().clamp_min(1.0)
        alpha_loss = alpha_c_loss + alpha_d_loss

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha_c.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            self.log_alpha_d.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            for p_src, p_dst in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_dst.data.mul_(1.0 - self.tau).add_(p_src.data, alpha=self.tau)

        if not return_metrics:
            return {}
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_c": float(self.log_alpha_c.exp().item()),
            "alpha_d": float(self.log_alpha_d.exp().item()),
            "q1_mean": float(q1.mean().item()),
        }
