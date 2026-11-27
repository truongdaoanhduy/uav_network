"""Unchanged vanilla stochastic actor and MASAC, moved into a dedicated module."""
from __future__ import annotations
import copy
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_network.algorithms.baselines.vanilla import (
    MASK_FILL_VALUE, LOG_ALPHA_MIN, LOG_ALPHA_MAX, build_plain_mlp,
    append_one_hot_agent_id, masked_st_gumbel_softmax, GPUHybridReplayBuffer,
    SharedTwinCentralizedCritic, VanillaDeterministicHybridActor, VanillaMATD3,
)

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

        use_fused = self.device.type == 'cuda'
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, fused=use_fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, fused=use_fused)
        self.alpha_opt = torch.optim.Adam([self.log_alpha_c, self.log_alpha_d], lr=lr, fused=use_fused)

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
            src_p = [p.data for p in self.critic.parameters()]
            dst_p = [p.data for p in self.target_critic.parameters()]
            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(dst_p, src_p, self.tau)
            else:
                for ps, pd in zip(src_p, dst_p):
                    pd.mul_(1.0 - self.tau).add_(ps, alpha=self.tau)

        if not return_metrics:
            return {}
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_c": float(self.log_alpha_c.exp().item()),
            "alpha_d": float(self.log_alpha_d.exp().item()),
            "q1_mean": float(q1.mean().item()),
        }
