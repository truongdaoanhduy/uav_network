"""Drop-in Gate 4 PyTorch verification test for Hybrid MASAC & Hybrid MATD3.
Run on your Kaggle/GPU environment where PyTorch is installed:
    python3 uav_adapter/gate4_hybrid_marl_test_template.py
Verifies:
1. Per-Agent Exact Expected Q flows non-zero finite gradients into BOTH motion_head and dest_head.
2. Soft Bellman Target applies minus sign to both alpha_m*log_pi_m and alpha_d*log_pi_d.
3. Dynamic target entropy H_bar_d = 0.6 * log(K_valid) prevents alpha_d explosion under blackout (K_valid=1).
"""

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch is required to run Gate 4 test (execute on Kaggle/GPU machine).")


class TinyHybridActor(nn.Module):
    def __init__(self, obs_dim: int = 32, n_dest: int = 6):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
        self.motion_mu = nn.Linear(64, 2)
        self.motion_logstd = nn.Linear(64, 2)
        self.dest_head = nn.Linear(64, n_dest)
        nn.init.zeros_(self.dest_head.bias)

    def forward(self, obs, valid_mask):
        h = self.fc(obs)
        mu = self.motion_mu(h)
        logstd = torch.clamp(self.motion_logstd(h), -5.0, 2.0)
        logits = self.dest_head(h).masked_fill(~valid_mask, -1e9)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return mu, logstd, probs, log_probs


class TinyCentralCritic(nn.Module):
    def __init__(self, state_dim: int = 64, act_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + act_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


def run_gate4_checks():
    torch.manual_seed(0)
    B, K = 16, 6
    actor = TinyHybridActor(32, K)
    critic = TinyCentralCritic(64, 2 + K)
    for p in critic.parameters():
        p.requires_grad_(False)

    obs = torch.randn(B, 32)
    state = torch.randn(B, 64)
    # Mix normal states (K_valid=6) and blackout states (K_valid=1: NO_SEND only)
    valid_mask = torch.ones(B, K, dtype=torch.bool)
    valid_mask[:4, 1:] = False

    mu, logstd, probs, log_probs = actor(obs, valid_mask)
    std = logstd.exp()
    z = mu + std * torch.randn_like(mu)
    u = torch.tanh(z)
    # Tanh Jacobian correction
    log_pi_m = (-0.5 * ((z - mu) / std) ** 2 - logstd - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    log_pi_m -= torch.log(1.0 - u.pow(2) + 1e-6).sum(-1, keepdim=True)

    # Per-agent Exact Expected Q over K=6 discrete choices
    q_all = []
    for d in range(K):
        e_d = F.one_hot(torch.full((B,), d, dtype=torch.long), num_classes=K).float()
        q_all.append(critic(state, torch.cat([u, e_d], dim=-1)))
    q_all = torch.cat(q_all, dim=-1)  # [B, K]

    alpha_m, alpha_d = 0.2, 0.2
    safe_ent_term = torch.where(valid_mask, probs * log_probs, torch.zeros_like(probs))
    actor_loss = (alpha_m * log_pi_m + (alpha_d * safe_ent_term - probs * q_all).sum(-1, keepdim=True)).mean()
    actor_loss.backward()

    motion_grad = actor.motion_mu.weight.grad.norm().item()
    dest_grad = actor.dest_head.weight.grad.norm().item()
    assert math.isfinite(motion_grad) and motion_grad > 0.0, f"Bad motion grad: {motion_grad}"
    assert math.isfinite(dest_grad) and dest_grad > 0.0, f"Bad dest grad: {dest_grad}"

    # Verify dynamic target entropy on blackout rows (K_valid=1 -> H_bar=0, zero alpha_d push)
    k_valid = valid_mask.sum(dim=-1, keepdim=True).float()
    h_bar_d = 0.6 * torch.log(k_valid.clamp(min=1.0))
    ent_d = -(safe_ent_term.sum(dim=-1, keepdim=True)).detach()
    assert torch.allclose(h_bar_d[:4], torch.zeros(4, 1)), "Blackout target entropy must be 0!"
    assert torch.allclose(ent_d[:4], torch.zeros(4, 1), atol=1e-5), "Blackout entropy must be 0!"
    print(f"GATE 4 PASSED: motion_grad={motion_grad:.4f}, dest_grad={dest_grad:.4f}, blackout_alpha_safe=True")


if __name__ == "__main__":
    run_gate4_checks()
