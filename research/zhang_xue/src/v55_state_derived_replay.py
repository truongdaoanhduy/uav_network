"""Memory-bounded MARL replay for states that are exactly concatenated actor observations.

Drops duplicate (state, next_state) storage and reconstructs on sampling.
Only valid for ZhangXue7NodePacketEnv; NOT a general CTDE replay replacement.

v5.7 fix (B16): Lazy-import `torch` inside StateDerivedReplayBuffer so `estimate_replay_bytes`
and memory-budget unit tests run cleanly in pure-Python/NumPy environments without PyTorch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def estimate_replay_bytes(
    capacity: int,
    num_agents: int,
    obs_dim: int,
    cont_dim: int,
    disc_dim: int,
    store_state: bool = False,
) -> int:
    if min(capacity, num_agents, obs_dim, cont_dim, disc_dim) <= 0:
        raise ValueError("all dimensions must be positive")
    b = (
        2 * num_agents * obs_dim * 2
        + num_agents * cont_dim * 4
        + num_agents * disc_dim * 4
        + 2 * num_agents * disc_dim
        + 8
    )
    if store_state:
        b += 2 * num_agents * obs_dim * 2
    return b * capacity


class StateDerivedReplayBuffer:
    """Compatible add_batch/sample interface with VanillaMATD3/VanillaMASAC pilot.

    Constraints: caller guarantees state = obs.reshape(B,-1); this is validated
    on the first small batch, with low overhead for subsequent GPU inserts.
    """

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        cont_dim: int,
        disc_dim: int,
        device: Any,
    ) -> None:
        import torch

        self._torch = torch
        if state_dim != num_agents * obs_dim:
            raise ValueError("state must be the concatenation of per-UAV observations")
        if min(capacity, num_agents, obs_dim, cont_dim, disc_dim) <= 0:
            raise ValueError("invalid dimensions")
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self.ptr = 0
        self.size = 0
        self._validated = False
        self.obs = torch.empty((capacity, num_agents, obs_dim), dtype=torch.float16, device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.act_cont = torch.empty((capacity, num_agents, cont_dim), dtype=torch.float32, device=device)
        self.act_disc = torch.empty((capacity, num_agents, disc_dim), dtype=torch.float32, device=device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.mask = torch.empty((capacity, num_agents, disc_dim), dtype=torch.uint8, device=device)
        self.next_mask = torch.empty_like(self.mask)
        self.estimated_bytes = estimate_replay_bytes(capacity, num_agents, obs_dim, cont_dim, disc_dim, False)

    def add_batch(
        self,
        obs: Any,
        state: Any,
        act_cont: Any,
        act_disc_onehot: Any,
        rewards: Any,
        next_obs: Any,
        next_state: Any,
        dones: Any,
        mask: Any,
        next_mask: Any,
    ) -> None:
        torch = self._torch
        with torch.no_grad():
            b = int(obs.shape[0])
            if not b:
                return
            if b > self.capacity:
                raise ValueError("one insertion cannot exceed replay capacity")
            if not self._validated:
                if (
                    tuple(state.shape) != (b, self.num_agents * self.obs_dim)
                    or tuple(next_state.shape) != (b, self.num_agents * self.obs_dim)
                    or not torch.allclose(state, obs.flatten(1), atol=1e-6, rtol=1e-6)
                    or not torch.allclose(next_state, next_obs.flatten(1), atol=1e-6, rtol=1e-6)
                ):
                    raise ValueError("input state is not concat(actor observations)")
                self._validated = True
            pairs = [
                (self.obs, obs),
                (self.next_obs, next_obs),
                (self.act_cont, act_cont),
                (self.act_disc, act_disc_onehot),
                (self.rewards, rewards.reshape(b, 1)),
                (self.dones, dones.reshape(b, 1)),
                (self.mask, mask),
                (self.next_mask, next_mask),
            ]
            end = self.ptr + b
            for dest, source in pairs:
                cast = source.to(self.device, dtype=dest.dtype, non_blocking=True)
                if end <= self.capacity:
                    dest[self.ptr : end].copy_(cast, non_blocking=True)
                else:
                    n1 = self.capacity - self.ptr
                    dest[self.ptr :].copy_(cast[:n1], non_blocking=True)
                    dest[: b - n1].copy_(cast[n1:], non_blocking=True)
            self.ptr = end % self.capacity
            self.size = min(self.capacity, self.size + b)

    def sample(self, batch_size: int) -> dict[str, Any]:
        torch = self._torch
        if self.size < 1:
            raise RuntimeError("empty replay")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        obs = self.obs[idx].float()
        next_obs = self.next_obs[idx].float()
        return {
            "obs": obs,
            "next_obs": next_obs,
            "state": obs.reshape(batch_size, -1),
            "next_state": next_obs.reshape(batch_size, -1),
            "act_cont": self.act_cont[idx],
            "act_disc": self.act_disc[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "mask": self.mask[idx].bool(),
            "next_mask": self.next_mask[idx].bool(),
        }

    def vram_bytes(self) -> int:
        return sum(
            t.nelement() * t.element_size()
            for t in (
                self.obs,
                self.next_obs,
                self.act_cont,
                self.act_disc,
                self.rewards,
                self.dones,
                self.mask,
                self.next_mask,
            )
        )
