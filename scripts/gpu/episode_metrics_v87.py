"""V87 Pre-reset completed-episode metrics tracker (with IQM) & held-out deterministic evaluator.

Fixes the V86 E64x900 evaluation blind spot where `mission_snapshot(world)` ran AFTER
`world.auto_reset_gpu()`, reading freshly-reset 0%-delivery worlds instead of the 52–63
episodes that just completed at step 900.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def compute_iqm(values) -> float:
    """Interquartile Mean (Agarwal et al., NeurIPS 2021 rliable): mean of middle 50%."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = int(arr.size)
    if n == 0:
        return 0.0
    lower, upper = 0.25 * n, 0.75 * n
    idx = np.arange(n, dtype=np.float64)
    w = np.maximum(0.0, np.minimum(idx + 1.0, upper) - np.maximum(idx, lower))
    return float(np.dot(w, arr) / (upper - lower))


def summarize_series(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "iqm": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "iqm": compute_iqm(arr),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class CompletedEpisodeTracker:
    """Tracks per-env running return/length and snapshots completed worlds BEFORE auto_reset."""

    def __init__(self, num_envs: int, e_max_j: float = 180_000.0, num_targets: int = 15):
        self.num_envs = int(num_envs)
        self.e_max_j = float(e_max_j)
        self.num_targets = int(num_targets)
        self.running_return = np.zeros(self.num_envs, dtype=np.float64)
        self.running_length = np.zeros(self.num_envs, dtype=np.int64)
        self.records: Dict[str, List[float]] = {
            "return": [],
            "length": [],
            "gcs_delivery_pct": [],
            "false_confirmed_cells": [],
            "team_coverage_pct": [],
            "battery_pct": [],
            "full15_success": [],
            "safety_failed": [],
        }

    def record_step_np(
        self,
        reward,
        done,
        termination_reason,
        gcs_tp_counts,
        gcs_fp_counts,
        coverage_pct,
        battery_pct,
    ) -> int:
        """Pure-NumPy update called BEFORE `auto_reset_gpu()`. Returns newly completed count."""
        rew = np.asarray(reward, dtype=np.float64)
        dn = np.asarray(done, dtype=np.float64) > 0.5
        tr = np.asarray(termination_reason, dtype=np.int64)
        self.running_return += rew
        self.running_length += 1
        idxs = np.nonzero(dn)[0]
        if idxs.size == 0:
            return 0
        tp = np.asarray(gcs_tp_counts, dtype=np.float64)
        fp = np.asarray(gcs_fp_counts, dtype=np.float64)
        cov = np.asarray(coverage_pct, dtype=np.float64)
        bat = np.asarray(battery_pct, dtype=np.float64)
        for e in idxs:
            self.records["return"].append(float(self.running_return[e]))
            self.records["length"].append(float(self.running_length[e]))
            self.records["gcs_delivery_pct"].append(float(tp[e] / max(1, self.num_targets) * 100.0))
            self.records["false_confirmed_cells"].append(float(fp[e]))
            self.records["team_coverage_pct"].append(float(cov[e]))
            self.records["battery_pct"].append(float(bat[e]))
            self.records["full15_success"].append(1.0 if (tr[e] & 4) != 0 else 0.0)
            self.records["safety_failed"].append(1.0 if (tr[e] & 1) != 0 else 0.0)
            self.running_return[e] = 0.0
            self.running_length[e] = 0
        return int(idxs.size)

    def record_step_world(self, world, rew_t, done_t, t_reason_t) -> int:
        """Torch/Warp wrapper: only copies to host on steps where at least one episode ended."""
        import torch
        import warp as wp

        rew_np = rew_t.detach().cpu().numpy()
        dn_np = done_t.detach().cpu().numpy()
        if not np.any(dn_np > 0.5):
            self.running_return += rew_np
            self.running_length += 1
            return 0
        b, n = world.buffers, world.network.data
        tp = wp.to_torch(b["gcs_tp"]).float().sum(-1).cpu().numpy()
        fp = wp.to_torch(b["gcs_fp"]).float().sum(-1).cpu().numpy()
        gen = wp.to_torch(n["ledger_gen"])
        ids = torch.arange(6, device=gen.device)
        cov = ((gen[:, ids, ids] >= 0).any(1).float().mean(-1) * 100.0).cpu().numpy()
        bat = (wp.to_torch(b["battery"]).float().mean(-1) / self.e_max_j * 100.0).cpu().numpy()
        tr_np = t_reason_t.detach().cpu().numpy()
        return self.record_step_np(rew_np, dn_np, tr_np, tp, fp, cov, bat)

    def summary(self) -> Dict[str, object]:
        return {
            "completed_episodes": len(self.records["return"]),
            "metrics": {k: summarize_series(v) for k, v in self.records.items()},
        }


def evaluate_policy_heldout(
    learner,
    scenario_path,
    scenario_overrides,
    *,
    episodes: int,
    eval_seed: int,
    horizon: int,
    world_device: str,
    learner_device: str,
    strict_braking_terminate: bool = False,
) -> Dict[str, object]:
    """Run deterministic policy evaluation on independent held-out worlds (no training updates)."""
    import torch
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

    params = GPUWorldV80Params.from_yaml(
        scenario_path, envs=int(episodes), device=world_device, seed=int(eval_seed),
        scenario_overrides=tuple(scenario_overrides),
        horizon=int(horizon), strict_braking_terminate=bool(strict_braking_terminate),
    )
    world = WarpParallelMissionV80(params)
    tracker = CompletedEpisodeTracker(int(episodes), e_max_j=float(params.e_max_j), num_targets=15)
    finished = np.zeros(int(episodes), dtype=bool)

    from zhang_xue.learners.actor_adapter import encode_actor_input_torch

    def _obs():
        raw = wp.to_torch(world.buffers["obs"]).to(learner_device)
        masks = wp.to_torch(world.buffers["mask"]).to(learner_device, dtype=torch.float32)
        return encode_actor_input_torch(raw, masks)

    for _ in range(int(horizon)):
        if finished.all():
            break
        obs = _obs()
        action = learner.act_batch(obs, deterministic=True, return_tensors=True)
        act_cont = action.continuous.to(world_device).contiguous()
        act_dest = action.destination.to(device=world_device, dtype=torch.int32).contiguous()
        world.step(wp.from_torch(act_cont, dtype=wp.float32), wp.from_torch(act_dest, dtype=wp.int32))
        rew = wp.to_torch(world.buffers["reward"]).to(learner_device, dtype=torch.float32)
        done = wp.to_torch(world.buffers["done"]).to(learner_device, dtype=torch.float32)
        tr = wp.to_torch(world.buffers["termination_reason"]).to(learner_device)
        # Mask out already-finished envs so each held-out seed is counted exactly once.
        active_mask = torch.as_tensor((~finished).astype(np.float32), device=learner_device)
        tracker.record_step_world(world, rew * active_mask, done * active_mask, tr)
        finished |= (done.cpu().numpy() > 0.5)
    out = tracker.summary()
    out["eval_seed"] = int(eval_seed)
    out["eval_episodes_requested"] = int(episodes)
    out["deterministic"] = True
    return out
