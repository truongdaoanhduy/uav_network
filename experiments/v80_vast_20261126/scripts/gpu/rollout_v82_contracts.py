"""Validated rollout settings, dual-tuple shield replay helpers, and provenance."""
from pathlib import Path
import hashlib
import math
import subprocess

import numpy as np

REPLAY_ACTION_MODES = ("raw", "raw_penalized", "executed", "dual")


def configure_learner(cfg, args):
    for key in ("envs", "steps", "updates_per_step", "batch_size", "replay_capacity", "log_interval"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if args.learning_starts < 1:
        raise ValueError("learning_starts must be a positive integer")
    if args.hidden is not None and args.hidden < 1:
        raise ValueError("hidden must be a positive integer")
    replay_mode = getattr(args, "replay_action_mode", "raw")
    if replay_mode not in REPLAY_ACTION_MODES:
        raise ValueError(f"replay_action_mode must be one of {REPLAY_ACTION_MODES}")
    lam = float(getattr(args, "shield_penalty_lambda", 0.0))
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("shield_penalty_lambda must be finite and non-negative")
    if args.replay_capacity < max(args.batch_size, args.envs, args.learning_starts):
        raise ValueError("replay_capacity must cover envs, batch_size and learning_starts")
    if args.envs * args.steps < max(args.learning_starts, args.batch_size) and not args.allow_no_updates:
        raise ValueError("learning_starts cannot be reached; use --allow-no-updates for an explicit smoke run")
    cfg.runtime.algorithm, cfg.runtime.num_envs = args.algorithm, args.envs
    hp = getattr(cfg, args.algorithm)
    hp.batch_size, hp.replay_capacity = args.batch_size, args.replay_capacity
    hp.learning_starts, hp.updates_per_step = args.learning_starts, args.updates_per_step
    if args.hidden is not None:
        hp.hidden_sizes = [args.hidden, args.hidden]
    cfg.validate()
    return hp


def compute_dual_replay_penalty_np(raw_cont, exec_cont, reward, terminated, penalty_lambda: float, tol: float = 1e-6):
    """Compute penalized nominal-to-safe difference and experimental penalty rewards.

    Returns (intervened_mask, penalized_reward) for valid worlds where the
    shield modified the continuous velocity command beyond `tol`.
    """
    raw = np.asarray(raw_cont, dtype=np.float32)
    exe = np.asarray(exec_cont, dtype=np.float32)
    rew = np.asarray(reward, dtype=np.float32)
    term = np.asarray(terminated, dtype=np.float32)
    delta_sq = np.sum((raw - exe) ** 2, axis=(-1, -2))
    mask = (term < 0.5) & (delta_sq > float(tol))
    penalized = rew - float(penalty_lambda) * delta_sq
    return mask, penalized


def provenance(root, cfg, scenario_path, learner_path, seed):
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return dict(git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain")),
                seed=seed, effective_learner_config=cfg.to_dict(),
                input_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in (scenario_path, learner_path)})


def mission_snapshot(world):
    """Current live worlds, not complete-episode evaluation or success counts."""
    import torch
    import warp as wp
    b, n = world.buffers, world.network.data
    tp = wp.to_torch(b["gcs_tp"]).float().sum(-1)
    fp = wp.to_torch(b["gcs_fp"]).float().sum(-1)
    gen = wp.to_torch(n["ledger_gen"])
    ids = torch.arange(6, device=gen.device)
    coverage = (gen[:, ids, ids] >= 0).any(1).float().mean(-1)
    values = torch.stack((tp.mean() / 15 * 100, fp.mean(), coverage.mean() * 100,
                          wp.to_torch(b["battery"]).float().mean() / world.params.e_max_j * 100)).cpu().tolist()
    return dict(zip(("current_latched_gcs_delivery_pct", "current_false_confirmed_cells_mean",
                     "current_team_sensed_coverage_pct", "current_battery_pct"), values))
