"""Bounded MASAC TX-power critic calibration against matched simulator branches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..algorithms.action_utils import canonical_continuous, joint_action, valid_destinations
from ..algorithms.checkpoints import load_policy
from ..communication.link_budget import power_from_action
from ..environment.env import Action
from .evaluation import make_evaluation_env


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rollout_from_snapshot(env, learner, snapshot, first_action, horizon=120):
    """Restore one exact state, execute one counterfactual action, then follow the policy."""
    restored = env.vector.restore_state(snapshot)
    env._refresh()
    observation = _numpy(restored["observation"])[0]
    total = discounted = 0.0
    gamma = float(learner.config.masac.gamma)
    final_info = {}
    for step in range(int(horizon)):
        action = first_action if step == 0 else learner.act(observation, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total += float(reward)
        discounted += (gamma ** step) * float(reward)
        if terminated or truncated:
            break
    diagnostics = final_info.get("network", {}).get("diagnostics", {})
    return {
        "return": float(total),
        "discounted_return": float(discounted),
        "delivery_rate_percent": float(final_info.get("delivery_rate_percent", 0.0)),
        "phy_successes": int(diagnostics.get("data_phy_successes", 0)),
        "steps": step + 1,
    }


def calibration_summary(rows):
    if not rows:
        raise ValueError("Power calibration requires at least one matched snapshot")

    def mean(values):
        return float(np.mean(np.asarray(values, dtype=float)))

    q_delta = [row["q_high"] - row["q_low"] for row in rows]
    return_delta = [row["high"]["return"] - row["low"]["return"] for row in rows]
    discounted_delta = [row["high"]["discounted_return"] - row["low"]["discounted_return"] for row in rows]
    delivery_delta = [row["high"]["delivery_rate_percent"] - row["low"]["delivery_rate_percent"] for row in rows]
    phy_delta = [row["high"]["phy_successes"] - row["low"]["phy_successes"] for row in rows]
    agreement = [int((q == 0 and ret == 0) or q * ret > 0) for q, ret in zip(q_delta, discounted_delta)]
    result = {
        "snapshots": len(rows),
        "critic_q_high_minus_low_mean": mean(q_delta),
        "environment_return_high_minus_low_mean": mean(return_delta),
        "environment_discounted_return_high_minus_low_mean": mean(discounted_delta),
        "delivery_high_minus_low_mean": mean(delivery_delta),
        "phy_success_high_minus_low_mean": mean(phy_delta),
        "ordering_agreement_fraction": mean(agreement),
    }
    policy_power = [row["policy_power_w_mean"] for row in rows if row.get("policy_power_w_mean") is not None]
    if policy_power:
        result["policy_power_w_mean"] = mean(policy_power)
    return result


def _forced_power_pair(learner, observation, low=-1.0, high=1.0):
    obs = torch.as_tensor(observation, dtype=torch.float32, device=learner.device)
    with torch.no_grad():
        continuous, _, probabilities, _ = learner._sample_agents(obs[None], deterministic=True)
    continuous, probabilities = continuous[0], probabilities[0]
    mask = valid_destinations(obs, learner.destinations)
    destination = probabilities.argmax(-1)
    forced = torch.zeros_like(destination, dtype=torch.bool)
    for index in range(learner.num_uavs):
        candidates = mask[index].clone()
        candidates[0] = False
        if bool(candidates.any()):
            scores = probabilities[index].clone()
            scores[~candidates] = -1
            destination[index] = scores.argmax()
            forced[index] = True
    base = canonical_continuous(
        continuous, obs, destination, learn_power=learner.learn_power).clone()
    low_continuous, high_continuous = base.clone(), base.clone()
    low_continuous[forced, learner.continuous_dim - 1] = float(low)
    high_continuous[forced, learner.continuous_dim - 1] = float(high)
    low_action = Action(low_continuous.cpu().numpy(), destination.cpu().numpy().astype(np.int64))
    high_action = Action(high_continuous.cpu().numpy(), destination.cpu().numpy().astype(np.int64))
    policy_power = continuous[forced, learner.continuous_dim - 1].detach().cpu().tolist()
    physical = [power_from_action(value, learner.config.network) for value in policy_power]
    return low_action, high_action, int(forced.sum()), (float(np.mean(physical)) if physical else None)


def _critic_value(learner, state, action):
    state = torch.as_tensor(state, dtype=torch.float32, device=learner.device)[None]
    continuous = torch.as_tensor(action.continuous, dtype=torch.float32, device=learner.device)[None]
    destination = torch.as_tensor(action.destination, dtype=torch.long, device=learner.device)[None]
    joint = joint_action(continuous, destination, learner.destinations, learn_power=learner.learn_power)
    with torch.no_grad():
        if learner.share_critic_parameters:
            q1, q2 = learner.critics[0](state, joint)
            return float(torch.minimum(q1, q2).item())
        values = []
        for critic in learner.critics:
            q1, q2 = critic(state, joint)
            values.append(torch.minimum(q1, q2))
        return float(torch.stack(values).mean().item())


def probe_power_calibration(checkpoint, *, seed_start=None, snapshots=3, horizon=120,
                            scan_steps=500, low=-1.0, high=1.0):
    learner = load_policy(checkpoint, device="cpu")
    if learner.algorithm != "masac" or not learner.learn_power:
        raise ValueError("Power calibration requires MASAC with learned TX power")
    seed_start = learner.config.runtime.eval_seed if seed_start is None else int(seed_start)
    rows = []
    env = make_evaluation_env(learner.config)
    try:
        for offset in range(int(snapshots)):
            seed = seed_start + offset
            observation, _ = env.reset(seed=seed)
            row = None
            for step in range(int(scan_steps)):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                if step >= 1 and bool(valid_destinations(obs_tensor, learner.destinations)[:, 1:].any()):
                    snapshot = env.vector.snapshot_state()
                    state = _numpy(env.vector._buffers[env.vector._buffer]["state"])[0].copy()
                    low_action, high_action, forced_count, policy_power = _forced_power_pair(
                        learner, observation, low=low, high=high)
                    if forced_count:
                        row = {
                            "seed": seed,
                            "scan_step": step,
                            "forced_agents": forced_count,
                            "policy_power_w_mean": policy_power,
                            "q_low": _critic_value(learner, state, low_action),
                            "q_high": _critic_value(learner, state, high_action),
                            "low": rollout_from_snapshot(env, learner, snapshot, low_action, horizon=horizon),
                            "high": rollout_from_snapshot(env, learner, snapshot, high_action, horizon=horizon),
                        }
                        break
                action = learner.act(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            if row is not None:
                rows.append(row)
    finally:
        env.close()
    result = calibration_summary(rows)
    result.update({
        "checkpoint": str(checkpoint),
        "seed_start": seed_start,
        "requested_snapshots": int(snapshots),
        "horizon": int(horizon),
        "low_power_action": float(low),
        "high_power_action": float(high),
        "rows": rows,
    })
    return result


def save_calibration(result, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return path
