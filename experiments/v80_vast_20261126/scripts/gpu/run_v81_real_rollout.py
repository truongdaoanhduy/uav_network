#!/usr/bin/env python3
"""V84 verified GPU-vectorized Warp worlds -> MATD3/MASAC rollout with shield-aware replay.

Extends V82 without breaking V80/V82 contracts:
- Adds `--replay-action-mode {raw,executed,dual}` and `--shield-penalty-lambda` as experimental learning-tuple variants.
  Raw replay is a valid shielded-environment formulation; variants require ablations.
- Accumulates episode/termination/safety counters on-device and syncs only on `--log-interval`
  or final step (Issue #24), while retaining every-step `require_no_warp_overflow` before
  `world.auto_reset_gpu()` and fail-closed physical commit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def require_no_warp_overflow(world, vector_step: int) -> None:
    """Fail closed on any MAC event overflow before the earliest auto reset."""
    import warp as wp
    count = int(wp.to_torch(world.network.data["overflow"]).sum().item())
    if count:
        raise RuntimeError(
            f"Warp packet event/gen overflow at vector_step={vector_step}: "
            f"{count} events before selective auto-reset"
        )


def _store_transitions(replay, old, state, action, rew, nextobs, nextstate, term,
                       world, params, learner_device, mode, lam, *, return_details=False):
    from replay_v84 import store_transitions
    return store_transitions(replay, old, state, action, rew, nextobs, nextstate, term,
                             world, params, learner_device, mode, lam,
                             return_details=return_details)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["matd3", "masac"], required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml")
    parser.add_argument("--learner-config", type=Path, default=ROOT / "configs/learners/v81_matd3_masac_stable.yaml")
    parser.add_argument("--scenario-set", action="append", default=[])
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--strict-braking-terminate", action="store_true")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=4096)
    parser.add_argument("--replay-capacity", type=int, default=65536)
    parser.add_argument("--replay-action-mode", choices=["raw", "raw_penalized", "executed", "dual"], default="raw")
    parser.add_argument("--baseline-mode", choices=["hybrid", "canonical_continuous"], default="hybrid")
    parser.add_argument("--shield-penalty-lambda", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--allow-no-updates", action="store_true")
    parser.add_argument("--world-device", default="cuda:0")
    parser.add_argument("--learner-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--allow-experimental-physics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".progress.jsonl").exists():
        parser.error("output or progress already exists; choose a fresh output path")
    result = {
        "status": "RUNNING", "version": ("V86-CANONICAL" if args.baseline_mode == "canonical_continuous" else "V84"), "envs": args.envs, "steps": args.steps,
        "algorithm": args.algorithm, "batch_size": args.batch_size,
        "learner_config": str(args.learner_config),
        "learning_starts": int(args.learning_starts), "log_interval": args.log_interval,
        "replay_capacity_requested": int(args.replay_capacity),
        "replay_action_mode": args.replay_action_mode,
        "baseline_mode": args.baseline_mode,
        "comm_policy": ("event_driven_zero_oracle" if args.baseline_mode == "canonical_continuous" else "learned_compact_3"),
        "shield_penalty_lambda": float(args.shield_penalty_lambda),
        "dual_penalty_transitions_added": 0,
        "world_device": args.world_device, "learner_device": args.learner_device,
        "feasible_66cand_shield": True, "l2_corner_braking_fix": True,
        "strict_braking_terminate": bool(args.strict_braking_terminate),
        "updates": 0, "world_steps": 0, "episodes_ended_total": 0,
        "termination_reason_counts": {
            "safety_failed": 0, "all_uavs_depleted": 0,
            "all_targets_confirmed_at_gcs": 0, "horizon_truncated": 0,
        },
        "safety_failure_type_counts": {
            "boundary_collision": 0, "building_collision": 0,
            "uav_pair_collision": 0, "braking_margin_warning": 0,
        },
    }
    gpu_totals = None
    start = time.time()
    try:
        if not args.allow_experimental_physics:
            result["status"] = "BLOCKED_EXPERIMENTAL_PHYSICS_REQUIRES_ACK"
            raise RuntimeError("add --allow-experimental-physics after reviewing V83 scope")
        import torch
        import warp as wp
        from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
        from uav_network.algorithms.configuration.loader import load_learner_config
        from uav_network.algorithms.matd3.learner import MATD3
        from uav_network.algorithms.masac.learner import MASAC
        from uav_network.algorithms.core.replay import TensorReplayBuffer

        for dev in (args.world_device, args.learner_device):
            if dev == "cuda" or dev.startswith("cuda:"):
                idx = 0 if dev == "cuda" else int(dev.split(":")[1])
                if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA unavailable: {dev}")
                torch.cuda.set_device(idx)
                torch.cuda.reset_peak_memory_stats(idx)

        cfg = load_learner_config(args.learner_config)
        from rollout_v82_contracts import configure_learner, provenance, mission_snapshot
        hp = configure_learner(cfg, args)
        result.update(provenance(ROOT, cfg, args.scenario, args.learner_config, args.seed))
        result["learner_hyperparameters"] = asdict(hp)
        torch.manual_seed(args.seed)

        overrides = {"strict_braking_terminate": bool(args.strict_braking_terminate)}
        if args.horizon is not None:
            overrides["horizon"] = int(args.horizon)
        params = GPUWorldV80Params.from_yaml(
            args.scenario, envs=args.envs, device=args.world_device, seed=args.seed,
            scenario_overrides=tuple(args.scenario_set), **overrides,
        )
        result["world_params"] = asdict(params)
        world = WarpParallelMissionV80(params)
        result["initial_world_invariants"] = world.guard()

        obs_dim = 518
        state_dim = 6 * obs_dim
        if args.baseline_mode == "canonical_continuous":
            from uav_network.algorithms.baselines.canonical_continuous import (
                CanonicalContinuousMASAC, CanonicalContinuousMATD3,
            )
            learner_cls = CanonicalContinuousMATD3 if args.algorithm == "matd3" else CanonicalContinuousMASAC
        else:
            learner_cls = MATD3 if args.algorithm == "matd3" else MASAC
        learner = learner_cls(
            cfg, observation_dim=obs_dim, state_dim=state_dim, device=args.learner_device,
        )
        replay = TensorReplayBuffer(
            hp.replay_capacity, (6, obs_dim), state_dim,
            device=args.learner_device, continuous_dim=2, seed=args.seed,
        )
        result["replay_allocated_bytes"] = replay.allocated_bytes

        def encoded_obs():
            raw = wp.to_torch(world.buffers["obs"]).to(args.learner_device).clone()
            masks = wp.to_torch(world.buffers["mask"]).to(args.learner_device, dtype=torch.float32).clone()
            dist = raw[..., 9:10].clone()
            raw[..., 9] = (raw[..., 4] > 0).to(torch.float32)
            return torch.cat([raw, dist, masks], dim=-1)

        min_update_size = max(args.batch_size, int(args.learning_starts))
        gpu_totals = torch.zeros(9, dtype=torch.int64, device=args.learner_device)
        last_record = None
        previous_logged_totals = [0] * 9
        previous_logged_step = 0

        def sync_totals():
            if gpu_totals is None:
                return
            tot = gpu_totals.cpu().tolist()
            result["episodes_ended_total"] = tot[0]
            result["termination_reason_counts"] = dict(zip(
                ("safety_failed", "all_uavs_depleted", "all_targets_confirmed_at_gcs", "horizon_truncated"), tot[1:5]))
            result["safety_failure_type_counts"] = dict(zip(
                ("boundary_collision", "building_collision", "uav_pair_collision", "braking_margin_warning"), tot[5:9]))
            result["counts_complete_for_recorded_world_steps"] = True

        for i in range(args.steps):
            old = encoded_obs()
            state = old.reshape(args.envs, -1)
            action = learner.act_batch(old, deterministic=False, return_tensors=True)
            act_cont = action.continuous.to(args.world_device).contiguous()
            act_dest = action.destination.to(device=args.world_device, dtype=torch.int32).contiguous()
            world.step(wp.from_torch(act_cont, dtype=wp.float32), wp.from_torch(act_dest, dtype=wp.int32))
            nextobs = encoded_obs()
            nextstate = nextobs.reshape(args.envs, -1)
            rew = wp.to_torch(world.buffers["reward"]).to(args.learner_device, dtype=torch.float32).clone()
            term = wp.to_torch(world.buffers["terminated"]).to(args.learner_device, dtype=torch.float32).clone()
            trunc = wp.to_torch(world.buffers["truncated"]).to(args.learner_device, dtype=torch.float32).clone()
            t_reason = wp.to_torch(world.buffers["termination_reason"]).to(args.learner_device).clone()
            s_reason = wp.to_torch(world.buffers["safety_reason"]).to(args.learner_device).clone()
            require_no_warp_overflow(world, i + 1)
            result["overflow_checked_steps"] = i + 1
            stored = _store_transitions(
                replay, old, state, action, rew, nextobs, nextstate, term,
                world, params, args.learner_device,
                args.replay_action_mode, float(args.shield_penalty_lambda), return_details=True,
            )
            result["dual_penalty_transitions_added"] += stored["extra_rows"]
            result["true_transition_count"] = (i + 1) * args.envs
            result["replay_rows_inserted"] = result.get("replay_rows_inserted", 0) + args.envs + stored["extra_rows"]
            world.auto_reset_gpu()

            step_vec = torch.stack((
                (term + trunc).clamp(max=1.0).to(torch.int64).sum(),
                ((t_reason & 1) != 0).to(torch.int64).sum(),
                ((t_reason & 2) != 0).to(torch.int64).sum(),
                ((t_reason & 4) != 0).to(torch.int64).sum(),
                ((t_reason & 8) != 0).to(torch.int64).sum(),
                ((s_reason & 1) != 0).to(torch.int64).sum(),
                ((s_reason & 2) != 0).to(torch.int64).sum(),
                ((s_reason & 4) != 0).to(torch.int64).sum(),
                ((s_reason & 8) != 0).to(torch.int64).sum(),
            ))
            gpu_totals += step_vec
            should_log = ((i + 1) % args.log_interval == 0 or i + 1 == args.steps)
            step_updates = []
            result["world_steps"] = i + 1
            if result["true_transition_count"] >= min_update_size and len(replay) >= args.batch_size:
                for _ in range(args.updates_per_step):
                    metrics = {k: float(v) for k, v in learner.update(replay.sample(args.batch_size)).items()}
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError("non-finite learner metrics")
                    step_updates.append(metrics)
                    result["updates"] += 1
            result["world_steps"] = i + 1
            if should_log:
                cumulative = gpu_totals.cpu().tolist()
                sv = [x - y for x, y in zip(cumulative, previous_logged_totals)]
                window_start = previous_logged_step + 1
                previous_logged_totals, previous_logged_step = cumulative, i + 1
                step_tr = {
                    "safety_failed": sv[1], "all_uavs_depleted": sv[2],
                    "all_targets_confirmed_at_gcs": sv[3], "horizon_truncated": sv[4],
                }
                step_sr = {
                    "boundary_collision": sv[5], "building_collision": sv[6],
                    "uav_pair_collision": sv[7], "braking_margin_warning": sv[8],
                }
                step_result = {
                    "vector_step": i + 1, "true_transition_count": (i + 1) * args.envs,
                    "counts_scope": "since_previous_progress_record",
                    "window_start_vector_step": window_start,
                    "cumulative_episode_count": cumulative[0],
                    "training_reward_mean": float(stored["training_reward"].mean().item()),
                    "shield_cost_mean": float(stored["shield_cost"].mean().item()),
                    "mean_reward": float(rew.mean().item()), "episodes_ended": sv[0],
                    "termination_reason_counts": step_tr, "safety_failure_type_counts": step_sr,
                }
                if step_updates:
                    step_result["updates"] = step_updates
                last_record = step_result
                with args.output.with_suffix(".progress.jsonl").open("a") as progress:
                    progress.write(json.dumps(step_result, allow_nan=False) + "\n")

        sync_totals()
        result["final_world_invariants"] = world.guard()
        result["cuda_peak_memory_bytes"] = {
            dev: {
                "max_allocated": int(torch.cuda.max_memory_allocated(0 if dev == "cuda" else int(dev.split(":")[1]))),
                "max_reserved": int(torch.cuda.max_memory_reserved(0 if dev == "cuda" else int(dev.split(":")[1]))),
            }
            for dev in sorted(set((args.world_device, args.learner_device)))
            if dev == "cuda" or dev.startswith("cuda:")
        }
        if any(not torch.isfinite(p).all() for p in learner.parameters()):
            raise FloatingPointError("non-finite final model parameters")
        result["mission_snapshot"] = mission_snapshot(world)
        result["integration_pass"] = True
        result["safety_gate_pass"] = result["termination_reason_counts"]["safety_failed"] == 0
        result["learning_updates_executed"] = result["updates"] > 0
        result["scientific_status"] = "CONVERGENCE_AND_MISSION_SUCCESS_UNPROVEN"
        result["replay_formulation"] = (
            "SHIELDED_ENVIRONMENT_MDP" if args.replay_action_mode == "raw" else
            "ADAPTION_PENALTY_OBJECTIVE" if args.replay_action_mode == "raw_penalized" else
            "EXPERIMENTAL_RELABELLED_SAFE_ACTION_TUPLES")
        result["safe_action_relabelling_idempotence_certified"] = False
        result["status"] = "V84_ROLLOUT_COMPLETED"
        result["final_record"] = last_record
    except Exception as exc:
        if result["status"] == "RUNNING":
            result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if gpu_totals is not None:
            try:
                sync_totals()
            except Exception as counter_error:
                result["counts_complete_for_recorded_world_steps"] = False
                result["counter_readback_error"] = str(counter_error)
    finally:
        result["elapsed_s"] = round(time.time() - start, 3)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "V84_ROLLOUT_COMPLETED" else (3 if result["status"].startswith("BLOCKED") else 1)


if __name__ == "__main__":
    sys.exit(main())
