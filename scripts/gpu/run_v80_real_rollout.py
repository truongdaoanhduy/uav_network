#!/usr/bin/env python3
"""V80 GPU-vectorized Warp worlds -> MATD3/MASAC rollout with Feasible Safety Shield.

Fixes over V79 (`run_v79_real_rollout.py`):
- Uses `WarpParallelMissionV80` (66-candidate feasible lateral projection + 3-sweep CD +
  CPU-parity physical collision detector; keeps `safe_d=50m`, `a_max=5`, fail-closed termination).
- Checks network event overflow on EVERY step BEFORE selective reset, and final invariants at end.
- Handles both `"cuda"` and `"cuda:N"` with `torch.cuda.set_device(...)`.
- Tracks 4-bit safety subtypes: `boundary_collision`, `building_collision`,
  `uav_pair_collision`, and `braking_margin_warning`.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["matd3", "masac"], required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml")
    parser.add_argument("--scenario-set", action="append", default=[])
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--strict-braking-terminate", action="store_true")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--world-device", default="cuda:0")
    parser.add_argument("--learner-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--allow-experimental-physics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING", "version": "V80", "envs": args.envs, "steps": args.steps,
        "algorithm": args.algorithm, "batch_size": args.batch_size,
        "world_device": args.world_device, "learner_device": args.learner_device,
        "feasible_66cand_shield": True, "strict_braking_terminate": bool(args.strict_braking_terminate),
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
    start = time.time()
    try:
        if not args.allow_experimental_physics:
            result["status"] = "BLOCKED_EXPERIMENTAL_PHYSICS_REQUIRES_ACK"
            raise RuntimeError("add --allow-experimental-physics after reviewing V80 scope")
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

        cfg = load_learner_config(ROOT / "configs/learners/e4096_b8192.yaml")
        cfg.runtime.algorithm = args.algorithm
        cfg.runtime.num_envs = args.envs
        hp = getattr(cfg, args.algorithm)
        hp.batch_size = args.batch_size
        hp.replay_capacity = max(args.batch_size * 4, args.envs * 8)
        hp.hidden_sizes = [args.hidden, args.hidden]
        cfg.validate()
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
        learner = (MATD3 if args.algorithm == "matd3" else MASAC)(
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

        statuses = []
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
            # Per-world overflow may be cleared by selective reset. Check EVERY
            # simulated tick, not merely every tenth tick, before any auto-reset.
            require_no_warp_overflow(world, i + 1)
            result["overflow_checked_steps"] = i + 1
            replay.add_batch(old, state, action, rew, nextobs, nextstate, term)
            world.auto_reset_gpu()

            ep_done = int((term + trunc).clamp(max=1.0).sum().item())
            result["episodes_ended_total"] += ep_done
            step_tr = {
                "safety_failed": int(((t_reason & 1) != 0).sum().item()),
                "all_uavs_depleted": int(((t_reason & 2) != 0).sum().item()),
                "all_targets_confirmed_at_gcs": int(((t_reason & 4) != 0).sum().item()),
                "horizon_truncated": int(((t_reason & 8) != 0).sum().item()),
            }
            step_sr = {
                "boundary_collision": int(((s_reason & 1) != 0).sum().item()),
                "building_collision": int(((s_reason & 2) != 0).sum().item()),
                "uav_pair_collision": int(((s_reason & 4) != 0).sum().item()),
                "braking_margin_warning": int(((s_reason & 8) != 0).sum().item()),
            }
            for k, v in step_tr.items():
                result["termination_reason_counts"][k] += v
            for k, v in step_sr.items():
                result["safety_failure_type_counts"][k] += v
            step_result = {
                "vector_step": i + 1, "true_transition_count": (i + 1) * args.envs,
                "mean_reward": float(rew.mean().item()), "episodes_ended": ep_done,
                "termination_reason_counts": step_tr, "safety_failure_type_counts": step_sr,
            }
            if len(replay) >= args.batch_size:
                step_result["updates"] = []
                for _ in range(args.updates_per_step):
                    metrics = {k: float(v) for k, v in learner.update(replay.sample(args.batch_size)).items()}
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError("non-finite learner metrics")
                    step_result["updates"].append(metrics)
                    result["updates"] += 1
            statuses.append(step_result)
            result["world_steps"] = i + 1
            args.output.with_suffix(".progress.jsonl").open("a").write(json.dumps(step_result, allow_nan=False) + "\n")

        result["final_world_invariants"] = world.guard()
        result["cuda_peak_memory_bytes"] = {
            dev: {
                "max_allocated": int(torch.cuda.max_memory_allocated(0 if dev == "cuda" else int(dev.split(":")[1]))),
                "max_reserved": int(torch.cuda.max_memory_reserved(0 if dev == "cuda" else int(dev.split(":")[1]))),
            }
            for dev in sorted(set((args.world_device, args.learner_device)))
            if dev == "cuda" or dev.startswith("cuda:")
        }
        result["status"] = "V80_FEASIBLE_SHIELD_GPU_ROLLOUT_PASS"
        result["final_record"] = statuses[-1]
    except Exception as exc:
        if result["status"] == "RUNNING":
            result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = round(time.time() - start, 3)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "V80_FEASIBLE_SHIELD_GPU_ROLLOUT_PASS" else (3 if result["status"].startswith("BLOCKED") else 1)


if __name__ == "__main__":
    sys.exit(main())
