#!/usr/bin/env python3
"""V79 GPU-vectorized Warp worlds -> MATD3/MASAC rollout (YAML-synced, zero-sync auto-reset).

Key upgrades over V78 (`run_v78_real_rollout.py`):
- Loads exact RF/DS/reward parameters from scenario YAML (no silent 12 dB vs 20 dB drift).
- Passes ONLY `terminated` (never `truncated`) into replay buffer to preserve horizon bootstrapping.
- Uses `world.auto_reset_gpu()` (masked 600ms warmup on reset worlds only, zero host-sync on `done`).
- Supports `--horizon` override (e.g. `--horizon 2`) to validate multi-episode selective auto-reset.
- Logs peak `torch.cuda.max_memory_allocated` and `max_memory_reserved` per GPU.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["matd3", "masac"], required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml")
    parser.add_argument("--scenario-set", action="append", default=[])
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--world-device", default="cpu")
    parser.add_argument("--learner-device", default="cpu")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--allow-experimental-physics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never concatenate the current run with an earlier Kaggle execution.
    args.output.with_suffix(".progress.jsonl").write_text("")
    result = {
        "status": "RUNNING", "version": "V79", "envs": args.envs, "steps": args.steps,
        "algorithm": args.algorithm, "batch_size": args.batch_size,
        "world_device": args.world_device, "learner_device": args.learner_device,
        "independent_world_runtime": True, "uses_repeated_cpu_fixtures": False,
        "masked_selective_reset": True, "separates_terminated_truncated": True,
        "cpu_warp_parity_proven": False, "gpu_training_converged": False,
        "scientific_paper_ready": False, "step_logging_causes_host_sync": True,
        "source_path": str(ROOT), "scenario_path": str(args.scenario),
        "updates": 0, "world_steps": 0, "episodes_ended_total": 0,
    }
    start = time.time()
    try:
        if not args.allow_experimental_physics:
            result["status"] = "BLOCKED_EXPERIMENTAL_PHYSICS_REQUIRES_ACK"
            raise RuntimeError("add --allow-experimental-physics after reviewing V79 scope")
        import torch
        import warp as wp
        from zhang_xue.gpu.mission_v79 import GPUWorldV79Params, WarpParallelMissionV79
        from uav_network.algorithms.configuration.loader import load_learner_config
        from uav_network.algorithms.matd3.learner import MATD3
        from uav_network.algorithms.masac.learner import MASAC
        from uav_network.algorithms.core.replay import TensorReplayBuffer

        for dev in (args.world_device, args.learner_device):
            if dev.startswith("cuda:"):
                if not torch.cuda.is_available() or int(dev.split(":")[1]) >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA unavailable: {dev}")
                # PyTorch 2.14 requires this CUDA device to be initialized before resetting per-device stats.
                torch.cuda.set_device(dev)
                torch.cuda.reset_peak_memory_stats(dev)

        cfg = load_learner_config(ROOT / "configs/learners/e4096_b8192.yaml")
        cfg.runtime.algorithm = args.algorithm
        cfg.runtime.num_envs = args.envs
        hp = getattr(cfg, args.algorithm)
        hp.batch_size = args.batch_size
        hp.replay_capacity = max(args.batch_size * 4, args.envs * 8)
        hp.hidden_sizes = [args.hidden, args.hidden]
        cfg.validate()
        torch.manual_seed(args.seed)

        overrides = {} if args.horizon is None else {"horizon": int(args.horizon)}
        params = GPUWorldV79Params.from_yaml(
            args.scenario, envs=args.envs, device=args.world_device, seed=args.seed,
            scenario_overrides=tuple(args.scenario_set), **overrides,
        )
        result["world_params"] = asdict(params)
        import hashlib
        result["scenario_sha256"] = hashlib.sha256(args.scenario.read_bytes()).hexdigest()
        result["project_manifest_sha256"] = hashlib.sha256(
            (ROOT / "SOURCE_SHA256_MANIFEST.json").read_bytes()
        ).hexdigest()
        world = WarpParallelMissionV79(params)
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
            reasons = wp.to_torch(world.buffers["termination_reason"]).to(args.learner_device, dtype=torch.int32).clone()
            safety_flags = wp.to_torch(world.buffers["safety_reason"]).to(args.learner_device, dtype=torch.int32).clone()
            # Strictly pass `terminated` (NOT `done`) to preserve horizon bootstrapping!
            replay.add_batch(old, state, action, rew, nextobs, nextstate, term)
            # Zero host-sync GPU selective auto-reset (masked 600ms warmup for done worlds only)
            world.auto_reset_gpu()
            ep_done = int((term + trunc).clamp(max=1.0).sum().item())
            result["episodes_ended_total"] += ep_done
            step_result = {
                "vector_step": i + 1, "true_transition_count": (i + 1) * args.envs,
                "mean_reward": float(rew.mean().item()), "episodes_ended": ep_done,
                "terminated_count": int(term.sum().item()), "truncated_count": int(trunc.sum().item()),
                "safety_failure_type_counts": {
                    "geometry_or_boundary": int(((safety_flags & 1) != 0).sum().item()),
                    "uav_pair_collision_risk": int(((safety_flags & 2) != 0).sum().item()),
                },
                "termination_reason_counts": {
                    "safety_failed": int(((reasons & 1) != 0).sum().item()),
                    "all_uavs_depleted": int(((reasons & 2) != 0).sum().item()),
                    "all_targets_confirmed_at_gcs": int(((reasons & 4) != 0).sum().item()),
                    "horizon_truncated": int(((reasons & 8) != 0).sum().item()),
                },
            }
            if len(replay) >= args.batch_size:
                step_result["updates"] = []
                for _ in range(args.updates_per_step):
                    metrics = {k: float(v) for k, v in learner.update(replay.sample(args.batch_size)).items()}
                    # Diagnostic normalization only: do NOT alter optimizer loss or actor/critic gradients.
                    # MATD3 logs SUM of two Q MSEs; MASAC logs average two heads,
                    # multiplied by critic module count only in independent-critic mode.
                    # Bellman targets differ across algorithms, hence not a causal apples-to-apples loss.
                    if "critic_loss" in metrics:
                        if args.algorithm == "matd3":
                            head_count_factor = 2.0
                        else:
                            head_count_factor = float(len(learner.critics)) if not learner.share_critic_parameters else 1.0
                        metrics["critic_loss_per_head_proxy"] = metrics["critic_loss"] / head_count_factor
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError("non-finite learner metrics")
                    step_result["updates"].append(metrics)
                    result["updates"] += 1
            statuses.append(step_result)
            result["world_steps"] = i + 1
            if (i + 1) % max(1, min(args.steps, 10)) == 0:
                world.guard()
            args.output.with_suffix(".progress.jsonl").open("a").write(json.dumps(step_result, allow_nan=False) + "\n")

        result["final_world_invariants"] = world.guard()
        result["cuda_peak_memory_bytes"] = {
            dev: {
                "max_allocated": int(torch.cuda.max_memory_allocated(dev)),
                "max_reserved": int(torch.cuda.max_memory_reserved(dev)),
            }
            for dev in sorted(set((args.world_device, args.learner_device)))
            if dev.startswith("cuda:")
        }
        # Report the backend actually used. CPU smoke tests are not GPU verification.
        if args.world_device.startswith("cuda:") and args.learner_device.startswith("cuda:"):
            backend = "GPU"
        elif args.world_device == "cpu" and args.learner_device == "cpu":
            backend = "CPU"
        else:
            backend = "MIXED"
        result["status"] = f"V79_{backend}_ROLLOUT_PASS_VERIFY_PARITY_AND_CONVERGENCE"
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
    return 0 if "_ROLLOUT_PASS_" in result["status"] else (3 if result["status"].startswith("BLOCKED") else 1)


if __name__ == "__main__":
    sys.exit(main())
