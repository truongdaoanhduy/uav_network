"""Read-only Warp CPU/CUDA per-kernel profiler for NoOracle V9/V10.

Do not use this benchmark as a training result. It profiles only the environment
with randomly sampled actions. Run on the SAME Kaggle T4x2 hardware and env count
when comparing source commits; warm-up compilations are excluded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time

import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def aggregate_activities(activities):
    """Warp timing result elapsed is measured in milliseconds."""
    totals = defaultdict(float)
    for item in activities:
        name = str(getattr(item, "name", "unknown")).split(".")[-1]
        totals[name] += float(getattr(item, "elapsed", 0.0))
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--network-disabled", action="store_true")
    args = p.parse_args()
    if args.envs < 1 or args.steps < 1 or args.warmup < 1:
        raise ValueError("envs, steps and warmup must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; use --device cpu")
    cfg = load_config(args.config)
    cfg.runtime.device = args.device
    cfg.runtime.num_envs = args.envs
    if args.network_disabled:
        cfg.network.enabled = False
    torch.manual_seed(44)
    v = WarpVectorEnv(cfg, device=args.device)
    try:
        obs, _ = v.reset(seed=list(range(44, 44 + args.envs)))
        def step():
            nonlocal obs
            act = v.random_actions(obs["observation"])
            obs, _, _, _, _ = v.step({"continuous": act.continuous, "destination": act.destination})
        for _ in range(args.warmup):
            step()
        wp.synchronize_device(v.wp_device)
        if args.device == "cuda":
            wp.timing_begin(cuda_filter=wp.TIMING_KERNEL)
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            activities = wp.timing_end()
            wp.synchronize_device(v.wp_device)
            elapsed = time.perf_counter() - start
            timings = aggregate_activities(activities)
        else:
            original_launch = v._launch
            timings = defaultdict(float)
            def timed(kernel, dim, *inputs):
                now = time.perf_counter()
                result = original_launch(kernel, dim, *inputs)
                timings[getattr(kernel, "key", getattr(kernel, "__name__", str(kernel)))] += (
                    1000 * (time.perf_counter() - now))
                return result
            v._launch = timed
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            elapsed = time.perf_counter() - start
            timings = dict(sorted(timings.items(), key=lambda kv: -kv[1]))
        total_kernel_ms = sum(timings.values())
        result = {
            "benchmark_kind": "environment_random_actions_not_training",
            "device": args.device,
            "envs": args.envs,
            "steps": args.steps,
            "report_slots": int(v.p.report_slots),
            "network_enabled": bool(cfg.network.enabled),
            "wall_seconds": elapsed,
            "env_transitions_per_second": args.envs * args.steps / elapsed,
            "per_kernel_ms_total": timings,
            "per_kernel_percent": {
                k: 100.0 * v / total_kernel_ms
                for k, v in timings.items()
            } if total_kernel_ms else {},
        }
        print("UAV_WARP_PROFILE_JSON", json.dumps(result, sort_keys=True), flush=True)
    finally:
        v.close()


if __name__ == "__main__":
    main()
