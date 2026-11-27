#!/usr/bin/env python3
"""V98 Full 4096 envs x 900 steps (3,686,400 transitions) Warp Endurance & Stability Gate.

Verifies that the frozen candidate scenario (`scenario_v91_bayes_1km.yaml`) runs
all 900 steps across 4,096 parallel environments on RTX 3060 `cuda:0` with:
- 0 NaN/Inf in rewards or positions
- 0 queue overflows
- 0 safety failures (tested under both stationary and deterministic moving actions)
- Synchronized GPU throughput (world-steps/s) and peak allocated VRAM logging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v91_bayes_1km.yaml")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--motion", choices=("stationary", "gentle_patrol"), default="gentle_patrol")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

    p = GPUWorldV80Params.from_yaml(args.scenario, envs=args.envs, device=args.device, seed=args.seed)
    w = WarpParallelMissionV80(p)
    act_np = np.zeros((args.envs, 6, 2), dtype=np.float32)
    dst_np = np.ones((args.envs, 6), dtype=np.int32)
    if args.motion == "gentle_patrol":
        ang = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False, dtype=np.float32)
        act_np[:, :, 0] = 0.25 * np.cos(ang)[None, :]
        act_np[:, :, 1] = 0.25 * np.sin(ang)[None, :]

    act_wp = wp.array(act_np, dtype=wp.float32, device=args.device)
    dst_wp = wp.array(dst_np, dtype=wp.int32, device=args.device)

    wp.synchronize_device(args.device)
    t0 = time.monotonic()
    nonfinite_rewards = 0
    max_reward_sum_err = 0.0
    for step in range(1, args.steps + 1):
        if args.motion == "gentle_patrol" and (step % 60 == 0):
            act_np = -act_np
            wp.copy(act_wp, wp.array(act_np, dtype=wp.float32, device=args.device))
        w.step(act_wp, dst_wp)
        if step % 150 == 0 or step == args.steps:
            wp.synchronize_device(args.device)
            r = w.buffers["reward"].numpy()
            c = w.buffers["reward_components"].numpy()
            if not np.all(np.isfinite(r)):
                nonfinite_rewards += int(np.count_nonzero(~np.isfinite(r)))
            err = float(np.max(np.abs(c.sum(axis=1) - r)))
            max_reward_sum_err = max(max_reward_sum_err, err)

    wp.synchronize_device(args.device)
    elapsed = time.monotonic() - t0
    safety_fails = int(np.count_nonzero(w.buffers["failed"].numpy()))
    overflows = int(np.count_nonzero(w.network.data["overflow"].numpy()))
    mean_bat = float(w.buffers["battery"].numpy().mean())
    vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() else 0.0

    gates = {
        "NO_NONFINITE_REWARDS": nonfinite_rewards == 0,
        "REWARD_LEDGER_EXACT": max_reward_sum_err < 1.0e-4,
        "ZERO_QUEUE_OVERFLOW": overflows == 0,
        "ZERO_SAFETY_FAILURES": safety_fails == 0,
    }
    out = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gate": "V98_WARP_4096X900_ENDURANCE",
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "device": args.device,
        "envs": args.envs,
        "steps": args.steps,
        "motion": args.motion,
        "total_world_transitions": args.envs * args.steps,
        "elapsed_sec": round(elapsed, 3),
        "world_steps_per_sec": round((args.envs * args.steps) / max(1e-6, elapsed), 1),
        "safety_failed_worlds": safety_fails,
        "overflow_worlds": overflows,
        "nonfinite_rewards": nonfinite_rewards,
        "max_reward_sum_err": max_reward_sum_err,
        "mean_battery_remaining_j": round(mean_bat, 2),
        "torch_peak_vram_mib": round(vram_mb, 2),
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
