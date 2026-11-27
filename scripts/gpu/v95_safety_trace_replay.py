#!/usr/bin/env python3
"""Read-only CPU replay of first-failure V93 NPZ traces with V94/V95 shield.

This is a *terminal-state replay*, not a proof of full closed-loop viability.
The V93 V94 trace first-terminal fields are snapshots of pre-step positions,
velocities, actions; no hidden ground truth is fed to the controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield, trajectory_safety_violations,
)


def replay(paths: list[Path]) -> dict:
    buildings, _, _ = scale_canonical_geometry(1000.0, 20)
    building_boxes = [tuple(map(float, b[:4])) for b in buildings]
    cases = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"failed", "previous_positions", "previous_velocities",
                        "raw_continuous", "battery", "safety_reason", "episode_step"}
            if not required.issubset(data.files):
                raise ValueError(f"Missing fields in {path}: {sorted(required - set(data.files))}")
            for world in np.flatnonzero(data["failed"]):
                pos = data["previous_positions"][world, :6, :2].astype(np.float64)
                vel = data["previous_velocities"][world].astype(np.float64)
                cmd = np.clip(data["raw_continuous"][world].astype(np.float64), -1.0, 1.0) * 15.0
                alive = data["battery"][world] > 0.0
                safe, _ = kinematic_safety_shield(
                    pos, vel, cmd, alive, building_boxes,
                    map_size_m=1000., dt_s=1., v_max_mps=15.,
                    a_max_mps2=5., uav_safe_distance_m=50.,
                )
                violations = trajectory_safety_violations(
                    pos, safe, alive, building_boxes,
                    map_size_m=1000., dt_s=1., uav_safe_distance_m=50.,
                )
                pair_viable = True
                min_optimistic = float("inf")
                for i in range(6):
                    for j in range(i + 1, 6):
                        a_budget = 5.0 * (int(alive[i]) + int(alive[j]))
                        fx = pos[i] - pos[j] + 2.0 * (safe[i] - safe[j])
                        optimistic = float(np.linalg.norm(fx)) + a_budget
                        min_optimistic = min(min_optimistic, optimistic)
                        if optimistic + 1e-3 < 50.0:
                            pair_viable = False
                physical_safe = not any(violations.values())
                accel_ok = bool(np.all(np.linalg.norm(safe[alive] - vel[alive], axis=1) <= 5.0 + 1e-5))
                speed_ok = bool(np.all(np.linalg.norm(safe[alive], axis=1) <= 15.0 + 1e-5))
                cases.append({
                    "trace": path.name, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "world_id": int(world), "step": int(data["episode_step"][world]),
                    "v93_failure_bits": int(data["safety_reason"][world]),
                    "physical_safe": physical_safe, "next_step_viable_necessary": bool(pair_viable),
                    "acceleration_ok": accel_ok, "speed_ok": speed_ok,
                    "violation_counts": {k: len(v) for k, v in violations.items()},
                    "min_optimistic_next_sep_m": round(min_optimistic, 5),
                })
    return {"scope": "V93 terminal snapshot CPU replay only; NOT full 900-step CUDA rollout",
            "total_failures_from_v93": len(cases),
            "physical_safe_count": sum(c["physical_safe"] for c in cases),
            "necessary_viability_count": sum(c["next_step_viable_necessary"] for c in cases),
            "full_pass_count": sum(c["physical_safe"] and c["next_step_viable_necessary"] and c["acceleration_ok"] and c["speed_ok"] for c in cases),
            "cases": cases}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = replay(args.traces)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k != 'cases'}, indent=2))
    # Red/orange gate: any safety failure remains; never claim SCENARIO_FROZEN.
    return 0 if out["total_failures_from_v93"] and out["full_pass_count"] == out["total_failures_from_v93"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
