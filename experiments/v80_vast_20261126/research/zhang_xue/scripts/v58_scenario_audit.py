#!/usr/bin/env python3
"""Read-only v5.8 research audit, deliberately NOT a GPU/training launcher."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zhang_xue.compat.env import make_v57_config
from zhang_xue.compat.network import CANONICAL_BUILDINGS_5
from zhang_xue.compat.warp_mac import backing_bytes_per_environment, SHAPES
from zhang_xue.gpu.kernels.phy import v57_gpu_memory_budget

# Separate research scenario supplied in the UAV project discussions.
STUDY_RESEARCH_3000 = {
    "map_size_m": 3000.0,
    "uav_count": 6,
    "target_count": 35,
    "belief_grid_side": 40,
    "obstacle_count": 25,
    "max_speed_mps": 20.0,
    "max_accel_mps2": 2.0,
    "horizon_steps": 1500,
    "battery_capacity_j": 277200.0,
    "confirm_threshold": 0.99,
    "gcs_target_exclusion_m": 150.0,
    "target_min_separation_m": 250.0,
    "tx_power_max_w": 0.4,
}


def collect_audit() -> dict:
    cfg = make_v57_config()
    actual = {
        "map_size_m": cfg.map_size_m,
        "uav_count": 6,
        "target_count": cfg.num_targets,
        "belief_grid_side": 20,
        "obstacle_count": len(CANONICAL_BUILDINGS_5),
        "max_speed_mps": cfg.v_max_mps,
        "max_accel_mps2": cfg.a_max_mps2,
        "horizon_steps": cfg.safety_horizon_s,
        "battery_capacity_j": cfg.e_max_j,
        "confirm_threshold": cfg.confirm_th,
        "gcs_target_exclusion_m": cfg.gcs_target_exclusion_m,
        "target_min_separation_m": cfg.target_min_separation_m,
        "tx_power_max_w": cfg.tx_power_w,
    }
    mismatch = {k: {"study_3000": v, "zip_v57": actual[k]} for k, v in STUDY_RESEARCH_3000.items() if actual[k] != v}
    budget = v57_gpu_memory_budget()
    return {
        "zip_version": "v5.8 audited continuation of v5.7",
        "scenario_config_actual": actual,
        "original_study_profile": STUDY_RESEARCH_3000,
        "scenario_mismatches": mismatch,
        "warp_cpu_cuda_event_parity_verified": False,
        "cuda_t4_executed_in_this_artifact": False,
        "warp_phy_device_connected_to_packet_net": False,
        "gpu_motion_battery_sensing_rewards_end_to_end": False,
        "sample_level_ewma_transmitted_to_receiver": False,
        "network_allocated_bytes_per_world": backing_bytes_per_environment(),
        "network_backing_buffers_count": len(SHAPES),
        "network_memory_planning_only": budget,
        "production_training_allowed": False,
        "blockers": [
            "P0: study-3000 scenario mismatch; v5.7 is a different benchmark",
            "P0: no Warp CPU-CUDA packet-by-packet parity",
            "P0: Warp HOL source fix not compiled/executed on Warp in this ZIP",
            "P0: not all motion/sensor/reward/training kernels exist on CUDA",
            "P0: trajectory safety detector flags infeasible dynamics but cannot prevent collision",
            "P1: per-sample AoI EWMA not carried by existing packed 8-byte protocol",
            "P1: only memory lower bounds / provisional CUDA overhead, no T4 measurement",
            "P1: fixed radio positions during one-second network event period",
            "P1: dynamic TX-power control absent (tx_power_w constant)",
            "P1: Pd/Pf fixed rather than altitude-dependent table from study",
            "P1: target/GCS and target/target spacing differ from original study",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write a JSON audit alongside stdout")
    parser.add_argument("--strict", action="store_true", help="Return 2 while training blockers remain")
    args = parser.parse_args()
    audit = collect_audit()
    value = json.dumps(audit, indent=2, ensure_ascii=False)
    print(value)
    if args.output:
        Path(args.output).write_text(value + "\n", encoding="utf-8")
    return 2 if args.strict and audit["blockers"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
