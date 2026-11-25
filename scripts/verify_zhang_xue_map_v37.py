"""Run Zhang-Xue map-only parallel Warp sanity flow: no network, no RL.

Usage:
  python -m scripts.verify_zhang_xue_map_v37 --envs 4096 --steps 5 \
     --seed 44 --device cpu --output-json docs/superpowers/evidence/map_v37.json

For a CUDA device use --device cuda:0 only if Warp sees a real CUDA driver.
This script's E4096 loop is map mechanics, NOT training throughput.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from uav_network.environment.zhang_xue_coop import ZhangXueConfig
from uav_network.environment.zhang_xue_map_geometry import sample_scenario_batch
from uav_network.environment.gpu.zhang_xue_map_warp import ZhangXueWarpMapBatch
from uav_network.environment.gpu.zhang_xue_motion_warp import warp_move_batch


def run_map_flow(*,envs=4096,steps=5,seed=44,device="cpu"):
    if envs<=0 or steps<=0:
        raise ValueError("envs and steps must be positive")
    config=ZhangXueConfig()
    t0=time.perf_counter()
    seeds=np.arange(int(seed),int(seed)+int(envs),dtype=np.int64)
    scenes=sample_scenario_batch(config,seeds)
    scenario_seconds=time.perf_counter()-t0
    positions=scenes["positions"].copy()
    rng=np.random.default_rng(int(seed)+777)
    backend=ZhangXueWarpMapBatch(
        envs,device=device,
        altitude_m=config.fixed_altitude_meter,
        camera_fov_degrees=config.camera_fov_degrees,
        pd=config.pd,pf=config.pf,
    )
    backend.reset()
    blocked_s=0
    blocked_b=0
    t1=time.perf_counter()
    for frame in range(steps):
        actions=rng.uniform(-1,1,size=(envs,6,2)).astype(np.float32)
        positions,velocity,s,b=warp_move_batch(
            positions,scenes["no_fly_zones"],actions,
            dt=config.dt_seconds,max_speed=config.v_max_mps,
            safe_distance=config.safe_distance_meter,device=device,
        )
        blocked_s+=int(np.sum(s))
        blocked_b+=int(np.sum(b))
        backend.step(
            positions,scenes["target_mask"].astype(np.int32),
            seed=int(seed),step_index=frame,
        )
    run_seconds=time.perf_counter()-t1
    state=backend.read()
    per_cell_visited=np.any(state["pos"]+state["neg"]>0,axis=1)
    coverage=np.mean(per_cell_visited,axis=1)
    masses_finite=all(np.isfinite(state[k]).all() for k in (
        "mass_target","unknown","team_mass_target","team_unknown",
    ))
    coords_ok=bool(np.all((positions>=0.) & (positions<config.map_size_meter)))
    targets_ok=bool(np.all(scenes["target_mask"].sum(axis=1)==config.target_count))
    # The map API exposes ONLY the accumulated evidence and DS masses,
    # never a truth copy. This does not itself prove the future actor encoder.
    result={
        "scenario":"zhang_xue_v37_1000m_6uav_10targets_90deg_camera",
        "environment_count":int(envs),
        "steps_per_env":int(steps),
        "total_map_env_steps":int(envs*steps),
        "map_device":device,
        "target_count_valid":targets_ok,
        "targets_per_world":config.target_count,
        "map_sensing_mass_finite":bool(masses_finite),
        "all_map_positions_in_bounds":coords_ok,
        "target_ground_truth_hidden_from_actor_state":(
            "target_mask" not in state and "truth" not in state
        ),
        "coverage_fraction_mean":float(np.mean(coverage)),
        "coverage_fraction_min":float(np.min(coverage)),
        "coverage_fraction_max":float(np.max(coverage)),
        "total_evidence_counts":int(
            state["pos"].sum(dtype=np.int64)+state["neg"].sum(dtype=np.int64)
        ),
        "safety_motion_rejections":blocked_s,
        "boundary_motion_rejections":blocked_b,
        "scene_reset_seconds":float(scenario_seconds),
        "motion_plus_sensing_and_ds_seconds":float(run_seconds),
        "map_env_steps_per_second_cpu_or_device":float((envs*steps)/run_seconds),
        "no_network_or_training":True,
        "important_limitations":[
            "No communication or packet fusion; no MAC/routing/ACK",
            "No actor, critic, replay or gradient update; not training",
            "Static 2D no-fly circles and radio-blackout diamonds",
            "Sensor visibility uses cell centers within 100m nadir footprint",
            "Independent Bernoulli measurements; no correlated sensor noise",
            "Warp CPU results are not a CUDA throughput measurement",
        ],
    }
    if not (coords_ok and targets_ok and masses_finite):
        raise AssertionError("Map flow invariant failed")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs",type=int,default=4096)
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output-json",type=Path)
    args=parser.parse_args()
    result=run_map_flow(
        envs=args.envs,steps=args.steps,seed=args.seed,device=args.device,
    )
    content=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(content+"\n")
    print(content)


if __name__=="__main__":
    main()
