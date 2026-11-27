#!/usr/bin/env python3
"""V89 SCENARIO-ONLY acceptance runner. NO learner, replay, optimization or tuning.

Runs physical world, static evidence sensor, packet MAC/PHY and rewards
for the selected number of seconds, without resetting terminal worlds before
metrics can be inspected. Writes a deterministic, inspectable JSON report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"research/zhang_xue/src")]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--envs",type=int,default=4)
    parser.add_argument("--steps",type=int,default=900)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--scenario",type=Path,default=ROOT/"research/zhang_xue/configs/scenario_v89_static_1km.yaml")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
    p=GPUWorldV80Params.from_yaml(args.scenario,envs=args.envs,device=args.device,seed=args.seed)
    w=WarpParallelMissionV80(p)
    a=wp.zeros((args.envs,6,2),dtype=wp.float32,device=args.device)
    d=wp.ones((args.envs,6),dtype=wp.int32,device=args.device)
    start=time.monotonic()
    for _ in range(args.steps):
        w.step(a,d)
    wp.synchronize_device(args.device)
    reward=w.buffers["reward"].numpy().astype(np.float64)
    comps=w.buffers["reward_components"].numpy().astype(np.float64)
    done=w.buffers["done"].numpy()
    episode_step=w.buffers["episode_step"].numpy()
    reasons=w.buffers["termination_reason"].numpy()
    overflow=w.network.data["overflow"].numpy()
    failed=w.buffers["failed"].numpy()
    sources=w.network.data["ledger_pos"].numpy()[:,:6,:,:].astype(np.int64)
    misses=w.network.data["ledger_neg"].numpy()[:,:6,:,:].astype(np.int64)
    generations=w.network.data["ledger_gen"].numpy()
    valid_hello=w.network.data["hello_rx_at_us"].numpy()[:,:6,:6]>=0
    r={
      "status":"SCENARIO_ONLY_COMPLETE",
      "scenario_path":str(args.scenario),
      "scenario_sha256":hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
      "envs":int(args.envs),"steps":int(args.steps),"seed":int(args.seed),
      "device":str(args.device),"elapsed_sec":round(time.monotonic()-start,4),
      "world_horizon":int(p.horizon),"static_raw_count":bool(p.static_raw_count),
      "obs_schema_v2":bool(p.obs_schema_v2),
      "obs_dim_raw":int(w.buffers["obs"].shape[-1]),
      "reward_sum_max_abs_error":float(np.max(np.abs(comps.sum(axis=1)-reward))),
      "reward_nonfinite":int((~np.isfinite(comps)).sum()),
      "overflow_worlds":int(np.count_nonzero(overflow)),
      "physical_safety_failed_worlds":int(np.count_nonzero(failed)),
      "done_worlds":int(np.count_nonzero(done)),
      "horizon_truncated_worlds":int(np.count_nonzero(reasons&8)),
      "fully_delivered_worlds":int(np.count_nonzero(reasons&4)),
      "episode_step_min":int(episode_step.min()),"episode_step_max":int(episode_step.max()),
      "sensor_total_samples_max_origin_cell":int((sources+misses).max()),
      "network_packets_delivered_to_gcs":int(w.network.data["stats_gcs"].numpy().sum()),
      "network_packets_delivered_to_peer":int(w.network.data["stats_peer"].numpy().sum()),
      "hello_received_cache_records":int(np.count_nonzero(valid_hello)),
      "gcs_tp_total":int(w.buffers["gcs_tp"].numpy().sum()),
      "battery_pct_mean":float(w.buffers["battery"].numpy().mean()/p.e_max_j*100.0),
      "truth_min":int(w.buffers["truth"].numpy().sum(axis=1).min()),
      "truth_max":int(w.buffers["truth"].numpy().sum(axis=1).max())
    }
    gates={
        "NO_OVERFLOW":r["overflow_worlds"]==0,
        "NO_PHYSICAL_COLLISION":r["physical_safety_failed_worlds"]==0,
        "REWARD_SUM_PARITY":r["reward_sum_max_abs_error"]<1e-4 and r["reward_nonfinite"]==0,
        "TRUTH_TARGETS":r["truth_min"]==r["truth_max"]==15,
        "SENSOR_COUNTER_BOUND":r["sensor_total_samples_max_origin_cell"]<=args.steps,
        "REAL_RF_PACKET_DELIVERY":r["network_packets_delivered_to_gcs"]>0,
        "HELLO_RECEIVED":r["hello_received_cache_records"]>0,
    }
    if args.steps==p.horizon:
        gates["HORIZON_TERMINATION"]=r["horizon_truncated_worlds"]==args.envs and r["episode_step_min"]==p.horizon
    else:
        gates["NO_EARLY_TERMINATION"]=r["done_worlds"]==0
    r["gates"]=gates
    r["status"]="SCENARIO_GATE_PASS" if all(gates.values()) else "SCENARIO_GATE_FAIL"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps(r,indent=2))
    return 0 if all(gates.values()) else 1

if __name__=="__main__":
    raise SystemExit(main())
