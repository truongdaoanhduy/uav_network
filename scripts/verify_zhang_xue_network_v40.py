"""Gate 5H: real native UavNetSim-v1 MAC RNG replay versus bounded Warp.

Difference between local UavNetSim adapter PDR and original source ACK
success was fixed in v4.0; now compare ground-truth native AppData delivery.
The matched seed is *the actual upstream MAC stream after HELLO*, not merely
random.Random(seed), which can have different consumed-prefix draws.

This verifies one-hop independent packet trials only; do NOT train on it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from uav_network.communication.gpu.zhang_xue_mac_retry_warp import simulate_one_hop_retry_batch

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_V1_ROOT=ROOT/"outputs/reference_sources/UavNetSim-v1-classic"


def _native_trace(start_seed, num_seeds):
    v1=os.environ.get("UAV_CLASSIC_V1_ROOT",str(DEFAULT_V1_ROOT))
    with tempfile.TemporaryDirectory(prefix="v40_native_uavnetsim_") as tmp:
        env={
            **os.environ,
            "UAV_CLASSIC_V1_ROOT":v1,
            "PYTHONPATH":str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH",""),
        }
        proc=subprocess.run(
            [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
             "--case","collision_trace","--trace-start-seed",str(start_seed),
             "--trace-num-seeds",str(num_seeds),"--json"],
            env=env,cwd=tmp,text=True,capture_output=True,timeout=120,
        )
    if proc.returncode:
        raise RuntimeError("Native SimPy trace failed: " + proc.stderr[-2400:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_parity(*,start_seed=44,num_seeds=6,device="cpu"):
    native=_native_trace(start_seed,num_seeds)
    n=len(native["seeds"])
    xyz=np.full((n,6,3),2500.,np.float32)
    xyz[:,:,2]=100.
    xyz[:,0,:]=[0.,0.,100.]
    xyz[:,1,:]=[400.,0.,100.]
    xyz[:,2,:]=[800.,0.,100.]
    dst=np.full((n,6),-1,np.int32)
    dst[:,[0,2]]=1
    sizes=np.zeros((n,6),np.int32)
    sizes[:,[0,2]]=566
    channels=np.ones((n,6),np.int32)
    drawn_slots=np.zeros((n,6,5),np.int32)

    bounds_ok=True
    draw_counts=np.zeros((n,6),np.int32)
    for e,draws in enumerate(native["mac_draws_by_seed"]):
        for u in (0,2):
            # The first five actual DATA backoff draws, after native
            # startup HELLO; may be other draws from relay E2E ACK.
            relevant=[d for d in draws if d["uav_id"]==u]
            draw_counts[e,u]=min(len(relevant),5)
            # An early native ACK legitimately ends attempts before #5.
            # Unseen retry draws MUST be marked censored if Warp needs them.
            for k,x in enumerate(relevant[:5]):
                cw=32*(1<<k)-1
                bounds_ok &= (x["lo"]==0 and x["hi"]==cw-1 and 0<=x["slot"]<cw)
                drawn_slots[e,u,k]=x["slot"]
    if not bounds_ok:
        raise AssertionError("Native MAC draw windows have unexpected bounds")
    got=simulate_one_hop_retry_batch(
        xyz,dst,sizes,channels,drawn_slots,
        np.full((n,6),10e6,np.float32),device=device,
    )
    censored=int(np.maximum(
        got["attempts"][:,[0,2]]-draw_counts[:,[0,2]],0
    ).sum())
    native_pdr=[float(x) for x in native["native_pdr_by_seed"]]
    warp_pdr=(got["data_received"][:,[0,2]].sum(axis=1)/2.).tolist()
    trace_errors=[]
    first_errors=[]
    native_attempts=[]
    native_first=[]
    for e,events in enumerate(native["events_by_seed"]):
        per_world=0
        for u in (0,2):
            starts=np.array([
                x["time_us"]-25000.
                for x in events if x["type"]=="data_tx" and x["uav_id"]==u
            ],np.float64)
            num_starts=int(got["attempts"][e,u])
            if len(starts)!=num_starts:
                # Mismatch in number of attempts is itself a parity issue.
                errors=[float("inf")]
            else:
                errors=np.abs(starts-got["start_us"][e,u,:num_starts])
            trace_errors.extend(errors)
            if len(starts):
                first_errors.append(abs(float(starts[0]-got["start_us"][e,u,0])))
                native_first.append(float(starts[0]))
            per_world+=len(starts)
        native_attempts.append(per_world)
    physical=native["native_delivered_ids_by_seed"]
    metric=native["metric_delivered_ids_by_seed"]
    score_from_ids=[len(s)/2 for s in physical]
    source_metric_ok=(physical==metric and native_pdr==score_from_ids)
    return {
        "stage":"Gate5H_native_rng_replay_v40",
        "native_source_sha":native["source_sha"],
        "seeds":native["seeds"],
        "mac_backoff_matches_actual_native_rng":bool(bounds_ok and censored==0),
        "native_draw_counts_by_seed":draw_counts[:,[0,2]].tolist(),
        "censored_warp_retry_count":censored,
        "native_metric_matches_physical_data_delivery":bool(source_metric_ok),
        "native_pdr_by_seed":native_pdr,
        "warp_pdr_by_seed":warp_pdr,
        "pdr_seed_mismatch_count":int(np.count_nonzero(
            ~np.isclose(native_pdr,warp_pdr)
        )),
        "data_tx_attempt_total_native":int(sum(native_attempts)),
        "data_tx_attempt_total_warp":int(got["attempts"][:,[0,2]].sum()),
        "max_first_attempt_start_error_us":float(max(first_errors)),
        "max_any_attempt_start_error_us":float(max(trace_errors)),
        "nonzero_attempt_start_error_count":int(np.count_nonzero(
            np.asarray(trace_errors)>1e-4
        )),
        "native_app_delivered_ids":physical,
        "warp_MAC_ack_collisions":int(got["ack_collided"].sum()),
        "scope_notes":[
            "Only two app sources and one receiver, both on 802.11b channel 1",
            "RNG inputs are captured from the actual native per-UAV instances",
            "Native packets begin at 25ms; Warp times are relative to 0ms",
            "The original app metric type bug is corrected in adapter",
            "No actual packet queue, HELLO, routing, E2E ACK or CUDA verification",
        ],
        "full_gate5_status":"NOT_PASSED_ONE_HOP_ONLY",
        "training_permitted":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed",type=int,default=44)
    parser.add_argument("--num-seeds",type=int,default=6)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output-json",type=Path)
    args=parser.parse_args()
    report=run_parity(start_seed=args.start_seed,num_seeds=args.num_seeds,device=args.device)
    print(json.dumps(report,indent=2,sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
