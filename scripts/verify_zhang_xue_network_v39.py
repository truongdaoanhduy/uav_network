"""Reproducible Gate5G check: real pinned UavNetSim-v1 vs bounded Warp retry.

Do NOT use this comparison as proof of identical per-packet RNG or event
schedules. The native test includes routing/HELLO/SimPy and starts packet
generation at t=25ms; the Warp prototype does neither.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CLASSIC=ROOT/"outputs/reference_sources/UavNetSim-v1-classic"


def _six_seed_warp():
    seeds=list(range(44,50))
    e=len(seeds)
    pos=np.full((e,6,3),2500.,np.float32)
    pos[:,:,2]=100.
    pos[:,0,:]=[0,0,100.]
    pos[:,1,:]=[400,0,100.]
    pos[:,2,:]=[800,0,100.]
    dst=np.full((e,6),-1,np.int32)
    dst[:,[0,2]]=1
    sizes=np.zeros((e,6),np.int32)
    sizes[:,[0,2]]=566
    ch=np.ones((e,6),np.int32)
    slots=np.zeros((e,6,5),np.int32)
    for i,seed in enumerate(seeds):
        for u in (0,2):
            rr=random.Random(seed+u+5)
            for a in range(5):
                slots[i,u,a]=rr.randint(0,32*(2**a)-2)
    exp=np.full((e,6),10e6,np.float32)
    tic=time.perf_counter()
    result=simulate_one_hop_retry_batch(pos,dst,sizes,ch,slots,exp)
    cpu_elapsed=time.perf_counter()-tic
    return {
        "seeds":seeds,
        "data_pdr_by_seed":(
            result["delivered"][:,[0,2]].sum(axis=1)/2.
        ).tolist(),
        "num_attempts_by_seed":result["attempts"][:,[0,2]].sum(axis=1).tolist(),
        "data_collision_events_by_seed":result["collided"][:,[0,2],:].sum(axis=(1,2)).tolist(),
        "ack_collision_events_by_seed":result["ack_collided"][:,[0,2],:].sum(axis=(1,2)).tolist(),
        "cpu_seconds":cpu_elapsed,
    }


def _native_case():
    v1=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",DEFAULT_CLASSIC))
    if not (v1/"entities/drone.py").exists():
        raise FileNotFoundError("Classic pinned UavNetSim-v1 missing")
    env=os.environ.copy()
    env["UAV_CLASSIC_V1_ROOT"]=str(v1)
    env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
    with tempfile.TemporaryDirectory() as tmp:
        proc=subprocess.run(
            [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
             "--case","collision","--json"],
            cwd=tmp,env=env,text=True,capture_output=True,timeout=90,
        )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run():
    native=_native_case()
    warped=_six_seed_warp()
    native_pdr=native["native_pdr_by_seed"]
    warp_pdr=warped["data_pdr_by_seed"]
    mismatch=sum(abs(a-b)>1e-9 for a,b in zip(native_pdr,warp_pdr))
    return {
        "stage":"Gate5G_v39_warp_cpu_bounded_retries_and_ack_interference",
        "source_pin":native["source_sha"],
        "native_engine":native["engine"],
        "warp_device":"cpu",
        "native_pdr_by_seed":native_pdr,
        "warp_pdr_by_seed":warp_pdr,
        "native_attempts_by_seed":native["native_tx_attempts_by_seed"],
        "warp_attempts_by_seed":warped["num_attempts_by_seed"],
        "pdr_seed_mismatches":mismatch,
        "warp_data_collisions_by_seed":warped["data_collision_events_by_seed"],
        "warp_ack_collisions_by_seed":warped["ack_collision_events_by_seed"],
        "warp_first_compile_inclusive_seconds":warped["cpu_seconds"],
        "full_gate5":"FAIL_NOT_PARITY_APPROVED",
        "training_permitted":False,
        "scope":"one-hop isolated frame per sender; no HELLO/routing/queues/E2E ACK; no CUDA",
        "unmatched_randomness":"Python RNG initialized with native seed formula, but native HELLO and routing consume draws; do not treat seeds as matched on-air MAC draws",
        "next":"extract exact timestamped native MAC trace and match traffic, then implement queues, multi-hop E2E ACK, and map-application fusion before train",
    }


if __name__=="__main__":
    report=run()
    out=ROOT/"docs/superpowers/evidence/zhang_xue_v39_simpy_vs_warp_retry_cpu_20261125.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
