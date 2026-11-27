#!/usr/bin/env python3
"""Bounded DATA/ACK collision, hidden-terminal and CPU/CUDA event parity audit.

32 matched seeds here compare this project's Warp CPU and CUDA executions.
This is explicitly NOT a matched UavNetSim macro validation or RF calibration.
Synthetic gains separate MAC contention from obstacle/link-budget assumptions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/"research/zhang_xue/src")]


def simulate(device, seeds=32, *, hidden=False, forced_tie=False):
    import warp as wp
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.state_v67 import NEVER_US, ST_BACKOFF
    net = WarpCompressedRadioNetworkV67(seeds, num_uavs=2, grid_w=2, device=device, seed=44)

    def put(name, data):
        a=net.data[name]
        wp.copy(a,wp.array(np.asarray(data),dtype=a.dtype,device=device))

    gain=np.full((seeds,3,3),1e-4,dtype=np.float32)
    gain[:,np.arange(3),np.arange(3)]=0
    if hidden:
        gain[:,0,1]=gain[:,1,0]=0
    net.set_channel_gain(gain)
    put("hello_due",np.full((seeds,3),NEVER_US,dtype=np.int32))
    hop=net.data["route_hop"].numpy()
    expiry=net.data["route_expiry"].numpy()
    dist=net.data["route_dist"].numpy()
    masks=net.data["route_mask"].numpy()
    for src in range(3):
        for dst in range(3):
            if src!=dst:
                hop[:,src,dst]=dst;expiry[:,src,dst]=NEVER_US
                dist[:,src,dst]=1;masks[:,src,dst]=(1<<src)|(1<<dst)
    for key,a in [("route_hop",hop),("route_expiry",expiry),("route_dist",dist),("route_mask",masks)]:
        put(key,a)
    net.sense_batch(np.tile([0,1],(seeds,1)).astype(np.int32),np.ones((seeds,2),dtype=np.int32),1)
    net.inject_compact3(np.ones((seeds,2),dtype=np.int32),np.zeros((seeds,2,2),dtype=np.float32))
    state=net.data["state"].numpy();deadline=net.data["deadline"].numpy()
    draws=[]
    for e in range(seeds):
        rng=np.random.default_rng(44+e)
        slots=np.zeros(2,dtype=int) if forced_tie else rng.integers(0,32,size=2)
        draws.append(slots.tolist());state[e,:2]=ST_BACKOFF;deadline[e,:2]=100+20*slots
    put("state",state);put("deadline",deadline)
    # At 450us tied 312us DATA frames have ended; the first ACK timeout is 592us.
    net.step(duration_us=450)
    initial_gcs=net.data["stats_gcs"].numpy().copy()
    net.step(duration_us=50_000)
    keys=("stats_gcs","stats_mac_ack","stats_e2e","stats_retry","stats_drop","stats_ttl","overflow","q_count")
    arrays={k:net.data[k].numpy().astype(np.int64).tolist() for k in keys}
    return {"device":device,"worlds":seeds,"hidden":hidden,"forced_tie":forced_tie,
            "offered_data_packets":2*seeds,"initial_slots":draws,
            "gcs_before_first_retry":initial_gcs.astype(int).tolist(),
            "delivered_data_packets":int(sum(arrays["stats_gcs"])),
            "pdr":float(sum(arrays["stats_gcs"])/(2*seeds)),
            "retries":int(sum(arrays["stats_retry"])),
            "drops":int(sum(arrays["stats_drop"])),"raw":arrays}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seeds",type=int,default=32)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.seeds<=256:
        p.error("seeds must be in [1,256]")
    runs={}
    for label,hidden,tied in [("clear",False,False),("hidden",True,False),("same_slot",False,True)]:
        cpu=simulate("cpu",a.seeds,hidden=hidden,forced_tie=tied)
        gpu=simulate(a.device,a.seeds,hidden=hidden,forced_tie=tied)
        runs[label]={"cpu":cpu,"device":gpu,"integer_event_parity":cpu["raw"]==gpu["raw"] and cpu["gcs_before_first_retry"]==gpu["gcs_before_first_retry"]}
    gates={
        "MATCHED_CPU_CUDA_EVENT_COUNTERS":all(r["integer_event_parity"] for r in runs.values()),
        "SAME_SLOT_DATA_COLLISION":all(x==0 for x in runs["same_slot"]["device"]["gcs_before_first_retry"]),
        "SAME_SLOT_RETRANSMISSIONS":runs["same_slot"]["device"]["retries"]>=2*a.seeds,
        "HIDDEN_TERMINAL_EXCESS_RETRIES":runs["hidden"]["device"]["retries"]>runs["clear"]["device"]["retries"],
        "NO_EVENT_OVERFLOW":all(not any(r["device"]["raw"]["overflow"]) for r in runs.values()),
    }
    out={"status":"PASS" if all(gates.values()) else "FAIL","gates":gates,"runs":runs,
         "upstream_macro_parity":"OPEN: routing/payload/RNG/HELLO differ from UavNetSim",
         "scope":"Synthetic-gain two-source DATA/ACK microprofile, not saturation throughput or field calibration"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
    return 0 if all(gates.values()) else 2


if __name__=="__main__":
    raise SystemExit(main())
