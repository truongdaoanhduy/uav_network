"""Controlled E4096 persistent Warp queue + one-hop MAC + application ACK probe.

Not a full UavNetSim service or a learned controller. Four deterministic
world groups: DATA at 0s, after recovery at 3s, after recovery at 9s,
or never recovers and expires at exactly 10s. The receiving UAV queues a
real-size 90-byte application ACK, delivered on the next simulated step.
One-hop topology is intentionally benign and isolated (no carrier collision).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import (
    ZhangXueWarpPacketQueue,
)


def _payload(envs,src,ids,dest,bytes_,kind,created_us,ttl_us=10_000_000):
    """One packet/UAV or NO_PACKET=-1; all fields [E,6]."""
    msg=np.full((envs,6),-1,np.int64)
    dst=np.full((envs,6),-1,np.int32)
    sz=np.zeros((envs,6),np.int32)
    kinds=np.zeros((envs,6),np.int32)
    created=np.zeros((envs,6),np.int64)
    ttl=np.ones((envs,6),np.int64)
    msg[:,src]=ids
    dst[:,src]=dest
    sz[:,src]=bytes_
    kinds[:,src]=kind
    created[:,src]=created_us
    ttl[:,src]=ttl_us
    return msg,dst,sz,kinds,created,ttl


def run_queued_onehop_e2e_probe(*,envs=4096,seed=44,device="cpu"):
    if envs<4 or envs%4!=0:
        raise ValueError("envs must be a positive multiple of four")
    start=time.perf_counter()
    groups=np.arange(envs,dtype=np.int32)%4
    queue=ZhangXueWarpPacketQueue(envs,capacity=3,device=device)
    rng=np.random.default_rng(seed)
    original_ids=np.arange(1,envs+1,dtype=np.int64)
    packet=_payload(envs,0,original_ids,1,566,0,0)
    admitted=queue.enqueue(*packet)
    assert int((admitted==1).sum())==envs
    positions=np.full((envs,6,3),2500.,np.float32)
    positions[:,:,2]=100.
    positions[:,0,:]=[0.,0.,100.]
    positions[:,1,:]=[400.,0.,100.]
    channels=np.ones((envs,6),np.int32)
    seen_data=set()
    seen_ack=set()
    ack_generated=0
    mac_acks=0
    blackout_send_count=0
    outcomes=[]
    timestamps_us=(0,1_000_000,3_000_000,4_000_000,
                   9_000_000,10_000_000,11_000_000)
    for now in timestamps_us:
        radio=np.ones((envs,6),np.int32)
        radio[:,0]=(
            (groups==0) |
            ((groups==1)&(now>=3_000_000)) |
            ((groups==2)&(now>=9_000_000))
        ).astype(np.int32)
        active=queue.dispatch(now,radio_enabled=radio)
        present=active["packet_id"]>=0
        blackout_send_count+=int(np.count_nonzero(present&(radio==0)))
        if not present.any():
            outcomes.append({"time_us":now,"frame_count":0})
            continue
        deadlines=np.where(present,
            active["created_us"]+active["ttl_us"]-now,
            10_000_000,
        ).astype(np.float32)
        if np.any(deadlines[present]<=0):
            raise AssertionError("Queue dispatched an expired packet")
        slots=np.empty((envs,6,5),np.int32)
        for attempt in range(5):
            slots[:,:,attempt]=rng.integers(
                0,32*(2**attempt)-1,size=(envs,6),
            )
        frame=simulate_one_hop_retry_batch(
            positions,active["dest"].astype(np.int32),
            active["frame_bytes"].astype(np.int32),channels,
            slots,deadlines,blackout=1-radio,device=device,
        )
        receipt=(frame["data_received"]!=0)&present
        mac_success=(frame["mac_acked"]!=0)&present
        mac_acks+=int(mac_success.sum())
        result=queue.resolve(
            active["packet_id"],
            np.where(present,np.where(mac_success,1,2),0).astype(np.int32),
        )
        if not np.all(result[present]==1):
            raise AssertionError("Unresolved packet after one-hop MAC trial")
        # Type DATA=0 originated at source UAV0. Only a received
        # DATA payload can trigger a new application E2E ACK packet.
        data_rcv=receipt&(active["kind"]==0)
        for e,u in zip(*np.nonzero(data_rcv)):
            key=(int(e),int(active["packet_id"][e,u]))
            if key in seen_data:
                raise AssertionError("Duplicate application DATA reward")
            seen_data.add(key)
        ack_msg=np.full((envs,),-1,np.int64)
        e_idx,u_idx=np.nonzero(data_rcv)
        if e_idx.size:
            if np.any(u_idx!=0):
                raise AssertionError("Only UAV0 source DATA expected")
            if np.any(active["dest"][e_idx,u_idx]!=1):
                raise AssertionError("Unexpected packet destination")
            ack_msg[e_idx]=1_000_000+active["packet_id"][e_idx,u_idx]
            ack_payload=_payload(envs,1,ack_msg,0,90,1,now)
            accepted=queue.enqueue(*ack_payload)
            if not np.all(accepted[e_idx,1]==1):
                raise AssertionError("Receiver E2E ACK queue admission failure")
            ack_generated+=len(e_idx)
        # Type ACK=1 is still an application DATA packet at the MAC.
        # Only receipt at original UAV0 clears its app ACK watermark.
        app_ack_rcv=receipt&(active["kind"]==1)
        for e,u in zip(*np.nonzero(app_ack_rcv)):
            ack_id=int(active["packet_id"][e,u])-1_000_000
            assert u==1 and int(active["dest"][e,u])==0
            key=(int(e),ack_id)
            if key not in seen_data:
                raise AssertionError("E2E ACK refers to unreceived data")
            if key in seen_ack:
                raise AssertionError("Duplicate E2E ACK / watermark")
            seen_ack.add(key)
        outcomes.append({
            "time_us":now,
            "frames":int(present.sum()),
            "app_data_receipts":int(data_rcv.sum()),
            "e2e_ack_receipts":int(app_ack_rcv.sum()),
        })
    state=queue.snapshot()
    expired=int(state["expired"].sum())
    if len(seen_data)!=len(seen_ack):
        raise AssertionError("Expected returned E2E ACK for every received DATA")
    return {
        "stage":"v41_multistep_gpu_resident_queue_plus_onehop_mac_and_e2e_ack_smoke",
        "envs":envs,
        "data_generated":envs,
        "data_received":len(seen_data),
        "e2e_ack_generated":ack_generated,
        "e2e_ack_received":len(seen_ack),
        "mac_ack_success":mac_acks,
        "queue_expired":expired,
        "queues_empty_after_final_step":bool(np.all(state["queue_length"]==0)),
        "no_blackout_transmissions":blackout_send_count==0,
        "no_duplicate_e2e_ack_reward":len(seen_ack)==ack_generated,
        "truth_not_serialized":True,
        "queue_device":device,
        "simulated_time_us":list(timestamps_us),
        "per_time_events":outcomes,
        "elapsed_seconds_cpu_or_device":time.perf_counter()-start,
        "training_launched":False,
        "full_gate5":"NOT_PASSED",
        "limitations":[
            "One isolated source DATA to UAV1 and app ACK back to UAV0",
            "Synthetic blackout/recovery schedule, not sampled scenario",
            "No route selection, HELLO or two-hop forwarding",
            "MAC scheduler host transfer per step, not full fused GPU",
            "Application ACK payload content represented by metadata only",
            "No actor, DS map, learned reward or training",
        ],
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs",type=int,default=4096)
    p.add_argument("--seed",type=int,default=44)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output-json",type=Path)
    a=p.parse_args()
    r=run_queued_onehop_e2e_probe(envs=a.envs,seed=a.seed,device=a.device)
    if a.output_json:
        a.output_json.parent.mkdir(parents=True,exist_ok=True)
        a.output_json.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r,indent=2))
