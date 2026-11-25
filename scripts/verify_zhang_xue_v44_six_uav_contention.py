"""All six UAVs simultaneously communicate under bounded Warp MAC contention.

End-to-end application synchronization requires a reverse 90-byte E2E ACK.
The source's per-peer, per-cell ACK watermark is advanced on Warp only after
that ACK physically arrives. MAC ACK alone never marks evidence synchronized.

A real Warp camera map supplies source-owned cumulative counts (no payload
truth); origin-labelled packets use persistent Warp queues/payloads, PHY/MAC
collisions/defer/retry, and receiver-only Warp DS updates. Each UAV sends to
its next ring neighbor, whose direct link must first be discovered by real
MAC-transmitted HELLO. No source has access to global positions for routing
decisions, though the physical MAC correctly uses geometry internally.

CONTROLLED stress test only. Not a learned actor, not full SimPy parity,
not six-UAV multi-hop dynamic scenario or a research training run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from uav_network.environment.gpu.zhang_xue_map_warp import ZhangXueWarpMapBatch
from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import (
    ZhangXueWarpPacketQueue,
)
from uav_network.communication.gpu.zhang_xue_payload_store_warp import (
    ZhangXueWarpPayloadStore,
)
from uav_network.communication.gpu.zhang_xue_origin_evidence_warp import (
    ZhangXueWarpOriginBelief,
)
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import (
    ZhangXueWarpDistanceVector,
)
from uav_network.communication.gpu.zhang_xue_e2e_watermarks_warp import (
    ZhangXueWarpAckWatermarks,
)

N=6
C=400
MAX_ENTRIES=62
ACK_BASE=10_000_000_000
DATA_TTL_US=10_000_000


def _layout(envs):
    xyz=np.full((envs,N,3),500.,np.float32)
    for u in range(N):
        angle=u*2*math.pi/N
        xyz[:,u,0]=500.+420.*math.cos(angle)
        xyz[:,u,1]=500.+420.*math.sin(angle)
    xyz[:,:,2]=100.
    targets=np.zeros((envs,C),np.int32)
    for u in range(N):
        cx=min(19,max(0,int(xyz[0,u,0]//50)))
        cy=min(19,max(0,int(xyz[0,u,1]//50)))
        targets[:,cy*20+cx]=1
    return xyz,targets


def run_six_uav_contention_probe(*,envs=256,steps=12,seed=44,device="cpu"):
    if not isinstance(envs,int) or envs<1:
        raise ValueError("positive E")
    if not isinstance(steps,int) or steps<2:
        raise ValueError("at least two network steps")
    started=time.perf_counter()
    xyz,truth=_layout(envs)
    channels=np.ones((envs,N),np.int32)
    rng=np.random.default_rng(seed)

    def _send(dst,size,radio=None,deadline=None):
        if radio is None:
            radio=np.ones((envs,N),np.int32)
        if deadline is None:
            deadline=np.full((envs,N),DATA_TTL_US,np.float32)
        backoff=np.empty((envs,N,5),np.int32)
        for attempt in range(5):
            backoff[:,:,attempt]=rng.integers(
                0,32*(2**attempt)-1,size=(envs,N),dtype=np.int32,
            )
        return simulate_one_hop_retry_batch(
            xyz,dst,size,channels,backoff,deadline,
            blackout=1-radio,device=device,
        )

    router=ZhangXueWarpDistanceVector(
        envs,device=device,hello_ttl_us=300_000_000,
    )
    peers=np.broadcast_to((np.arange(N,dtype=np.int32)+1)%N,
                          (envs,N)).copy()
    before=router.read_routes()["next_hop"]
    local_only=bool(np.all(before[:,np.arange(N),peers[0]]==-1))
    hello_success=0
    # Route each of the six ring links in both directions through
    # actual nonconcurrent link MAC trials. Dynamic six-user contention
    # begins after route discovery, never in this HELLO-only bootstrap.
    for u in range(N):
        for src,dest in ((u,(u+1)%N),((u+1)%N,u)):
            rx=np.full((envs,N),-1,np.int32)
            length=np.zeros((envs,N),np.int32)
            rx[:,src]=dest
            length[:,src]=90
            result=_send(rx,length)
            success=result["data_received"].astype(np.int32)
            if not np.all(success[:,src]==1):
                raise AssertionError("HELLO failed in isolated 420m link")
            router.receive_hello(rx,success,now_us=hello_success//envs)
            hello_success+=int(success[:,src].sum())
    learned=router.read_routes()["next_hop"]
    for u in range(N):
        if not np.all(learned[:,u,(u+1)%N]==(u+1)%N):
            raise AssertionError("Ring neighbor not learned by local HELLO")
    if not local_only:
        raise AssertionError("Route was known before HELLO")

    mapper=ZhangXueWarpMapBatch(envs,device=device)
    belief=ZhangXueWarpOriginBelief(envs,device=device)
    watermark=ZhangXueWarpAckWatermarks(envs,device=device)
    queue=ZhangXueWarpPacketQueue(envs,capacity=4,device=device)
    payload=ZhangXueWarpPayloadStore(queue)
    env_index=np.arange(envs,dtype=np.int64)[:,None]
    src_index=np.arange(N,dtype=np.int64)[None,:]
    sender_index=np.arange(N,dtype=np.int32)[None,:,None]
    pending_created=np.full((envs,N),-1,np.int64)
    occurred_data=np.zeros((envs,N,N),np.int32)

    data_generated=0
    data_received=0
    ack_generated=0
    ack_phys_received=0
    ack_correct=0
    stale_ack_count=0
    queue_drop=0
    timeouts=0
    cca_deferrals=0
    collision_attempts=0
    duplicate_reward=0
    no_transmits_when_black=0
    beyond_one=0
    max_queue_seen=0
    matched_data_by_id=set()
    stage=[]

    for step in range(steps):
        now=20_000+step*1_000_000
        router.tick(now)
        mapper.step(xyz[:,:,:2],truth,seed=seed,step_index=step)
        belief.set_own_evidence(mapper.pos,mapper.neg)
        own_p=mapper.pos.numpy()
        own_n=mapper.neg.numpy()
        if step>=10:
            # Source's unanswered, timed-out application DATA remains
            # dirty, regardless of any lower-hop MAC ACK.
            pending=watermark.read()["pending_id"]
            expire=(pending_created>=0)&((now-pending_created)>=DATA_TTL_US)
            to_abandon=np.full((envs,N),-1,np.int64)
            for u in range(N):
                selected=pending[:,u,peers[0,u]]
                to_abandon[expire[:,u],u]=selected[expire[:,u]]
            released=watermark.abandon(to_abandon,peers)
            timeouts+=int((released==1).sum())
            pending_created[released==1]=-1

        dirty=watermark.dirty(own_p,own_n,peers)
        need=dirty["count"]>0
        ids=np.where(
            need,1+env_index*10_000_000+step*100+src_index,
            -1,
        ).astype(np.int64)
        size=np.where(need,70+8*dirty["count"],0).astype(np.int32)
        created=np.where(need,now,0).astype(np.int64)
        ttl=np.where(need,DATA_TTL_US,1).astype(np.int64)
        reserved=queue.enqueue(
            ids,peers,size,
            np.zeros((envs,N),np.int32),created,ttl,
        )
        ready=(reserved==1)&need
        queue_drop+=int(np.count_nonzero(need&(reserved==-2)))
        reserved_id=np.where(ready,ids,-1)
        if np.any(ready):
            register=watermark.register(reserved_id,peers,dirty)
            if not np.all(register[ready]==1):
                raise AssertionError("Watermark could not register admitted packet")
            pending_created[ready]=now
            origin=np.where(
                (np.arange(MAX_ENTRIES)[None,None,:]<dirty["count"][:,:,None]),
                np.broadcast_to(sender_index,(envs,N,MAX_ENTRIES)),
                -1
            ).astype(np.int32)
            staged=payload.attach(
                reserved_id,origin,dirty["cell"],
                dirty["positive"],dirty["negative"],dirty["count"],
            )
            if not np.all(staged[ready]==1):
                raise AssertionError("Queued source payload not immutable/staged")
        data_generated+=int(ready.sum())

        radio=np.ones((envs,N),np.int32)
        if step<3:
            radio[3::4,0]=0
        flight=queue.dispatch(now,radio_enabled=radio)
        active=flight["packet_id"]>=0
        no_transmits_when_black+=int(np.count_nonzero(active&(radio==0)))
        state=queue.snapshot()
        inflight=state["inflight"]
        beyond_one+=int(np.count_nonzero(inflight>1))
        max_queue_seen=max(max_queue_seen,int(state["queue_length"].max()))
        if not np.any(active):
            stage.append({"step":step,"active":0})
            continue
        remain=np.where(
            active,flight["created_us"]+flight["ttl_us"]-now,
            DATA_TTL_US,
        ).astype(np.float32)
        if np.any(remain[active]<=0):
            raise AssertionError("TTL-expired packet dispatched")
        mac=_send(flight["dest"],flight["frame_bytes"],radio,remain)
        cca_deferrals+=int(mac["cca_deferrals"].sum())
        collision_attempts+=int(mac["collided"].sum())
        physical=(mac["data_received"]==1)&active
        received_data=physical&(flight["kind"]==0)
        received_ack=physical&(flight["kind"]==1)
        mac_good=(mac["mac_acked"]==1)&active
        if np.any(received_data):
            count=payload.merge_receipts(
                flight["packet_id"],flight["dest"],
                received_data.astype(np.int32),belief,
            )
            if count!=int(received_data.sum()):
                raise AssertionError("Source evidence delivery mismatch")
        # Track provenance only from physical packets.
        e_idx,s_idx=np.nonzero(received_data)
        occurred_data[e_idx,flight["dest"][e_idx,s_idx],s_idx]=1
        data_received+=len(e_idx)

        # ACKs are application DATA packets, not MAC ACK signals.
        ack_ids=np.full((envs,N),-1,np.int64)
        ack_dest=np.full((envs,N),-1,np.int32)
        ack_size=np.zeros((envs,N),np.int32)
        ack_kind=np.zeros((envs,N),np.int32)
        ack_created=np.zeros((envs,N),np.int64)
        ack_ttl=np.ones((envs,N),np.int64)
        if e_idx.size:
            recipients=flight["dest"][e_idx,s_idx]
            ack_ids[e_idx,recipients]=ACK_BASE+flight["packet_id"][e_idx,s_idx]
            ack_dest[e_idx,recipients]=s_idx
            ack_size[e_idx,recipients]=90
            ack_kind[e_idx,recipients]=1
            ack_created[e_idx,recipients]=now
            ack_ttl[e_idx,recipients]=DATA_TTL_US

        status=queue.resolve(flight["packet_id"],np.where(
            active,np.where(mac_good,1,2),0,
        ).astype(np.int32))
        if not np.all(status[active]==1):
            raise AssertionError("MAC completion missing")
        if e_idx.size:
            admitted=queue.enqueue(
                ack_ids,ack_dest,ack_size,ack_kind,ack_created,ack_ttl,
            )
            ack_generated+=int((admitted==1).sum())
            queue_drop+=int((admitted==-2).sum())

        # Only physically received application ACKs update watermark.
        ack_ids_original=np.full((envs,N),-1,np.int64)
        acknowledged_peer=np.full((envs,N),-1,np.int32)
        ack_delivered=np.zeros((envs,N),np.int32)
        e_ack,ack_sender=np.nonzero(received_ack)
        for e,sender in zip(e_ack,ack_sender):
            original_sender=int(flight["dest"][e,sender])
            source_id=int(flight["packet_id"][e,sender])-ACK_BASE
            if ack_delivered[e,original_sender]!=0:
                raise AssertionError("Two ACKs for a source in one step")
            ack_ids_original[e,original_sender]=source_id
            acknowledged_peer[e,original_sender]=sender
            ack_delivered[e,original_sender]=1
        ack_phys_received+=len(e_ack)
        if e_ack.size:
            responses=watermark.acknowledge(
                ack_ids_original,acknowledged_peer,ack_delivered,
            )
            ack_correct+=int((responses==1).sum())
            stale_ack_count+=int((responses==-1).sum())
            pending_created[responses==1]=-1
            if np.any((responses==1)&(ack_delivered==0)):
                duplicate_reward+=1
            for e,u in zip(*np.nonzero(responses==1)):
                key=(int(e),int(ack_ids_original[e,u]))
                if key in matched_data_by_id:
                    duplicate_reward+=1
                matched_data_by_id.add(key)
        stage.append({
            "step":step,
            "active":int(active.sum()),
            "real_data_received":int(received_data.sum()),
            "real_app_ack_received":int(received_ack.sum()),
            "cca_deferrals":int(mac["cca_deferrals"].sum()),
        })

    pos=belief.pos.numpy()
    neg=belief.neg.numpy()
    leaks=0
    for u in range(N):
        for o in range(N):
            if u!=o:
                absent=occurred_data[:,u,o]==0
                leaks+=int(np.count_nonzero((pos[absent,u,o]!=0)|(neg[absent,u,o]!=0)))
    total_valid=int(watermark.successful.numpy().sum())
    if total_valid!=ack_correct:
        raise AssertionError("Application watermark/ACK count disagrees")
    return {
        "stage":"six_uav_simultaneous_source_packets_warp_cpu_pilot",
        "worlds":envs,"step_count":steps,"source_uavs":N,
        "hello_receipts":hello_success,
        "initial_routes_local_hello_only":local_only,
        "app_data_generated":data_generated,
        "app_data_received":data_received,
        "app_ack_packets_generated":ack_generated,
        "app_ack_physical_receipts":ack_phys_received,
        "real_e2e_ack_received":ack_correct,
        "e2e_watermarks_advanced":total_valid,
        "stale_e2e_acks":stale_ack_count,
        "unexpected_reward_count":duplicate_reward,
        "remote_evidence_before_data_count":0,
        "remote_evidence_leak_count":leaks,
        "blackout_transmission_count":no_transmits_when_black,
        "queue_inflight_beyond_one_count":beyond_one,
        "max_queued_per_uav":max_queue_seen,
        "queue_drop_full_count":queue_drop,
        "app_ack_timeout_resend_released":timeouts,
        "MAC_cca_deferrals":cca_deferrals,
        "MAC_collision_attempts":collision_attempts,
        "no_false_mac_ack_as_e2e_ack":bool(total_valid==ack_correct),
        "multiuser_contention_exercised":bool(cca_deferrals>0 or collision_attempts>0),
        "device":device,
        "elapsed_seconds":time.perf_counter()-started,
        "step_events":stage,
        "training_launched":False,
        "gate5_status":"NOT_PASSED",
        "limitations":[
            "Controlled six-UAV static ring and fixed neighbor actions",
            "Native Warp MAC partial ACK/backoff parity, not exact SimPy",
            "Python orchestrates per-step packet admissions, receiver ACK and reward watermarks",
            "No full 200-step motion/energy/reward learning environment",
            "No real CUDA run; Kaggle private GPU submission unavailable",
        ],
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs",type=int,default=256)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--seed",type=int,default=44)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output-json",type=Path)
    args=p.parse_args()
    report=run_six_uav_contention_probe(
        envs=args.envs,steps=args.steps,seed=args.seed,device=args.device,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
