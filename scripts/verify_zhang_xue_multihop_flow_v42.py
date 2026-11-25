"""Controlled multi-step Zhang–Xue Warp map→route→packet→relay→E2E ACK.

This reproducer intentionally uses a transparent, isolated three-node line,
0 (at x=25m) <-> 1 (425m) <-> 2 (825m), rather than claiming a whole
real-world workload. HELLO and distance-vector advertisements genuinely
pass the Warp MAC before route tables learn them. DATA payloads are captured
from Warp camera-derived own evidence, kept in Warp queue slots, copied to a
relay preserving source/creation/TTL and merged at actual receiving UAVs.
ACKs return over two physical MAC hops. Blackout is a controlled schedule.

The source-target truth is provided ONLY to the sensor map, not the packet
encoder, router, actor or receiver belief.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np

from uav_network.environment.gpu.zhang_xue_map_warp import ZhangXueWarpMapBatch
from uav_network.communication.gpu.zhang_xue_mac_retry_warp import simulate_one_hop_retry_batch
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import ZhangXueWarpPacketQueue
from uav_network.communication.gpu.zhang_xue_payload_store_warp import ZhangXueWarpPayloadStore
from uav_network.communication.gpu.zhang_xue_origin_evidence_warp import ZhangXueWarpOriginBelief
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import ZhangXueWarpDistanceVector

N=6
TIME_TO_LIVE=10_000_000


def _pack(envs,source,dest,ids,kind,bytes_,created,ttl=TIME_TO_LIVE):
    pid=np.full((envs,N),-1,np.int64)
    destinations=np.full((envs,N),-1,np.int32)
    size=np.zeros((envs,N),np.int32)
    types=np.zeros((envs,N),np.int32)
    creation=np.zeros((envs,N),np.int64)
    expiry=np.ones((envs,N),np.int64)
    pid[:,source]=ids
    destinations[:,source]=dest
    size[:,source]=bytes_
    types[:,source]=kind
    creation[:,source]=created
    expiry[:,source]=ttl
    return pid,destinations,size,types,creation,expiry


def run_multihop_map_network_probe(*,envs=1024,device="cpu",seed=44):
    if not isinstance(envs,int) or envs<4 or envs%4:
        raise ValueError("envs must be divisible by 4, >=4")
    start=time.perf_counter()
    rng=np.random.default_rng(seed)
    xyz=np.full((envs,N,3),2400.,np.float32)
    xyz[:,:,2]=100.
    xyz[:,0,:]=[25.,25.,100.]
    xyz[:,1,:]=[425.,25.,100.]
    xyz[:,2,:]=[825.,25.,100.]
    channels=np.ones((envs,N),np.int32)

    def link_send(dest,sizes,radio=None):
        if radio is None:radio=np.ones((envs,N),np.int32)
        chosen=dest>=0
        sizes=np.where(chosen,sizes,0).astype(np.int32)
        slots=np.zeros((envs,N,5),np.int32)
        for k in range(5):
            slots[:,:,k]=rng.integers(0,32*(2**k)-1,size=(envs,N))
        out=simulate_one_hop_retry_batch(
            xyz,dest.astype(np.int32),sizes,channels,slots,
            np.full((envs,N),10_000_000,np.float32),
            blackout=1-radio,device=device,
        )
        return out

    router=ZhangXueWarpDistanceVector(
        envs,hello_ttl_us=20_000_000,device=device,
    )
    hellos=0
    # Each HELLO event is an actual PHY/MAC frame, not a synthesized neighbor.
    # Different rounds avoid hidden-terminal collisions in this reference
    # single-link baseline; subsequent contention tests must stress collisions.
    for tick,(src,peer) in enumerate(((0,1),(1,0),(1,2),(2,1))):
        receiver=np.full((envs,N),-1,np.int32)
        wire_bytes=np.zeros((envs,N),np.int32)
        receiver[:,src]=peer
        wire_bytes[:,src]=90
        link=link_send(receiver,wire_bytes)
        received=link["data_received"].astype(np.int32)
        assert np.all(received[:,src]==1),f"HELLO failed {src}->{peer}"
        hellos+=int(received.sum())
        router.receive_hello(receiver,received,now_us=tick)
    before=router.read_routes()
    no_route=bool(np.all(before["next_hop"][:,0,2]==-1))
    dv_sent=0
    for tick,peer in ((4,0),(5,2)):
        receiver=np.full((envs,N),-1,np.int32)
        size=np.zeros((envs,N),np.int32)
        receiver[:,1]=peer
        size[:,1]=90
        link=link_send(receiver,size)
        good=link["data_received"].astype(np.int32)
        assert np.all(good[:,1]==1),"DV advertisement failed"
        router.receive_advertisement(receiver,good,now_us=tick)
        dv_sent+=int(good.sum())
    routing=router.read_routes()["next_hop"]
    forward_route=bool(np.all(routing[:,0,2]==1))
    reverse_route=bool(np.all(routing[:,2,0]==1))
    if not forward_route or not reverse_route:
        raise AssertionError("No 2-hop route learned from HELLO/DV")

    # Stable Warp sensor counts, not target truth, are the ONLY
    # source of the eventual application packet payload.
    mapper=ZhangXueWarpMapBatch(envs,device=device)
    target_truth=np.zeros((envs,400),np.int32)
    target_truth[:,0]=1
    mapper.step(xyz[:,:,:2],target_truth,
                uniform_draws=np.full((envs,N,400),.5,np.float32),
                seed=seed,step_index=0)
    receiver_belief=ZhangXueWarpOriginBelief(envs,device=device)
    receiver_belief.set_own_evidence(mapper.pos,mapper.neg)
    own_count=mapper.pos.numpy()
    if not np.all(own_count[:,0,0]==1):
        raise AssertionError("Controlled source camera failed to see cell 0")
    if np.any(receiver_belief.pos.numpy()[:,2,0,0]):
        raise AssertionError("Remote belief truth leak before any packets")

    queue=ZhangXueWarpPacketQueue(envs,capacity=3,device=device)
    payload=ZhangXueWarpPayloadStore(queue)
    orig_ids=np.arange(1,envs+1,dtype=np.int64)
    generated=_pack(envs,0,1,orig_ids,0,78,0)
    assert np.all(queue.enqueue(*generated)[:,0]==1)
    origin=np.full((envs,N,62),-1,np.int32)
    cell=np.full_like(origin,-1)
    positive=np.zeros_like(origin)
    negative=np.zeros_like(origin)
    lengths=np.zeros((envs,N),np.int32)
    origin[:,0,0]=0
    cell[:,0,0]=0
    positive[:,0,0]=own_count[:,0,0]
    lengths[:,0]=1
    assert np.all(payload.attach(generated[0],origin,cell,positive,negative,lengths)[:,0]==1)

    groups=np.arange(envs,dtype=np.int32)%4
    seen_data=set()
    seen_ack=set()
    relay_data=0
    relay_ack=0
    invalid_blackout_sends=0
    metadata_copy_errors=0
    stages=[]
    for step in range(13):
        # Discovery occupied the initial microseconds 0..5; mission starts
        # strictly later so route clocks never run backwards.
        now=10_000+step*1_000_000
        router.tick(now)
        radio=np.ones((envs,N),np.int32)
        radio[:,0]=(
            (groups==0) |
            ((groups==1)&(step>=3)) |
            ((groups==2)&(step>=7))
        ).astype(np.int32)
        out=queue.dispatch(now,radio_enabled=radio)
        active=out["packet_id"]>=0
        invalid_blackout_sends+=int(np.count_nonzero(active&(radio==0)))
        if not active.any():
            stages.append({"step":step,"frames":0,"data_final":0,"ack_final":0})
            continue

        # Existing queued wire snapshot must be inspected BEFORE completing
        # its MAC lease; reused queue slots may change after resolve().
        wire=payload.read_dispatched(out["packet_id"])
        remaining=np.where(active,out["created_us"]+out["ttl_us"]-now,
                           10_000_000).astype(np.float32)
        if np.any(remaining[active]<=0):
            raise AssertionError("Expired packet was dispatched")
        slots=np.zeros((envs,N,5),np.int32)
        for k in range(5):
            slots[:,:,k]=rng.integers(0,32*(2**k)-1,size=(envs,N))
        mac=simulate_one_hop_retry_batch(
            xyz,out["dest"],out["frame_bytes"],channels,slots,remaining,
            blackout=1-radio,device=device,
        )
        data_at_dest=(mac["data_received"]!=0)&active&(out["kind"]==0)
        ack_at_dest=(mac["data_received"]!=0)&active&(out["kind"]==1)
        mac_success=(mac["mac_acked"]!=0)&active
        # Physically received source-origin evidence is merged on Warp,
        # including at relays, and only then can sender queue be resolved.
        if np.any(data_at_dest):
            recv_count=payload.merge_receipts(
                out["packet_id"],out["dest"],data_at_dest.astype(np.int32),
                receiver_belief,
            )
            assert recv_count==int(data_at_dest.sum())
        status=queue.resolve(out["packet_id"],np.where(
            active,np.where(mac_success,1,2),0,
        ).astype(np.int32))
        if not np.all(status[active]==1):
            raise AssertionError("MAC completion was lost")
        if not np.all(mac_success[active]):
            raise AssertionError("Unexpected collision in isolated reference topology")

        # A relay is a distinct UAV sender: preserve packet_id, source
        # evidence origin, immutable creation and original TTL on copy.
        relayed_data_mask=data_at_dest&(out["dest"]==1)
        receiver_final_mask=data_at_dest&(out["dest"]==2)
        ack_relay_mask=ack_at_dest&(out["dest"]==1)
        ack_final_mask=ack_at_dest&(out["dest"]==0)

        if np.any(relayed_data_mask):
            e_ids,u_ids=np.nonzero(relayed_data_mask)
            if np.any(u_ids!=0):
                raise AssertionError("Only original source UAV0 may send first hop")
            identities=np.full((envs,),-1,np.int64)
            identities[e_ids]=out["packet_id"][e_ids,0]
            next_2=routing[e_ids,1,2]
            if not np.all(next_2==2):
                raise AssertionError("Relay1 cannot route to 2")
            forwarded=_pack(envs,1,2,identities,0,78,0)
            assert np.all(queue.enqueue(*forwarded)[e_ids,1]==1)
            p2=np.full((envs,N,62),-1,np.int32)
            c2=np.full((envs,N,62),-1,np.int32)
            plus2=np.zeros((envs,N,62),np.int32)
            minus2=np.zeros((envs,N,62),np.int32)
            num2=np.zeros((envs,N),np.int32)
            p2[e_ids,1]=wire["origin"][e_ids,0]
            c2[e_ids,1]=wire["cell"][e_ids,0]
            plus2[e_ids,1]=wire["positive"][e_ids,0]
            minus2[e_ids,1]=wire["negative"][e_ids,0]
            num2[e_ids,1]=wire["entry_count"][e_ids,0]
            if np.any(p2[e_ids,1,0]!=0):
                metadata_copy_errors+=1
            attached=payload.attach(forwarded[0],p2,c2,plus2,minus2,num2)
            if not np.all(attached[e_ids,1]==1):
                raise AssertionError("Relay payload could not be staged")
            relay_data+=len(e_ids)

        if np.any(receiver_final_mask):
            e_ids,u_ids=np.nonzero(receiver_final_mask)
            if np.any(u_ids!=1):
                raise AssertionError("Two-hop DATA must finish from UAV1")
            for e in e_ids:
                key=(int(e),int(out["packet_id"][e,1]))
                if key in seen_data:
                    raise AssertionError("Duplicate final DATA reward")
                seen_data.add(key)
            ack_ids=np.full((envs,),-1,np.int64)
            ack_ids[e_ids]=1_000_000+out["packet_id"][e_ids,1]
            route_back=routing[e_ids,2,0]
            if not np.all(route_back==1):
                raise AssertionError("No discovered reverse ACK route")
            ack_out=_pack(envs,2,1,ack_ids,1,90,now)
            if not np.all(queue.enqueue(*ack_out)[e_ids,2]==1):
                raise AssertionError("E2E ACK was not queued at final recipient")

        if np.any(ack_relay_mask):
            e_ids,u_ids=np.nonzero(ack_relay_mask)
            if np.any(u_ids!=2):
                raise AssertionError("ACK first hop must originate UAV2")
            ids=np.full(envs,-1,np.int64)
            ids[e_ids]=out["packet_id"][e_ids,2]
            ack_origin_time=np.zeros((envs,),np.int64)
            ack_origin_time[e_ids]=out["created_us"][e_ids,2]
            # Preserve ACK creation/TTL: enqueue one timestamp per world
            # requires batch input rather than _pack scalar helper.
            forwarded=_pack(envs,1,0,ids,1,90,0)
            forwarded=list(forwarded)
            forwarded[4][e_ids,1]=ack_origin_time[e_ids]
            forwarded=tuple(forwarded)
            if not np.all(queue.enqueue(*forwarded)[e_ids,1]==1):
                raise AssertionError("ACK relay queue rejected hop")
            relay_ack+=len(e_ids)

        if np.any(ack_final_mask):
            e_ids,u_ids=np.nonzero(ack_final_mask)
            if np.any(u_ids!=1):
                raise AssertionError("Final ACK must originate relay1")
            for e in e_ids:
                orig=int(out["packet_id"][e,1])-1_000_000
                key=(int(e),orig)
                if key not in seen_data:
                    raise AssertionError("Forged or unpaired application ACK")
                if key in seen_ack:
                    raise AssertionError("Duplicate application ACK reward")
                seen_ack.add(key)
        stages.append({
            "step":step,"frames":int(active.sum()),
            "data_relay":int(relayed_data_mask.sum()),
            "data_final":int(receiver_final_mask.sum()),
            "ack_relay":int(ack_relay_mask.sum()),
            "ack_final":int(ack_final_mask.sum()),
        })
    counts=queue.snapshot()
    local=receiver_belief.pos.numpy()
    expected=(groups!=3)
    recipient_belief=(local[:,2,0,0]>0)
    only_after=bool(np.array_equal(recipient_belief,expected))
    no_interference=bool(np.all(local[~expected,2,0,0]==0))
    return {
        "worlds":envs,
        "scenario":"controlled_three_uav_0-1-2_line",
        "hello_mac_success":hellos,
        "distance_vector_mac_success":dv_sent,
        "no_route_0_to_2_before_advert":no_route,
        "learned_route_0_to_2_via_1":forward_route,
        "learned_route_2_to_0_via_1":reverse_route,
        "app_data_generated":envs,
        "app_data_received":len(seen_data),
        "e2e_ack_received":len(seen_ack),
        "data_ttl_expired":int(counts["expired"].sum()),
        "relay_data_count":relay_data,
        "relay_ack_count":relay_ack,
        "receiver_ds_only_after_physical_data":only_after and no_interference,
        "payload_origin_preserved":metadata_copy_errors==0,
        "duplicate_reward_count":0,
        "blackout_transmissions":invalid_blackout_sends,
        "queues_empty":bool(np.all(counts["queue_length"]==0)),
        "steps":stages,
        "device":device,
        "elapsed_seconds":time.perf_counter()-start,
        "training_started":False,
        "network_full_gate5":"BLOCKED",
        "limitations":[
            "Deterministic three-node line, one app DATA per environment",
            "HELLO and DV are one-hop MAC frames; no native SimPy PHY parity beyond earlier single topology",
            "Each relay is orchestrated from actual route table with Python glue; no fused persistent Warp routing scheduler",
            "No simultaneous multi-user contention, actor, reward, learned topology or Kaggle CUDA yet",
        ],
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output-json",type=Path)
    args=parser.parse_args()
    result=run_multihop_map_network_probe(envs=args.envs,device=args.device,seed=args.seed)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
