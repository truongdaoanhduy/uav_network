"""Controlled six-UAV 5-hop DATA and reverse 5-hop application E2E ACK.

This is an executable *pre-Gate5* CPU/GPU Warp packet-flow diagnostic.
A local camera creates source-owned evidence, genuine PHY/MAC receptions
establish bidirectional HELLO and delivered path-vector route knowledge,
and a persistent Warp queue carries immutable evidence+forwarding envelope.

DATA: UAV0→UAV1→UAV2→UAV3→UAV4→UAV5.
APP ACK: UAV5→UAV4→UAV3→UAV2→UAV1→UAV0.
One quarter of independent worlds permanently black out relay UAV2 after
it has received DATA, forcing relay TTL expiry without a false E2E ACK.

This is not a learned actor or native UavNetSim-synchronous full network:
control frames are isolated on MAC, data is one packet/world, and relay
admissions are orchestrated in Python between actual Warp kernels.
"""
from __future__ import annotations
import argparse
import json
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
from uav_network.communication.gpu.zhang_xue_origin_evidence_warp import (
    ZhangXueWarpOriginBelief,
)
from uav_network.communication.gpu.zhang_xue_payload_store_warp import (
    ZhangXueWarpPayloadStore,
)
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import (
    ZhangXueWarpDistanceVector,
)
from uav_network.communication.gpu.zhang_xue_e2e_watermarks_warp import (
    ZhangXueWarpAckWatermarks,
)
from uav_network.communication.gpu.zhang_xue_forward_envelope_warp import (
    ZhangXueWarpForwardingEnvelope,
)


N=6
K=62
TTL_US=10_000_000
ACK_BASE=1_000_000


def _geometry(worlds):
    # A folded chain within 1km; each adjacent hop is 450m, each
    # nonadjacent hop is ≥636m and is not an assumed route.
    points=np.array([
        (25.,25.,100.),(475.,25.,100.),(925.,25.,100.),
        (925.,475.,100.),(925.,925.,100.),(475.,925.,100.),
    ],dtype=np.float32)
    return np.broadcast_to(points,(worlds,N,3)).copy()


def _frame(worlds,src,wire_dst,packet_ids,byte_length,kind,
           creation_us,ttl_us):
    ids=np.full((worlds,N),-1,np.int64)
    dst=np.full((worlds,N),-1,np.int32)
    lengths=np.zeros((worlds,N),np.int32)
    types=np.zeros((worlds,N),np.int32)
    created=np.zeros((worlds,N),np.int64)
    ttl=np.ones((worlds,N),np.int64)
    active=packet_ids>=0
    e=np.nonzero(active)[0]
    ids[e,src]=packet_ids[e]
    dst[e,src]=wire_dst[e]
    lengths[e,src]=byte_length[e]
    types[e,src]=kind[e]
    created[e,src]=creation_us[e]
    ttl[e,src]=ttl_us[e]
    return ids,dst,lengths,types,created,ttl


def _source_metadata(worlds,holder,ids,app_origin,ultimate,hops,visited):
    result=[np.full((worlds,N),-1,np.int64)]
    result.extend(np.full((worlds,N),-1,np.int32) for _ in range(2))
    result.extend(np.zeros((worlds,N),np.int32) for _ in range(2))
    e=np.nonzero(ids>=0)[0]
    result[0][e,holder]=ids[e]
    result[1][e,holder]=app_origin[e]
    result[2][e,holder]=ultimate[e]
    result[3][e,holder]=hops[e]
    result[4][e,holder]=visited[e]
    return result


def run_six_uav_multihop_probe(*,worlds=1024,seed=44,device="cpu"):
    if not isinstance(worlds,int) or worlds<4 or worlds%4:
        raise ValueError("worlds must be a positive multiple of 4")
    start=time.perf_counter()
    xyz=_geometry(worlds)
    rng=np.random.default_rng(seed)
    channels=np.ones((worlds,N),np.int32)

    def mac_send(dest,frame_bytes,radio=None,deadline=None):
        if radio is None:
            radio=np.ones((worlds,N),np.int32)
        if deadline is None:
            deadline=np.full((worlds,N),TTL_US,np.float32)
        backoff=np.zeros((worlds,N,5),np.int32)
        for attempt in range(5):
            backoff[:,:,attempt]=rng.integers(
                0,32*(2**attempt)-1,size=(worlds,N),dtype=np.int32,
            )
        return simulate_one_hop_retry_batch(
            xyz,np.asarray(dest,dtype=np.int32),
            np.asarray(frame_bytes,dtype=np.int32),
            channels,backoff,deadline,blackout=1-radio,device=device,
        )

    router=ZhangXueWarpDistanceVector(
        worlds,device=device,hello_ttl_us=30_000_000,
    )
    no_route_before=bool(np.all(
        router.read_routes()["next_hop"][:,0,5]==-1
    ))
    hellos=0
    for order,src,dst in (
        (order,source,destination)
        for order,(source,destination) in enumerate(
            [(a,b) for a in range(N-1)
                   for b in (a+1,)]
            + [(a+1,a) for a in range(N-1)]
        )
    ):
        receivers=np.full((worlds,N),-1,np.int32)
        lengths=np.zeros((worlds,N),np.int32)
        receivers[:,src]=dst
        lengths[:,src]=90
        out=mac_send(receivers,lengths)
        success=out["data_received"].astype(np.int32)
        if not np.all(success[:,src]==1):
            raise AssertionError("HELLO was not physically received")
        router.receive_hello(receivers,success,now_us=order+1)
        hellos+=int(success[:,src].sum())
    before_dv=bool(np.all(
        router.read_routes()["next_hop"][:,0,5]==-1
    ))
    dv_count=0
    for order,(src,dst) in enumerate((
        (4,3),(3,2),(2,1),(1,0),
        (1,2),(2,3),(3,4),(4,5),
    )):
        receivers=np.full((worlds,N),-1,np.int32)
        lengths=np.zeros((worlds,N),np.int32)
        receivers[:,src]=dst
        lengths[:,src]=90
        out=mac_send(receivers,lengths)
        success=out["data_received"].astype(np.int32)
        if not np.all(success[:,src]==1):
            raise AssertionError("DV was not physically received")
        router.receive_advertisement(
            receivers,success,now_us=20+order,
        )
        dv_count+=int(success[:,src].sum())
    routing=router.read_routes()
    next_hops=routing["next_hop"]
    if not (np.all(next_hops[:,0,5]==1) and
            np.all(next_hops[:,5,0]==4)):
        raise AssertionError("App route cannot use non-oracle 5-hop path")
    source_route_learned=no_route_before and before_dv

    # One target is visible at UAV0's ground-camera cell0. The scene
    # target truth never enters packet encoding, route or E2E ACK fields.
    truth=np.zeros((worlds,400),np.int32)
    truth[:,0]=1
    mapper=ZhangXueWarpMapBatch(worlds,device=device)
    mapper.step(
        xyz[:,:,:2],truth,
        uniform_draws=np.full((worlds,N,400),.5,np.float32),
        seed=seed,step_index=0,
    )
    belief=ZhangXueWarpOriginBelief(worlds,device=device)
    belief.set_own_evidence(mapper.pos,mapper.neg)
    own_p=mapper.pos.numpy()
    own_n=mapper.neg.numpy()
    if not np.all(own_p[:,0,0]>0):
        raise AssertionError("Source local 2D map did not detect cell0")
    if np.any(belief.pos.numpy()[:,5,0,0]):
        raise AssertionError("Final receiver saw DATA before any packet")

    wm=ZhangXueWarpAckWatermarks(worlds,device=device)
    q=ZhangXueWarpPacketQueue(worlds,capacity=2,device=device)
    payload=ZhangXueWarpPayloadStore(q)
    envelope=ZhangXueWarpForwardingEnvelope(q)
    peers=np.full((worlds,N),-1,np.int32)
    peers[:,0]=5
    dirty=wm.dirty(own_p,own_n,peers)
    count=dirty["count"][:,0]
    if not np.all(count>=1):
        raise AssertionError("No camera-derived evidence selected")
    rawids=np.arange(1,worlds+1,dtype=np.int64)
    wire_dst=next_hops[:,0,5].copy()
    first=_frame(
        worlds,0,wire_dst,rawids,(70+8*count).astype(np.int32),
        np.zeros(worlds,np.int32),np.full(worlds,1000,np.int64),
        np.full(worlds,TTL_US,np.int64),
    )
    if not np.all(q.enqueue(*first)[:,0]==1):
        raise AssertionError("DATA queue rejected valid source")
    if not np.all(wm.register(first[0],peers,dirty)[:,0]==1):
        raise AssertionError("Source watermark could not reserve APP packet")
    initial_meta=_source_metadata(
        worlds,0,rawids,np.zeros(worlds,np.int32),
        np.full(worlds,5,np.int32),np.zeros(worlds,np.int32),
        np.ones(worlds,np.int32),
    )
    if not np.all(envelope.attach(*initial_meta)[:,0]==1):
        raise AssertionError("Missing source envelope")
    origin=np.full((worlds,N,K),-1,np.int32)
    cell=np.full_like(origin,-1)
    pos=np.zeros((worlds,N,K),np.int32)
    neg=np.zeros((worlds,N,K),np.int32)
    lengths=np.zeros((worlds,N),np.int32)
    origin[:,0,:]=np.where(
        np.arange(K)[None,:]<count[:,None],0,-1,
    )
    cell[:,0,:]=dirty["cell"][:,0,:]
    pos[:,0,:]=dirty["positive"][:,0,:]
    neg[:,0,:]=dirty["negative"][:,0,:]
    lengths[:,0]=count
    if not np.all(payload.attach(first[0],origin,cell,pos,neg,lengths)[:,0]==1):
        raise AssertionError("Source evidence was not stored on Warp")

    app_data_received=set()
    app_ack_received=set()
    data_hop_receipts=0
    ack_hop_receipts=0
    data_relay_admissions=0
    ack_relay_admissions=0
    loop_drops=0
    blackout_transmissions=0
    original_ttl_intact=True
    provenance_intact=True
    stages=[]
    source_born=1000
    ack_born=source_born+4_000_000
    for step in range(12):
        now=source_born+step*1_000_000
        router.tick(now)
        radio=np.ones((worlds,N),np.int32)
        if step>=2:
            radio[3::4,2]=0
        sent=q.dispatch(now,radio_enabled=radio)
        active=sent["packet_id"]>=0
        blackout_transmissions+=int(np.count_nonzero(active&(radio==0)))
        if not np.any(active):
            stages.append({"step":step,"active":0,"received":0})
            continue
        metadata=envelope.read_dispatched(sent["packet_id"])
        snapshots=payload.read_dispatched(sent["packet_id"])
        remaining=np.where(
            active,sent["created_us"]+sent["ttl_us"]-now,TTL_US,
        ).astype(np.float32)
        if np.any(remaining[active]<=0):
            raise AssertionError("Expired packet dispatched")
        mac=mac_send(sent["dest"],sent["frame_bytes"],radio,remaining)
        received=(mac["data_received"]==1)&active
        mac_ok=(mac["mac_acked"]==1)&active
        if not np.all(mac_ok[active]):
            raise AssertionError("Unexpected MAC loss on isolated 450m link")
        classified=envelope.classify_receipt(
            sent["packet_id"],sent["dest"],received.astype(np.int32),
        )
        loop_drops+=int(classified["loop_rejected"].sum()+
                        classified["hop_budget_exceeded"].sum())
        if np.any(received&(sent["kind"]==0)):
            payload.merge_receipts(
                sent["packet_id"],sent["dest"],
                (received&(sent["kind"]==0)).astype(np.int32),
                belief,
            )
        data_hop_receipts+=int((received&(sent["kind"]==0)).sum())
        ack_hop_receipts+=int((received&(sent["kind"]==1)).sum())
        active_data=active&(sent["kind"]==0)
        original_ttl_intact&=bool(
            np.all(sent["created_us"][active_data]==source_born)
            and np.all(sent["ttl_us"][active_data]==TTL_US)
        )
        if np.any(active_data):
            provenance_intact&=bool(
                np.all(metadata["original_source"][active_data]==0)
                and np.all(metadata["ultimate_destination"][active_data]==5)
                and np.all(snapshots["entry_count"][active_data]>=1)
                and np.all(snapshots["origin"][active_data,0]==0)
            )

        completed=q.resolve(
            sent["packet_id"],
            np.where(active,np.where(mac_ok,1,2),0).astype(np.int32),
        )
        if not np.all(completed[active]==1):
            raise AssertionError("Sender queue MAC completion mismatch")

        # Apply a separate admission for each original sending UAV.
        # All fields are copied from its *immutable* routed packet.
        for sender in range(N):
            env_ids=np.nonzero(classified["relay"][:,sender]==1)[0]
            if not len(env_ids):
                continue
            receivers=sent["dest"][env_ids,sender]
            ultimates=classified["ultimate_destination"][env_ids,sender]
            nhop=next_hops[env_ids,receivers,ultimates]
            if np.any(nhop<0):
                raise AssertionError("Receiver lacks local discovered next hop")
            if np.any((classified["visited_mask"][env_ids,sender] &
                       (1<<nhop))!=0):
                raise AssertionError("Route next hop revisits packet path")
            for holder in np.unique(receivers):
                sel=env_ids[receivers==holder]
                nids=np.full(worlds,-1,np.int64)
                nids[sel]=sent["packet_id"][sel,sender]
                ndst=np.full(worlds,-1,np.int32)
                ndst[sel]=next_hops[
                    sel,holder,classified["ultimate_destination"][sel,sender],
                ]
                length=np.zeros(worlds,np.int32)
                length[sel]=sent["frame_bytes"][sel,sender]
                kind=np.zeros(worlds,np.int32)
                kind[sel]=sent["kind"][sel,sender]
                created=np.zeros(worlds,np.int64)
                created[sel]=sent["created_us"][sel,sender]
                ttl=np.full(worlds,1,np.int64)
                ttl[sel]=sent["ttl_us"][sel,sender]
                forwarded=_frame(
                    worlds,int(holder),ndst,nids,length,kind,created,ttl,
                )
                acceptance=q.enqueue(*forwarded)
                if not np.all(acceptance[sel,holder]==1):
                    raise AssertionError("Relay queue cannot accept APP packet")
                copied=_source_metadata(
                    worlds,int(holder),nids,
                    np.where(nids>=0,
                        classified["original_source"][:,sender],-1).astype(np.int32),
                    np.where(nids>=0,
                        classified["ultimate_destination"][:,sender],-1).astype(np.int32),
                    np.where(nids>=0,
                        classified["hop_count"][:,sender],0).astype(np.int32),
                    np.where(nids>=0,
                        classified["visited_mask"][:,sender],0).astype(np.int32),
                )
                if not np.all(envelope.attach(*copied)[sel,holder]==1):
                    raise AssertionError("Relay lost original source/visited mask")
                if np.all(kind[sel]==0):
                    o=np.full((worlds,N,K),-1,np.int32)
                    c=np.full((worlds,N,K),-1,np.int32)
                    p=np.zeros((worlds,N,K),np.int32)
                    n=np.zeros((worlds,N,K),np.int32)
                    cnt=np.zeros((worlds,N),np.int32)
                    o[sel,holder,:]=snapshots["origin"][sel,sender,:]
                    c[sel,holder,:]=snapshots["cell"][sel,sender,:]
                    p[sel,holder,:]=snapshots["positive"][sel,sender,:]
                    n[sel,holder,:]=snapshots["negative"][sel,sender,:]
                    cnt[sel,holder]=snapshots["entry_count"][sel,sender]
                    if not np.all(payload.attach(forwarded[0],o,c,p,n,cnt)[sel,holder]==1):
                        raise AssertionError("Relay evidence snapshot missing")
                    data_relay_admissions+=len(sel)
                else:
                    ack_relay_admissions+=len(sel)

        # Application DATA only counts as delivered at ultimate UAV5.
        for sender in range(N):
            finalized=np.nonzero(
                (classified["final"][:,sender]==1)&
                (sent["kind"][:,sender]==0)
            )[0]
            if not len(finalized):
                continue
            assert sender==4
            for e in finalized:
                key=(int(e),int(sent["packet_id"][e,sender]))
                if key in app_data_received:
                    raise AssertionError("Double DATA at ultimate")
                app_data_received.add(key)
            reply_id=np.full(worlds,-1,np.int64)
            reply_id[finalized]=ACK_BASE+sent["packet_id"][finalized,sender]
            nhop=np.full(worlds,-1,np.int32)
            nhop[finalized]=next_hops[finalized,5,0]
            ack_frame=_frame(
                worlds,5,nhop,reply_id,
                np.full(worlds,90,np.int32),
                np.ones(worlds,np.int32),
                np.full(worlds,now,np.int64),
                np.full(worlds,TTL_US,np.int64),
            )
            if not np.all(q.enqueue(*ack_frame)[finalized,5]==1):
                raise AssertionError("Ultimate DATA receiver cannot queue E2E ACK")
            ack_meta=_source_metadata(
                worlds,5,reply_id,np.full(worlds,5,np.int32),
                np.zeros(worlds,np.int32),np.zeros(worlds,np.int32),
                np.full(worlds,1<<5,np.int32),
            )
            if not np.all(envelope.attach(*ack_meta)[finalized,5]==1):
                raise AssertionError("ACK missing reverse app envelope")

        # Returned APP ACK, not link MAC ACK, advances original-source
        # per-peer E2E evidence watermark exactly once.
        for sender in range(N):
            finalized=np.nonzero(
                (classified["final"][:,sender]==1)&
                (sent["kind"][:,sender]==1)
            )[0]
            if not len(finalized):
                continue
            assert sender==1
            original_ids=np.full((worlds,N),-1,np.int64)
            peers_back=np.full((worlds,N),-1,np.int32)
            real_ack=np.zeros((worlds,N),np.int32)
            original_ids[finalized,0]=sent["packet_id"][finalized,sender]-ACK_BASE
            peers_back[finalized,0]=5
            real_ack[finalized,0]=1
            accepted=wm.acknowledge(original_ids,peers_back,real_ack)
            if not np.all(accepted[finalized,0]==1):
                raise AssertionError("Returned APP ACK did not match source DATA")
            for e in finalized:
                key=(int(e),int(original_ids[e,0]))
                if key not in app_data_received or key in app_ack_received:
                    raise AssertionError("Forged/duplicate E2E ACK reward")
                app_ack_received.add(key)
        stages.append({
            "step":step,"active":int(active.sum()),
            "data_receipts":int((received&(sent["kind"]==0)).sum()),
            "ack_receipts":int((received&(sent["kind"]==1)).sum()),
        })
    state=q.snapshot()
    end=belief.read()
    ds_received=end["pos"][:,5,0,0]>0
    expected=(np.arange(worlds)%4)!=3
    if not np.array_equal(ds_received,expected):
        raise AssertionError("Receiver-local DS leaked before physical DATA")
    watermark=wm.read()
    wm_acked=(watermark["acked_positive"][:,0,5,0]>0)
    if not np.array_equal(wm_acked,expected):
        raise AssertionError("Source watermark advanced without return APP ACK")
    final_ids={e for e,_ in app_data_received}
    acked_ids={e for e,_ in app_ack_received}
    return {
        "stage":"controlled_six_uav_five_hop_warp_packet_envelope",
        "worlds":worlds,
        "uavs":N,
        "hello_frames_physically_received":hellos,
        "route_advertisements_received":dv_count,
        "source_route_only_after_dv":source_route_learned,
        "app_data_generated":worlds,
        "app_data_received_at_ultimate":len(final_ids),
        "data_hop_receipts":data_hop_receipts,
        "data_forwarded_hops":data_relay_admissions,
        "ack_hop_receipts":ack_hop_receipts,
        "ack_forwarded_hops":ack_relay_admissions,
        "application_ack_returned":len(acked_ids),
        "e2e_watermark_updated":int(watermark["successful_e2e_acks"].sum()),
        "radio_blackout_holds_no_send":blackout_transmissions==0,
        "data_ttl_expired_in_relay":int(state["expired"][:,2].sum()),
        "receiver_DS_only_after_data":True,
        "source_provenance_and_original_ttl_preserved":bool(
            provenance_intact and original_ttl_intact
        ),
        "no_packet_forwarding_loops":loop_drops==0,
        "no_duplicate_app_reward":len(app_ack_received)==len(acked_ids),
        "queues_empty_at_end":bool(not state["queue_length"].any()),
        "stage_events":stages,
        "elapsed_seconds":time.perf_counter()-start,
        "device":device,
        "training_launched":False,
        "gate5":"BLOCKED",
        "limitations":[
            "Static six-UAV folded chain, isolated one-APP-message/world",
            "HELLO/DV transmissions are MAC successes but not fully serialized upstream UavNetSim protocol packets",
            "Host Python relays packet metadata between Warp kernels per step",
            "No learned actor/motion, dynamic target distribution, production reward or actual GPU CUDA verification",
        ],
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output-json",type=Path)
    a=parser.parse_args()
    report=run_six_uav_multihop_probe(worlds=a.worlds,seed=a.seed,device=a.device)
    if a.output_json:
        a.output_json.parent.mkdir(parents=True,exist_ok=True)
        a.output_json.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
