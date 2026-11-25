"""Sampled Zhang-Xue 1km 6-UAV moving Warp map and network pilot.

Uses original 10-target/2-no-fly/2-blackout scene sampler, Warp safe
motion, Warp camera/DS, PHY/SINR/CCA MAC, per-receiver evidence, persistent
queue and application ACK watermarks. All local peer choices are derived
from successfully received HELLO packets, not global geometry.
Reward-only truth measures confirmation; no trainable actor or learner.

Explicit research PRE-GATE5 diagnostic, not real CUDA GPU training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from uav_network.environment.zhang_xue_coop import ZhangXueConfig
from uav_network.environment.zhang_xue_map_geometry import sample_scenario_batch
from uav_network.environment.gpu.zhang_xue_motion_warp import warp_move_batch
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


def run_sampled_scenario_network_probe(*,envs=128,steps=20,seed=44,device="cpu"):
    if not isinstance(envs,int) or envs<1:
        raise ValueError("positive E")
    if not isinstance(steps,int) or steps<2:
        raise ValueError("at least two network steps")
    started=time.perf_counter()
    config=ZhangXueConfig()
    scenes=sample_scenario_batch(
        config,np.arange(seed,seed+envs,dtype=np.int64),
    )
    xyz=np.empty((envs,N,3),np.float32)
    xyz[:,:,:2]=scenes["positions"]
    xyz[:,:,2]=config.fixed_altitude_meter
    truth=scenes["target_mask"].astype(np.int32)
    channels=np.ones((envs,N),np.int32)
    rng=np.random.default_rng(seed+777)
    safety_rejections=0
    boundary_rejections=0
    all_positions_in_bounds=True
    never_entered_nofly=True
    safe_separation=True
    blackout_exposures=0

    def _radio():
        coord=np.floor(xyz[:,:,:2]/config.cell_size_meter).astype(np.int32)
        blocked=np.zeros((envs,N),dtype=bool)
        for z in range(config.blackout_zone_count):
            distance=np.abs(
                coord-scenes["blackout_zones"][:,z,None,:]
            ).sum(axis=-1)
            blocked|=distance<=config.blackout_radius_cells
        return (~blocked).astype(np.int32)

    def _move():
        nonlocal safety_rejections,boundary_rejections
        nonlocal all_positions_in_bounds,never_entered_nofly,safe_separation
        actions=rng.uniform(-1,1,size=(envs,N,2)).astype(np.float32)
        moved,velocity,ns,nb=warp_move_batch(
            xyz[:,:,:2],scenes["no_fly_zones"],actions,
            dt=config.dt_seconds,max_speed=config.v_max_mps,
            safe_distance=config.safe_distance_meter,device=device,
        )
        safety_rejections+=int(ns.sum())
        boundary_rejections+=int(nb.sum())
        all_positions_in_bounds&=bool(
            np.all((moved>=0.)&(moved<config.map_size_meter))
        )
        for z in range(config.no_fly_zone_count):
            delta=moved-scenes["no_fly_zones"][:,z,None,:2]
            distance=np.linalg.norm(delta,axis=-1)
            never_entered_nofly&=bool(
                np.all(distance+1e-4>=scenes["no_fly_zones"][:,z,None,2])
            )
        for a in range(N):
            for b in range(a+1,N):
                d=np.linalg.norm(moved[:,a]-moved[:,b],axis=-1)
                safe_separation&=bool(np.all(d+1e-4>=config.safe_distance_meter))
        xyz[:,:,:2]=moved


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
        envs,device=device,hello_ttl_us=12_000_000,
    )
    before=router.read_routes()["next_hop"]
    local_only=bool(np.all(
        before[:,np.arange(N),(np.arange(N)+1)%N]==-1
    ))
    hello_success=0
    hello_attempts=0
    direct_routes_hello_only=True

    def _discover(now_start):
        nonlocal hello_success,hello_attempts
        radio=_radio()
        trial_index=0
        for src in range(N):
            for dest in range(N):
                if dest==src:
                    continue
                rx=np.full((envs,N),-1,np.int32)
                wire=np.zeros((envs,N),np.int32)
                rx[:,src]=dest
                wire[:,src]=90
                mac=_send(rx,wire,radio=radio)
                good=mac["data_received"].astype(np.int32)
                router.receive_hello(
                    rx,good,now_us=now_start+trial_index+1,
                )
                hello_success+=int(good[:,src].sum())
                hello_attempts+=envs
                trial_index+=1

    _discover(0)
    if not local_only:
        raise AssertionError("Any route was known before physical HELLO")
    initial=router.read_routes()
    for src in range(N):
        for dest in range(N):
            if src!=dest:
                route=initial["next_hop"][:,src,dest]
                link=initial["bidirectional_links"][:,src,dest]
                direct_routes_hello_only&=bool(
                    np.all((route!=dest)|(link==1))
                )
    peers=np.full((envs,N),-1,np.int32)
    mapper=ZhangXueWarpMapBatch(envs,device=device)
    belief=ZhangXueWarpOriginBelief(envs,device=device)
    watermark=ZhangXueWarpAckWatermarks(envs,device=device)
    queue=ZhangXueWarpPacketQueue(envs,capacity=4,device=device)
    payload=ZhangXueWarpPayloadStore(queue)
    env_index=np.arange(envs,dtype=np.int64)[:,None]
    src_index=np.arange(N,dtype=np.int64)[None,:]
    sender_index=np.arange(N,dtype=np.int32)[None,:,None]
    pending_created=np.full((envs,N,N),-1,np.int64)
    occurred_data=np.zeros((envs,N,N),np.int32)

    data_generated=0
    data_received=0
    ack_generated=0
    ack_attempted=0
    ack_enqueue_rejected=0
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
    tp_latch=np.zeros((envs,C),bool)
    fp_latch=np.zeros((envs,C),bool)
    reward_per_env=np.zeros(envs,np.float64)
    reward_only_truth=True

    def _truth_reward():
        # Truth is read only HERE and in the sensor kernel; never in the
        # route choice, actor input, DATA payload or ACK watermark.
        posterior=belief.mass_target.numpy().max(axis=1)
        confirmed=posterior>=config.confirm_threshold
        tp=confirmed&(truth!=0)&(~tp_latch)
        fp=confirmed&(truth==0)&(~fp_latch)
        reward_per_env[:]+=10.*tp.sum(axis=1)-5.*fp.sum(axis=1)
        tp_latch[:]|=tp
        fp_latch[:]|=fp


    for step in range(steps):
        now=20_000+step*1_000_000
        router.tick(now)
        if step and step%4==0:
            # Refresh all peer routes using only actual HELLO receptions
            # at current UAV poses. No global-position-based route choice.
            _discover(now)
            router.tick(now+100)
        learned=router.read_routes()
        routes=learned["next_hop"]
        links=learned["bidirectional_links"]
        peers[:]=-1
        for src in range(N):
            for rank in range(1,N):
                candidate=(src+rank+step)%N
                if candidate==src:
                    continue
                discovered=(routes[:,src,candidate]==candidate)&(
                    links[:,src,candidate]==1
                )
                choose=(peers[:,src]<0)&discovered
                peers[choose,src]=candidate
        for src in range(N):
            for dst in range(N):
                if src!=dst:
                    current_route=routes[:,src,dst]
                    current_link=links[:,src,dst]
                    direct_routes_hello_only&=bool(
                        np.all((current_route!=dst)|(current_link==1))
                    )
        mapper.step(xyz[:,:,:2],truth,seed=seed,step_index=step)
        belief.set_own_evidence(mapper.pos,mapper.neg)
        own_p=mapper.pos.numpy()
        own_n=mapper.neg.numpy()
        if step>=10:
            # The outstanding app ACK record is per source/peer and must
            # survive changes to the chosen next destination.
            pending=watermark.pending_id.numpy()
            expired=(pending_created>=0)&((now-pending_created)>=DATA_TTL_US)
            for peer_id in range(N):
                targets=np.full((envs,N),peer_id,np.int32)
                to_abandon=np.where(expired[:,:,peer_id],
                                    pending[:,:,peer_id],-1).astype(np.int64)
                to_abandon[:,peer_id]=-1
                released=watermark.abandon(to_abandon,targets)
                timeouts+=int((released==1).sum())
                pending_created[:,:,peer_id][released==1]=-1

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
            e_ready,u_ready=np.nonzero(ready)
            pending_created[e_ready,u_ready,peers[e_ready,u_ready]]=now
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

        radio=_radio()
        blackout_exposures+=int(np.count_nonzero(radio==0))
        flight=queue.dispatch(now,radio_enabled=radio)
        active=flight["packet_id"]>=0
        no_transmits_when_black+=int(np.count_nonzero(active&(radio==0)))
        state=queue.snapshot()
        inflight=state["inflight"]
        beyond_one+=int(np.count_nonzero(inflight>1))
        max_queue_seen=max(max_queue_seen,int(state["queue_length"].max()))
        if not np.any(active):
            _truth_reward()
            _move()
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

        # Every received DATA attempts a separate application ACK.
        # Multiple original sources may converge on the same receiver.
        # Enqueue one ACK batch per sender, never silently overwrite.
        status=queue.resolve(flight["packet_id"],np.where(
            active,np.where(mac_good,1,2),0,
        ).astype(np.int32))
        if not np.all(status[active]==1):
            raise AssertionError("MAC completion missing")
        for source in range(N):
            e_ids=np.nonzero(received_data[:,source])[0]
            if not len(e_ids):
                continue
            ack_attempted+=len(e_ids)
            destinations=flight["dest"][e_ids,source]
            ack_ids=np.full((envs,N),-1,np.int64)
            ack_dest=np.full((envs,N),-1,np.int32)
            ack_size=np.zeros((envs,N),np.int32)
            ack_kind=np.zeros((envs,N),np.int32)
            ack_created=np.zeros((envs,N),np.int64)
            ack_ttl=np.ones((envs,N),np.int64)
            ack_ids[e_ids,destinations]=(
                ACK_BASE+flight["packet_id"][e_ids,source]
            )
            ack_dest[e_ids,destinations]=source
            ack_size[e_ids,destinations]=90
            ack_kind[e_ids,destinations]=1
            ack_created[e_ids,destinations]=now
            ack_ttl[e_ids,destinations]=DATA_TTL_US
            admitted=queue.enqueue(
                ack_ids,ack_dest,ack_size,ack_kind,ack_created,ack_ttl,
            )
            statuses=admitted[e_ids,destinations]
            if np.any((statuses!=1)&(statuses!=-2)):
                raise AssertionError("Unexpected ACK enqueue outcome")
            ack_generated+=int((statuses==1).sum())
            ack_enqueue_rejected+=int((statuses==-2).sum())
            queue_drop+=int((statuses==-2).sum())
        if ack_attempted!=ack_generated+ack_enqueue_rejected:
            raise AssertionError("Some receiver ACK attempts were lost")

        # Each ACK sender can complete one packet per step, but distinct
        # ACK senders may reply to the SAME original source concurrently.
        # Process each independently: never overwrite an ACK for that source.
        ack_phys_received+=int(received_ack.sum())
        for ack_sender in range(N):
            e_ack=np.nonzero(received_ack[:,ack_sender])[0]
            if not len(e_ack):
                continue
            original_sources=flight["dest"][e_ack,ack_sender]
            original_ids=flight["packet_id"][e_ack,ack_sender]-ACK_BASE
            ack_ids_original=np.full((envs,N),-1,np.int64)
            acknowledged_peer=np.full((envs,N),-1,np.int32)
            ack_delivered=np.zeros((envs,N),np.int32)
            ack_ids_original[e_ack,original_sources]=original_ids
            acknowledged_peer[e_ack,original_sources]=ack_sender
            ack_delivered[e_ack,original_sources]=1
            responses=watermark.acknowledge(
                ack_ids_original,acknowledged_peer,ack_delivered,
            )
            ack_correct+=int((responses==1).sum())
            stale_ack_count+=int((responses==-1).sum())
            good_e,good_src=np.nonzero(responses==1)
            pending_created[good_e,good_src,ack_sender]=-1
            if np.any((responses==1)&(ack_delivered==0)):
                duplicate_reward+=1
            for e,u in zip(good_e,good_src):
                key=(int(e),int(ack_ids_original[e,u]))
                if key in matched_data_by_id:
                    duplicate_reward+=1
                matched_data_by_id.add(key)
        _truth_reward()
        _move()
        stage.append({
            "step":step,
            "active":int(active.sum()),
            "real_data_received":int(received_data.sum()),
            "real_app_ack_received":int(received_ack.sum()),
            "ack_enqueue_attempts_total":int(ack_attempted),
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
    final_map=mapper.read()
    visited=np.any(final_map["pos"]+final_map["neg"]>0,axis=1)
    source_count=int(
        final_map["pos"].sum(dtype=np.int64)+
        final_map["neg"].sum(dtype=np.int64)
    )
    if ack_attempted!=ack_generated+ack_enqueue_rejected:
        raise AssertionError("ACK attempt bookkeeping inconsistent")
    return {
        "stage":"sampled_moving_1km_warp_network_reward_cpu_pilot",
        "worlds":envs,"steps":steps,
        "scene_map_size_m":int(config.map_size_meter),
        "uavs":config.uav_count,
        "targets_per_world":config.target_count,
        "no_fly_disks_per_world":int(scenes["no_fly_zones"].shape[1]),
        "blackout_diamonds_per_world":int(scenes["blackout_zones"].shape[1]),
        "target_layout_valid":bool(
            np.all(truth.sum(axis=1)==config.target_count)
        ),
        "flight_positions_inside_world":all_positions_in_bounds,
        "flight_did_not_penetrate_no_fly":never_entered_nofly,
        "safe_inter_uav_separation":safe_separation,
        "blocked_unsafe_motion":safety_rejections,
        "blocked_boundary_motion":boundary_rejections,
        "observed_blackout_uav_steps":blackout_exposures,
        "real_mac_hello_attempts":hello_attempts,
        "real_mac_hello_received":hello_success,
        "all_direct_routes_from_received_hello_only":bool(
            direct_routes_hello_only and local_only
        ),
        "source_evidence_observations":source_count,
        "app_data_generated":data_generated,
        "app_data_physical_receipts":data_received,
        "app_ack_enqueues_attempted":ack_attempted,
        "app_ack_enqueued":ack_generated,
        "app_ack_queue_rejected":ack_enqueue_rejected,
        "app_ack_physical_receipts":ack_phys_received,
        "app_e2e_ack_valid":ack_correct,
        "stale_app_e2e_ack_count":stale_ack_count,
        "e2e_watermarks_advanced":total_valid,
        "reward_target_truth_only":reward_only_truth,
        "cumulative_truth_reward_mean":float(reward_per_env.mean()),
        "true_target_confirmations":int(tp_latch.sum()),
        "false_confirmations":int(fp_latch.sum()),
        "map_coverage_mean":float(visited.mean(axis=1).mean()),
        "uavs_communicating_during_blackout":no_transmits_when_black,
        "receiver_evidence_before_physical_receipt":leaks,
        "duplicate_reward_count":duplicate_reward,
        "MAC_cca_deferrals":cca_deferrals,
        "MAC_collision_attempts":collision_attempts,
        "app_ack_timeout_releases":timeouts,
        "queue_full_drops":queue_drop,
        "queue_max_occupancy":max_queue_seen,
        "inflight_over_one":beyond_one,
        "device":device,
        "elapsed_seconds":float(time.perf_counter()-started),
        "step_events":stage,
        "training_launched":False,
        "full_network_gate5":"BLOCKED",
        "limitations":[
            "Random nonlearning movement and deterministic local peer preference",
            "Direct bidirectional HELLO routes, no dynamic multi-hop forwarding",
            "Truth only sensor and reward, never actor/network/ACK",
            "Pilot reward only first target confirmations, not full production reward",
            "No full energy/battery constraints or trained actor",
            "Per-step network control still Python, Warp MAC not source-exact",
            "Real CUDA not tested; Kaggle private upload was blocked",
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
    report=run_sampled_scenario_network_probe(
        envs=args.envs,steps=args.steps,seed=args.seed,device=args.device,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
