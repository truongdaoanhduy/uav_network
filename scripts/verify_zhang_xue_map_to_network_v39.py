"""Diagnostic single-step map→packet→one-hop Warp network smoke, NO training.

Purpose: catch source-ownership, byte-format, blackout and delivery barriers
while Gate5 remains blocked. This is NOT the production distributed DS
fusion, not multi-hop and not a substitute for native SimPy parity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from uav_network.communication.gpu.zhang_xue_mac_retry_warp import simulate_one_hop_retry_batch
from uav_network.communication.zhang_xue_ref.packets_and_queues import (
    APP_HEADER_BYTES, ENTRY_BYTES, TOTAL_NET_HEADER_BYTES, MAX_ENTRIES_PER_PACKET,
)
from uav_network.environment.zhang_xue_coop import ZhangXueConfig
from uav_network.environment.zhang_xue_map_geometry import sample_scenario_batch
from uav_network.environment.gpu.zhang_xue_map_warp import ZhangXueWarpMapBatch
from uav_network.environment.gpu.zhang_xue_motion_warp import warp_move_batch


def run_map_network_probe(envs=128, seed=44, device="cpu"):
    cfg=ZhangXueConfig()
    seeds=np.arange(seed,seed+envs,dtype=np.int64)
    t0=time.perf_counter()
    scenes=sample_scenario_batch(cfg,seeds)
    rng=np.random.default_rng(seed+777)
    position_xy,_,safety,boundary=warp_move_batch(
        scenes["positions"],scenes["no_fly_zones"],
        rng.uniform(-1,1,(envs,6,2)).astype(np.float32),
        dt=cfg.dt_seconds,max_speed=cfg.v_max_mps,
        safe_distance=cfg.safe_distance_meter,device=device,
    )
    maps=ZhangXueWarpMapBatch(
        envs,device=device,altitude_m=cfg.fixed_altitude_meter,
        camera_fov_degrees=cfg.camera_fov_degrees,pd=cfg.pd,pf=cfg.pf,
    )
    maps.reset()
    maps.step(
        position_xy,scenes["target_mask"].astype(np.int32),
        seed=seed,step_index=0,
    )
    state=maps.read()
    counts=state["pos"].astype(np.int32)+state["neg"].astype(np.int32)
    assert counts.shape==(envs,6,400)
    assert all(np.isfinite(state[k]).all() for k in ("mass_target","unknown"))
    # Application packetization depends solely on own evidence counts.
    # Ground truth is never exposed to MAC, packet encoder or peer actor.
    index=np.full((envs,6,MAX_ENTRIES_PER_PACKET),-1,np.int32)
    entries=np.zeros((envs,6),np.int32)
    for e in range(envs):
        for u in range(6):
            selected=np.flatnonzero(counts[e,u])[:MAX_ENTRIES_PER_PACKET]
            n=len(selected)
            index[e,u,:n]=selected
            entries[e,u]=n
    destination=np.tile((np.arange(6,dtype=np.int32)+1)%6,(envs,1))
    destination[entries==0]=-1  # NO_SEND
    frame_bytes=np.where(
        entries>0,
        TOTAL_NET_HEADER_BYTES+APP_HEADER_BYTES+ENTRY_BYTES*entries,
        0,
    ).astype(np.int32)
    assert int(frame_bytes.max())<=TOTAL_NET_HEADER_BYTES+512
    xyz=np.empty((envs,6,3),np.float32)
    xyz[:,:,:2]=position_xy
    xyz[:,:,2]=cfg.fixed_altitude_meter
    grid_xy=np.floor(position_xy/cfg.cell_size_meter).astype(np.int32)
    blackout=np.zeros((envs,6),np.int32)
    for z in range(cfg.blackout_zone_count):
        delta=np.abs(grid_xy-scenes["blackout_zones"][:,z,None,:])
        blackout|=(delta.sum(axis=-1)<=cfg.blackout_radius_cells).astype(np.int32)
    slots=np.empty((envs,6,5),np.int32)
    for a in range(5):
        slots[:,:,a]=rng.integers(0,32*2**a-1,size=(envs,6))
    channels=np.ones((envs,6),np.int32)
    ttl=np.full((envs,6),cfg.packet_ttl_seconds*1e6,np.float32)
    t1=time.perf_counter()
    net=simulate_one_hop_retry_batch(
        xyz,destination,frame_bytes,channels,slots,ttl,
        blackout=blackout,tx_power_w=cfg.tx_power_w,device=device,
    )
    # A receiver only gets source measurements when DATA really arrived.
    # Apply count snapshots once (this smoke has no retransmitted app ACK).
    receiver_events={}
    # Receiver-local source-indexed count maps: the same representation as
    # BeliefTable [receiver, origin, cell], with an extra environment axis.
    # Unlike the diagnostic team map, NO receiver sees any remote count
    # until a MAC DATA frame is actually delivered to that receiver.
    r_pos=np.zeros((envs,6,6,400),np.uint16)
    r_neg=np.zeros_like(r_pos)
    ids=np.arange(6)
    r_pos[:,ids,ids,:]=state["pos"]
    r_neg[:,ids,ids,:]=state["neg"]
    for e,u in zip(*np.nonzero(net["delivered"])):
        rx=int(destination[e,u])
        selected=index[e,u,:entries[e,u]]
        if rx<0:
            raise AssertionError("no-send packet became delivered")
        key=(int(e),rx,int(u))
        if key in receiver_events:
            raise AssertionError("same source payload delivered more than once")
        p_data=state["pos"][e,u,selected].copy()
        n_data=state["neg"][e,u,selected].copy()
        receiver_events[key]=(p_data,n_data)
        r_pos[e,rx,u,selected]=np.maximum(
            r_pos[e,rx,u,selected],p_data,
        )
        r_neg[e,rx,u,selected]=np.maximum(
            r_neg[e,rx,u,selected],n_data,
        )
    assert len(receiver_events)==int(net["delivered"].sum())
    assert np.array_equal(r_pos[:,ids,ids,:],state["pos"])
    assert np.array_equal(r_neg[:,ids,ids,:],state["neg"])
    # Max-merge is idempotent even if duplicate frames arrive: no
    # double fusion/reward from the same cumulative source snapshot.
    before_pos=r_pos.copy()
    before_neg=r_neg.copy()
    for (e,rx,u),(p_data,n_data) in receiver_events.items():
        selected=index[e,u,:entries[e,u]]
        r_pos[e,rx,u,selected]=np.maximum(
            r_pos[e,rx,u,selected],p_data,
        )
        r_neg[e,rx,u,selected]=np.maximum(
            r_neg[e,rx,u,selected],n_data,
        )
    assert np.array_equal(before_pos,r_pos)
    assert np.array_equal(before_neg,r_neg)
    # Check two local recipients against the mathematical DS transform.
    # This assertion is diagnostic, not a Warp receiver fusion kernel.
    from uav_network.communication.zhang_xue_ref.ds_belief import (
        ds_log_space_safe,
    )
    for e in (0,envs-1):
        for rx in (0,5):
            p_total=r_pos[e,rx].sum(axis=0,dtype=np.int32)
            n_total=r_neg[e,rx].sum(axis=0,dtype=np.int32)
            mf,me,mu=ds_log_space_safe(p_total,n_total,cfg.ds_mf,cfg.ds_me)
            assert np.isfinite(mf).all() and np.isfinite(me).all()
            assert np.isfinite(mu).all()
            np.testing.assert_allclose(mf+me+mu,1.,atol=5e-6)
    remote_entries=int(np.count_nonzero(
        (r_pos+r_neg)[:,~np.eye(6,dtype=bool),:]
    ))
    assert np.all(net["delivered"][destination<0]==0)
    assert np.all(net["attempts"][blackout!=0]==0)
    assert np.all(index[entries==0]==-1)
    elapsed=time.perf_counter()-t0
    return {
        "stage":"map_v37_to_one_hop_mac_v39_preflight_only",
        "envs":envs,"map_device":device,"network_device":device,
        "scene_targets_per_env":cfg.target_count,
        "total_own_evidence_counts":int(counts.sum()),
        "packet_generated":int((destination>=0).sum()),
        "application_entries_total":int(entries.sum()),
        "frame_max_bytes":int(frame_bytes.max()),
        "blackout_uav_count":int(blackout.sum()),
        "first_network_data_delivered":int(net["delivered"].sum()),
        "first_network_data_expired":int(net["expired"].sum()),
        "first_network_mac_dropped":int(net["dropped"].sum()),
        "source_owned_receiver_events":len(receiver_events),
        "receiver_remote_nonzero_origin_cells":remote_entries,
        "receiver_max_merge_replay_idempotent":True,
        "receiver_ds_finite_and_normalized":True,
        "safety_rejections":int(safety.sum()),
        "boundary_rejections":int(boundary.sum()),
        "seconds_including_cpu_scene_build":elapsed,
        "truth_not_serialized":True,
        "delivery_barrier_verified":True,
        "full_network_gate5":"NOT_PASSED",
        "no_training":True,
        "limitations":[
            "Synthetic deterministic peer-action (u+1)%6; not a learned actor",
            "One packet/source with at most 62 source-owned evidence cells",
            "Received evidence gets NumPy receiver-local max+sum DS fusion only, not a persistent Warp fusion backend",
            "No routed E2E ACK, multihop or persistent queues",
            "No SimPy↔Warp event-trace parity; no CUDA driver on this host",
        ],
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs",type=int,default=128)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--output-json",type=Path)
    args=parser.parse_args()
    report=run_map_network_probe(args.envs,args.seed,args.device)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
