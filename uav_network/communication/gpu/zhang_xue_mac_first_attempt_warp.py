"""First-attempt 802.11b MAC scheduling on Warp for Zhang–Xue network pilot.

This kernel combines *initial* DIFS/CW backoff, one-hop CCA deferral,
frame airtime and receiver SINR on overlapping transmissions. It is a
scientific preflight component, not a complete MAC state machine.
Missing: retry attempts, packet TTL/queue, MAC ACK reception, E2E ACK
and multi-hop routing. Do not use as the RL training network yet.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from uav_network.communication.gpu.classic_v1_interference_warp import (
    compute_simultaneous_link_sinr,
)


@wp.kernel
def _schedule_initial(
    positions: wp.array(dtype=wp.vec3f,ndim=2),
    destinations: wp.array(dtype=wp.int32,ndim=2),
    frame_bytes: wp.array(dtype=wp.int32,ndim=2),
    backoff_slots: wp.array(dtype=wp.int32,ndim=2),
    start_out: wp.array(dtype=wp.float32,ndim=2),
    end_out: wp.array(dtype=wp.float32,ndim=2),
    reservation_end: wp.array(dtype=wp.float32,ndim=2),
    cca_deferrals: wp.array(dtype=wp.int32,ndim=2),
    scheduled: wp.array(dtype=wp.int32,ndim=2),
):
    e=wp.tid()
    # All time quantities use microseconds, matching v1 CSMA source.
    for u in range(6):
        start_out[e,u]=wp.float32(50+20*backoff_slots[e,u])
        end_out[e,u]=wp.float32(-1.)
        reservation_end[e,u]=wp.float32(-1.)
        cca_deferrals[e,u]=0
        scheduled[e,u]=0

    for _event in range(80):
        chosen=wp.int32(-1)
        soonest=wp.float32(1.e20)
        for u in range(6):
            if destinations[e,u]<0 or scheduled[e,u]!=0:
                continue
            t=start_out[e,u]
            if t<soonest:
                chosen=u
                soonest=t
        if chosen<0:
            break
        # CCA senses any other source occupying the channel within 750m,
        # regardless of data-channel index (same v1 check_channel_availability).
        must_wait=wp.int32(0)
        after=soonest
        for j in range(6):
            if scheduled[e,j]==0 or j==chosen:
                continue
            if start_out[e,j]>soonest or reservation_end[e,j]<=soonest:
                continue
            x=positions[e,j][0]-positions[e,chosen][0]
            y=positions[e,j][1]-positions[e,chosen][1]
            z=positions[e,j][2]-positions[e,chosen][2]
            if x*x+y*y+z*z < wp.float32(750.*750.):
                must_wait=1
                after=wp.max(after,reservation_end[e,j])

        if must_wait:
            # Approximate the v1 interrupted-backoff logic: frozen backoff
            # has already elapsed when the channel becomes occupied.
            # Restart DIFS once after last busy resource ends.
            start_out[e,chosen]=after+wp.float32(50.)
            cca_deferrals[e,chosen]+=1
        else:
            scheduled[e,chosen]=1
            end_out[e,chosen]=soonest+wp.float32(4.)*wp.float32(frame_bytes[e,chosen])
            # V1 source reserves SIFS10us + 30-byte MAC ACK at 2Mbps (120us)
            reservation_end[e,chosen]=end_out[e,chosen]+wp.float32(130.)


def schedule_initial_csma_frame_attempts(
    positions_xyz,
    destinations,
    frame_bytes,
    channels,
    backoff_draw_slots,
    *,
    tx_power_w=.5,
    device="cpu",
):
    """First on-air attempt of up to six UAV packets per independent world.

    Indices 0..5 identify the source UAV; -1 means NO_SEND.
    Only direct one-hop transmissions are modeled here.
    """
    xyz=np.asarray(positions_xyz,dtype=np.float32)
    dest=np.asarray(destinations,dtype=np.int32)
    sizes=np.asarray(frame_bytes,dtype=np.int32)
    ch=np.asarray(channels,dtype=np.int32)
    slots=np.asarray(backoff_draw_slots,dtype=np.int32)
    if xyz.ndim!=3 or xyz.shape[1:]!=(6,3) or xyz.shape[0]<1:
        raise ValueError("positions must be [E,6,3]")
    e=xyz.shape[0]
    for a in (dest,sizes,ch,slots):
        if a.shape!=(e,6):
            raise ValueError("MAC arrays must be [E,6]")
    if not np.isfinite(xyz).all() or np.any((dest<-1)|(dest>=6)):
        raise ValueError("Invalid position or destination")
    if np.any((dest>=0)&(dest==np.arange(6)[None,:])):
        raise ValueError("No self-send allowed")
    if np.any((sizes<0)|((dest>=0)&(sizes<=0))):
        raise ValueError("Invalid packet frame length")
    if np.any((ch<1)|(ch>14)) or np.any((slots<0)|(slots>=31)):
        raise ValueError("Invalid channel or first-attempt CW draw")
    wp.init()
    pos_w=wp.array(xyz,dtype=wp.vec3f,device=device)
    dest_w=wp.array(dest,dtype=wp.int32,device=device)
    bytes_w=wp.array(sizes,dtype=wp.int32,device=device)
    slots_w=wp.array(slots,dtype=wp.int32,device=device)
    times_w=wp.zeros((e,6),dtype=wp.float32,device=device)
    ends_w=wp.zeros((e,6),dtype=wp.float32,device=device)
    busy_w=wp.zeros((e,6),dtype=wp.float32,device=device)
    defer_w=wp.zeros((e,6),dtype=wp.int32,device=device)
    schedule_w=wp.zeros((e,6),dtype=wp.int32,device=device)
    wp.launch(
        _schedule_initial,dim=e,
        inputs=[pos_w,dest_w,bytes_w,slots_w,times_w,ends_w,busy_w,
                defer_w,schedule_w],device=device,
    )
    valid=(dest>=0).astype(np.int32)
    schedule=schedule_w.numpy()
    if np.any((schedule!=1)&(valid==1)):
        raise RuntimeError("Warp CCA scheduler failed to schedule a valid frame")
    starts_us=times_w.numpy()
    ends_us=ends_w.numpy()
    # For the PHY of the non-transmitting placeholders, use a harmless
    # positive duration and in-bounds destination; valid mask ignores them.
    src=np.broadcast_to(np.arange(6,dtype=np.int32),(e,6)).copy()
    rx=np.maximum(dest,0)
    duration=np.where(valid!=0,np.maximum(sizes,1)*4.,1.).astype(np.float32)/1e6
    result=compute_simultaneous_link_sinr(
        xyz,src,rx,ch,starts_us/1e6,duration,valid,
        tx_power_w=tx_power_w,device=device,
    )
    return {
        "start_us":starts_us,
        "end_us":ends_us,
        "cca_deferrals":defer_w.numpy(),
        "success":result["success"],
        "sinr_db":result["sinr_db"],
        "valid":valid,
        "stage":"one_hop_first_mac_attempt_only_not_full_gate5",
    }
