"""Bounded, collision-aware *one-hop* Warp retry prototype for Zhang-Xue.

New gate 5G engineering prototype, NOT a SimPy-exact MAC or a production
network backend. Per world, schedule up to five sender attempts in simulated
microseconds. Respect source-side 750m CCA, source RNG draws provided by
caller, data-airtime, overlapping-frame SINR, ACK timeout and TTL.

Known differences from upstream: CCA interruption freezes backoff only
approximately; successful MAC ACK is inferred from DATA success and isolated
reverse-link SNR (ACK interference is not simulated); no routing, HELLO,
application queues, E2E ACK, destination fusion or time-step persistence.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from uav_network.communication.gpu.classic_v1_interference_warp import _friis_gain


@wp.kernel
def _retry_one_hop(
    positions: wp.array(dtype=wp.vec3f, ndim=2),
    dest: wp.array(dtype=wp.int32, ndim=2),
    frame_bytes: wp.array(dtype=wp.int32, ndim=2),
    channel: wp.array(dtype=wp.int32, ndim=2),
    draw_slots: wp.array(dtype=wp.int32, ndim=3),
    deadline_us: wp.array(dtype=wp.float32, ndim=2),
    blackout: wp.array(dtype=wp.int32, ndim=2),
    phase: wp.array(dtype=wp.int32, ndim=2),
    next_start: wp.array(dtype=wp.float32, ndim=2),
    data_end: wp.array(dtype=wp.float32, ndim=2),
    busy_end: wp.array(dtype=wp.float32, ndim=2),
    latest_start: wp.array(dtype=wp.float32, ndim=2),
    collided: wp.array(dtype=wp.int32, ndim=2),
    ack_until: wp.array(dtype=wp.float32, ndim=2),
    ack_failed: wp.array(dtype=wp.int32, ndim=2),
    attempts: wp.array(dtype=wp.int32, ndim=2),
    delivered: wp.array(dtype=wp.int32, ndim=2),
    expired: wp.array(dtype=wp.int32, ndim=2),
    dropped: wp.array(dtype=wp.int32, ndim=2),
    deferred: wp.array(dtype=wp.int32, ndim=2),
    start_trace: wp.array(dtype=wp.float32, ndim=3),
    end_trace: wp.array(dtype=wp.float32, ndim=3),
    collision_trace: wp.array(dtype=wp.int32, ndim=3),
    ack_collision_trace: wp.array(dtype=wp.int32, ndim=3),
    tx_power: float,
    noise: float,
    threshold_db: float,
):
    e = wp.tid()
    for u in range(6):
        phase[e,u] = 3 if dest[e,u] < 0 else 0
        next_start[e,u] = wp.float32(50 + 20*draw_slots[e,u,0])
        data_end[e,u] = -1.0
        busy_end[e,u] = -1.0
        latest_start[e,u] = -1.0
        collided[e,u] = 0
        ack_until[e,u] = -1.0
        ack_failed[e,u] = 0
        attempts[e,u] = 0
        delivered[e,u] = 0
        expired[e,u] = 0
        dropped[e,u] = 0
        deferred[e,u] = 0
        for a in range(5):
            start_trace[e,u,a] = -1.0
            end_trace[e,u,a] = -1.0
            collision_trace[e,u,a] = 0
            ack_collision_trace[e,u,a] = 0

    # States: 0 waiting, 1 DATA airborne, 2 delivered, 3 terminal,
    # 4 MAC ACK airborne, 5 waiting ACK timeout; no application E2E ACK.
    # At most 30 start + 30 DATA end + 30 ACK end + 30 timeout events.
    for _ in range(220):
        chosen = wp.int32(-1)
        event_kind = wp.int32(-1)
        event_t = wp.float32(1.0e20)
        for u in range(6):
            if phase[e,u] == 0:
                t = next_start[e,u]
                if t < event_t:
                    chosen = u
                    event_t = t
                    event_kind = 0
            elif phase[e,u] == 1:
                t = data_end[e,u]
                if t <= event_t:
                    chosen = u
                    event_t = t
                    event_kind = 1
            elif phase[e,u] == 4:
                t = ack_until[e,u]
                if t <= event_t:
                    chosen = u
                    event_t = t
                    event_kind = 2
            elif phase[e,u] == 5:
                t = next_start[e,u]
                if t <= event_t:
                    chosen = u
                    event_t = t
                    event_kind = 3
        if chosen < 0:
            break

        if event_kind == 0:
            if event_t >= deadline_us[e,chosen]:
                expired[e,chosen] = 1
                phase[e,chosen] = 3
                continue
            if blackout[e,chosen] != 0:
                # Static blackout for this bounded packet trial:
                # packet stays unsent until its deadline.
                expired[e,chosen] = 1
                phase[e,chosen] = 3
                continue
            occupied_until = wp.float32(-1.0)
            for j in range(6):
                if j == chosen or latest_start[e,j] < 0.0:
                    continue
                if latest_start[e,j] <= event_t and event_t < busy_end[e,j]:
                    d = positions[e,chosen] - positions[e,j]
                    if wp.dot(d,d) < wp.float32(750.0*750.0):
                        occupied_until = wp.max(occupied_until, busy_end[e,j])
            if occupied_until >= 0.0:
                # Approximation: restart DIFS at end of carrier occupancy;
                # actual SimPy freezes remainder of interrupted backoff.
                next_start[e,chosen] = occupied_until + 50.0
                deferred[e,chosen] += 1
                continue

            a = attempts[e,chosen]
            latest_start[e,chosen] = event_t
            data_end[e,chosen] = event_t + wp.float32(frame_bytes[e,chosen])*4.0
            busy_end[e,chosen] = data_end[e,chosen] + 130.0
            phase[e,chosen] = 1
            attempts[e,chosen] = a + 1
            collided[e,chosen] = 0
            start_trace[e,chosen,a] = event_t
            end_trace[e,chosen,a] = data_end[e,chosen]

            # Evaluate SINR over ALL simultaneous on-air sources, not
            # independent link SNR. Conservatively flag any frame with
            # inadequate SINR during overlap; flags persist to completion.
            for j in range(6):
                if phase[e,j] != 1:
                    continue
                rx = dest[e,j]
                sig = tx_power * _friis_gain(positions[e,j],positions[e,rx],2.4e9)
                interference = wp.float32(0.0)
                for k in range(6):
                    if k == j or phase[e,k] != 1:
                        continue
                    if wp.abs(channel[e,k] - channel[e,j]) < 5:
                        interference += tx_power * _friis_gain(
                            positions[e,k],positions[e,rx],2.4e9
                        )
                # A simultaneous reverse ACK can also jam an in-flight
                # DATA receiver on an overlapping channel.
                for k in range(6):
                    if phase[e,k] == 4 and wp.abs(channel[e,k]-channel[e,j]) < 5:
                        if data_end[e,k]+10.0 < data_end[e,j] and latest_start[e,j] < ack_until[e,k]:
                            interference += tx_power*_friis_gain(
                                positions[e,dest[e,k]],positions[e,rx],2.4e9
                            )
                db = wp.float32(4.3429448190325175)*wp.log(
                    sig/(noise+interference)
                )
                if db < threshold_db or (phase[e,rx] == 1):
                    collided[e,j] = 1
                    collision_trace[e,j,attempts[e,j]-1] = 1
            # Newly-started DATA can jam already airborne reverse ACKs.
            for j in range(6):
                if phase[e,j] != 4 or wp.abs(channel[e,j]-channel[e,chosen]) >= 5:
                    continue
                if event_t < ack_until[e,j] and data_end[e,chosen] > data_end[e,j]+10.0:
                    sig = tx_power*_friis_gain(positions[e,dest[e,j]],positions[e,j],2.4e9)
                    interf = tx_power*_friis_gain(positions[e,chosen],positions[e,j],2.4e9)
                    if wp.float32(4.3429448190325175)*wp.log(sig/(noise+interf)) < threshold_db:
                        ack_failed[e,j] = 1
                        ack_collision_trace[e,j,attempts[e,j]-1] = 1
        elif event_kind == 1:
            a = attempts[e,chosen]
            rx = dest[e,chosen]
            reverse_snr = tx_power*_friis_gain(
                positions[e,rx],positions[e,chosen],2.4e9
            )/noise
            reverse_snr_db = wp.float32(4.3429448190325175)*wp.log(
                reverse_snr
            )
            if data_end[e,chosen] + 130.0 > deadline_us[e,chosen]:
                expired[e,chosen] = 1
                phase[e,chosen] = 3
            elif (collided[e,chosen] == 0 and blackout[e,rx] == 0
                  and reverse_snr_db >= threshold_db):
                # DATA was received; the reverse MAC ACK needs its own
                # interference test at the sender, and SIFS+ACK airtime.
                ack_failed[e,chosen] = 0
                ack_until[e,chosen] = data_end[e,chosen] + 130.0
                phase[e,chosen] = 4
                for k in range(6):
                    if k == chosen or phase[e,k] != 1:
                        continue
                    if wp.abs(channel[e,k]-channel[e,chosen]) >= 5:
                        continue
                    if data_end[e,k] > data_end[e,chosen]+10.0:
                        sig = tx_power*_friis_gain(
                            positions[e,rx],positions[e,chosen],2.4e9
                        )
                        interf = tx_power*_friis_gain(
                            positions[e,k],positions[e,chosen],2.4e9
                        )
                        db = wp.float32(4.3429448190325175)*wp.log(
                            sig/(noise+interf)
                        )
                        if db < threshold_db:
                            ack_failed[e,chosen] = 1
                            ack_collision_trace[e,chosen,a-1] = 1
            else:
                phase[e,chosen] = 5
                next_start[e,chosen] = data_end[e,chosen] + 180.0
        elif event_kind == 2:
            if ack_failed[e,chosen] == 0:
                delivered[e,chosen] = 1
                phase[e,chosen] = 2
            else:
                phase[e,chosen] = 5
                next_start[e,chosen] = data_end[e,chosen] + 180.0
        else:
            # MAC ACK timeout. Native v1 retry requeues before the
            # next DIFS+seeded contention window.
            a = attempts[e,chosen]
            if event_t >= deadline_us[e,chosen]:
                expired[e,chosen] = 1
                phase[e,chosen] = 3
            elif a >= 5:
                dropped[e,chosen] = 1
                phase[e,chosen] = 3
            else:
                phase[e,chosen] = 0
                next_start[e,chosen] = (
                    event_t + 50.0 + wp.float32(20*draw_slots[e,chosen,a])
                )


def simulate_one_hop_retry_batch(
    positions_xyz, destinations, frame_bytes, channels, backoff_draw_slots,
    deadline_us, *, blackout=None, tx_power_w=0.5, device="cpu",
):
    """Simulate independent worlds with one pending unicast packet/UAV.

    Supplied backoff draws have shape [E,6,5]; for attempt k=1..5,
    the source's randint range is [0, (32*2**(k-1)-1)-1].
    All outputs are [E,6] except per-attempt traces [E,6,5].
    """
    xyz = np.asarray(positions_xyz,dtype=np.float32)
    dst = np.asarray(destinations,dtype=np.int32)
    sizes = np.asarray(frame_bytes,dtype=np.int32)
    ch = np.asarray(channels,dtype=np.int32)
    slots = np.asarray(backoff_draw_slots)
    expiry = np.asarray(deadline_us,dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[1:] != (6,3) or xyz.shape[0] < 1:
        raise ValueError("positions must have shape [E,6,3]")
    e = xyz.shape[0]
    if any(a.shape != (e,6) for a in (dst,sizes,ch,expiry)):
        raise ValueError("per-UAV fields must have shape [E,6]")
    if slots.shape != (e,6,5) or not np.issubdtype(slots.dtype,np.integer):
        raise ValueError("backoff draws must be integer [E,6,5]")
    for a in range(5):
        if np.any((slots[:,:,a] < 0) | (slots[:,:,a] >= 32*(2**a)-1)):
            raise ValueError(f"backoff draw outside classic-v1 CW at attempt {a+1}")
    if blackout is None:
        blackout = np.zeros((e,6),np.int32)
    blackout = np.asarray(blackout,dtype=np.int32)
    if blackout.shape != (e,6) or np.any((blackout!=0)&(blackout!=1)):
        raise ValueError("blackout must be [E,6] binary")
    if (np.any((dst<-1)|(dst>=6)) or
        np.any((dst>=0)&(dst==np.arange(6)[None,:]))):
        raise ValueError("invalid unicast destination")
    if np.any((sizes<0)|((dst>=0)&(sizes==0))):
        raise ValueError("active data frames must have positive length")
    if np.any((ch<1)|(ch>14)):
        raise ValueError("IEEE 802.11b channels must be 1..14")
    if not np.isfinite(xyz).all() or not np.isfinite(expiry).all():
        raise ValueError("nonfinite position or deadline")
    if np.any(expiry <= 0) or not np.isfinite(tx_power_w) or tx_power_w <= 0:
        raise ValueError("positive deadline and transmit power required")
    wp.init()
    def arr(a, dtype):
        return wp.array(a,dtype=dtype,device=device)
    pos_w = arr(xyz,wp.vec3f)
    dst_w = arr(dst,wp.int32)
    size_w = arr(sizes,wp.int32)
    ch_w = arr(ch,wp.int32)
    slots_w = arr(slots.astype(np.int32),wp.int32)
    ttl_w = arr(expiry,wp.float32)
    black_w = arr(blackout,wp.int32)
    floats = [wp.zeros((e,6),dtype=wp.float32,device=device) for _ in range(4)]
    phase_w, collided_w, ack_failed_w, attempts_w, delivered_w, expired_w, dropped_w, deferred_w = [
        wp.zeros((e,6),dtype=wp.int32,device=device) for _ in range(8)
    ]
    trace_start = wp.zeros((e,6,5),dtype=wp.float32,device=device)
    trace_end = wp.zeros((e,6,5),dtype=wp.float32,device=device)
    trace_collision = wp.zeros((e,6,5),dtype=wp.int32,device=device)
    trace_ack_collision = wp.zeros((e,6,5),dtype=wp.int32,device=device)
    start_latest = wp.zeros((e,6),dtype=wp.float32,device=device)
    wp.launch(_retry_one_hop,dim=e,inputs=[
        pos_w,dst_w,size_w,ch_w,slots_w,ttl_w,black_w,
        phase_w,floats[0],floats[1],floats[2],start_latest,collided_w,
        floats[3],ack_failed_w,attempts_w,delivered_w,expired_w,dropped_w,deferred_w,
        trace_start,trace_end,trace_collision,trace_ack_collision,
        float(tx_power_w),4e-11,6.0,
    ],device=device)
    return {
        "attempts":attempts_w.numpy(),"delivered":delivered_w.numpy(),
        "expired":expired_w.numpy(),"dropped":dropped_w.numpy(),
        "cca_deferrals":deferred_w.numpy(),
        "start_us":trace_start.numpy(),"end_us":trace_end.numpy(),
        "collided":trace_collision.numpy(),
        "ack_collided":trace_ack_collision.numpy(),
        "stage":"bounded_one_hop_retry_partial_ack_interference_not_full_gate5",
    }
