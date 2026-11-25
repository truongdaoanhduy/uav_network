"""Device-resident bounded per-UAV application packet buffer for Zhang-Xue.

This is a persistent Warp data structure, not a full network simulator.
Provides admission, duplicate suppression, FIFO by admission serial,
blackout-safe waiting, microsecond-exact TTL and stale-completion checks.

The queue holds only wire metadata, not payload content or routing tables.
A future full network must add a source-owned evidence payload store,
per-hop route selection, E2E ACK, ACK watermarks and actual MAC service.
Python API copies small operation inputs/outputs across host/device for
controlled preflight tests. Keep the device arrays resident across steps.
"""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _queue_reset(
    state: wp.array(dtype=wp.int32,ndim=3),
    packet_id: wp.array(dtype=wp.int64,ndim=3),
    dst: wp.array(dtype=wp.int32,ndim=3),
    size_bytes: wp.array(dtype=wp.int32,ndim=3),
    kind: wp.array(dtype=wp.int32,ndim=3),
    created_us: wp.array(dtype=wp.int64,ndim=3),
    ttl_us: wp.array(dtype=wp.int64,ndim=3),
    serial: wp.array(dtype=wp.int64,ndim=3),
    next_serial: wp.array(dtype=wp.int64,ndim=2),
    duplicate: wp.array(dtype=wp.int32,ndim=2),
    dropped_full: wp.array(dtype=wp.int32,ndim=2),
    expired: wp.array(dtype=wp.int32,ndim=2),
    mac_acked: wp.array(dtype=wp.int32,ndim=2),
    mac_dropped: wp.array(dtype=wp.int32,ndim=2),
    invalid_completion: wp.array(dtype=wp.int32,ndim=2),
    qcapacity: int,
):
    e,u=wp.tid()
    next_serial[e,u]=wp.int64(0)
    duplicate[e,u]=0
    dropped_full[e,u]=0
    expired[e,u]=0
    mac_acked[e,u]=0
    mac_dropped[e,u]=0
    invalid_completion[e,u]=0
    for slot in range(qcapacity):
        state[e,u,slot]=0
        packet_id[e,u,slot]=wp.int64(-1)
        dst[e,u,slot]=-1
        size_bytes[e,u,slot]=0
        kind[e,u,slot]=0
        created_us[e,u,slot]=wp.int64(0)
        ttl_us[e,u,slot]=wp.int64(0)
        serial[e,u,slot]=wp.int64(0)


@wp.kernel
def _queue_enqueue(
    new_id: wp.array(dtype=wp.int64,ndim=2),
    new_dst: wp.array(dtype=wp.int32,ndim=2),
    new_bytes: wp.array(dtype=wp.int32,ndim=2),
    new_kind: wp.array(dtype=wp.int32,ndim=2),
    new_created_us: wp.array(dtype=wp.int64,ndim=2),
    new_ttl_us: wp.array(dtype=wp.int64,ndim=2),
    state: wp.array(dtype=wp.int32,ndim=3),
    packet_id: wp.array(dtype=wp.int64,ndim=3),
    dst: wp.array(dtype=wp.int32,ndim=3),
    size_bytes: wp.array(dtype=wp.int32,ndim=3),
    kind: wp.array(dtype=wp.int32,ndim=3),
    created_us: wp.array(dtype=wp.int64,ndim=3),
    ttl_us: wp.array(dtype=wp.int64,ndim=3),
    serial: wp.array(dtype=wp.int64,ndim=3),
    next_serial: wp.array(dtype=wp.int64,ndim=2),
    duplicate: wp.array(dtype=wp.int32,ndim=2),
    dropped_full: wp.array(dtype=wp.int32,ndim=2),
    result: wp.array(dtype=wp.int32,ndim=2),
    qcapacity: int,
):
    e,u=wp.tid()
    candidate=new_id[e,u]
    result[e,u]=0
    if candidate<wp.int64(0):
        return
    empty=wp.int32(-1)
    for i in range(qcapacity):
        if state[e,u,i]!=0 and packet_id[e,u,i]==candidate:
            duplicate[e,u]+=1
            result[e,u]=-1
            return
        if state[e,u,i]==0 and empty<0:
            empty=i
    if empty<0:
        dropped_full[e,u]+=1
        result[e,u]=-2
        return
    state[e,u,empty]=1
    packet_id[e,u,empty]=candidate
    dst[e,u,empty]=new_dst[e,u]
    size_bytes[e,u,empty]=new_bytes[e,u]
    kind[e,u,empty]=new_kind[e,u]
    created_us[e,u,empty]=new_created_us[e,u]
    ttl_us[e,u,empty]=new_ttl_us[e,u]
    serial[e,u,empty]=next_serial[e,u]
    next_serial[e,u]+=wp.int64(1)
    result[e,u]=1


@wp.kernel
def _queue_expire(
    now: wp.array(dtype=wp.int64,ndim=1),
    state: wp.array(dtype=wp.int32,ndim=3),
    created_us: wp.array(dtype=wp.int64,ndim=3),
    ttl_us: wp.array(dtype=wp.int64,ndim=3),
    expired: wp.array(dtype=wp.int32,ndim=2),
    qcapacity: int,
):
    e,u=wp.tid()
    for i in range(qcapacity):
        if state[e,u,i]!=0 and now[e]>=created_us[e,u,i]+ttl_us[e,u,i]:
            state[e,u,i]=0
            expired[e,u]+=1


@wp.kernel
def _queue_dispatch(
    now: wp.array(dtype=wp.int64,ndim=1),
    radio: wp.array(dtype=wp.int32,ndim=2),
    state: wp.array(dtype=wp.int32,ndim=3),
    packet_id: wp.array(dtype=wp.int64,ndim=3),
    dst: wp.array(dtype=wp.int32,ndim=3),
    size_bytes: wp.array(dtype=wp.int32,ndim=3),
    kind: wp.array(dtype=wp.int32,ndim=3),
    created_us: wp.array(dtype=wp.int64,ndim=3),
    serial: wp.array(dtype=wp.int64,ndim=3),
    out_id: wp.array(dtype=wp.int64,ndim=2),
    out_dst: wp.array(dtype=wp.int32,ndim=2),
    out_bytes: wp.array(dtype=wp.int32,ndim=2),
    out_kind: wp.array(dtype=wp.int32,ndim=2),
    out_created: wp.array(dtype=wp.int64,ndim=2),
    out_ttl: wp.array(dtype=wp.int64,ndim=2),
    ttl_us: wp.array(dtype=wp.int64,ndim=3),
    qcapacity: int,
):
    e,u=wp.tid()
    out_id[e,u]=wp.int64(-1)
    out_dst[e,u]=-1
    out_bytes[e,u]=0
    out_kind[e,u]=0
    out_created[e,u]=wp.int64(-1)
    out_ttl[e,u]=wp.int64(0)
    if radio[e,u]==0:
        return
    best=wp.int32(-1)
    best_serial=wp.int64(0)
    for i in range(qcapacity):
        # Per-source stop-and-wait: do not send another packet while
        # any previous source packet is awaiting a MAC completion.
        if state[e,u,i]==2:
            return
        if state[e,u,i]==1 and created_us[e,u,i]<=now[e]:
            if best<0 or serial[e,u,i]<best_serial:
                best=i
                best_serial=serial[e,u,i]
    if best>=0:
        state[e,u,best]=2
        out_id[e,u]=packet_id[e,u,best]
        out_dst[e,u]=dst[e,u,best]
        out_bytes[e,u]=size_bytes[e,u,best]
        out_kind[e,u]=kind[e,u,best]
        out_created[e,u]=created_us[e,u,best]
        out_ttl[e,u]=ttl_us[e,u,best]


@wp.kernel
def _queue_resolve(
    finished_id: wp.array(dtype=wp.int64,ndim=2),
    outcome: wp.array(dtype=wp.int32,ndim=2),
    state: wp.array(dtype=wp.int32,ndim=3),
    packet_id: wp.array(dtype=wp.int64,ndim=3),
    mac_acked: wp.array(dtype=wp.int32,ndim=2),
    mac_dropped: wp.array(dtype=wp.int32,ndim=2),
    invalid_completion: wp.array(dtype=wp.int32,ndim=2),
    out_status: wp.array(dtype=wp.int32,ndim=2),
    qcapacity: int,
):
    e,u=wp.tid()
    out_status[e,u]=0
    if finished_id[e,u]<wp.int64(0):
        return
    for i in range(qcapacity):
        if state[e,u,i]==2 and packet_id[e,u,i]==finished_id[e,u]:
            code=outcome[e,u]
            if code==1:
                state[e,u,i]=0
                mac_acked[e,u]+=1
            elif code==2:
                state[e,u,i]=0
                mac_dropped[e,u]+=1
            elif code==3:
                state[e,u,i]=1
            out_status[e,u]=1
            return
    invalid_completion[e,u]+=1
    out_status[e,u]=-1


class ZhangXueWarpPacketQueue:
    """Persistent 6-UAV Warp metadata queue with FIFO and TTL.

    enqueue: up to one new packet per UAV per call.
    dispatch: one MAC packet per UAV per call; freezes during blackout.
    resolve outcome codes: 1=MAC ACK, 2=MAC retry exhausted,
    3=MAC deferred/retry later, 0=NO_PACKET only.
    The app DATA receiver can already have evidence even when outcome=2.
    """

    def __init__(self,num_envs:int,capacity:int=50,*,device="cpu"):
        if not isinstance(num_envs,int) or num_envs<1:
            raise ValueError("num_envs must be positive")
        if not isinstance(capacity,int) or not 1<=capacity<=200:
            raise ValueError("capacity must be in [1,200]")
        self.num_envs=num_envs
        self.capacity=capacity
        self.device=device
        wp.init()
        def alloc(dtype,shape):
            return wp.zeros(shape,dtype=dtype,device=device)
        s=(num_envs,6,capacity)
        n=(num_envs,6)
        self.state=alloc(wp.int32,s)
        self.packet_id=alloc(wp.int64,s)
        self.dst=alloc(wp.int32,s)
        self.size_bytes=alloc(wp.int32,s)
        self.kind=alloc(wp.int32,s)
        self.created_us=alloc(wp.int64,s)
        self.ttl_us=alloc(wp.int64,s)
        self.serial=alloc(wp.int64,s)
        self.next_serial=alloc(wp.int64,n)
        self.duplicate=alloc(wp.int32,n)
        self.dropped_full=alloc(wp.int32,n)
        self.expired=alloc(wp.int32,n)
        self.mac_acked=alloc(wp.int32,n)
        self.mac_dropped=alloc(wp.int32,n)
        self.invalid_completion=alloc(wp.int32,n)
        self.reset()

    def _device_array(self,source,dtype):
        return wp.array(source,dtype=dtype,device=self.device)

    def reset(self):
        wp.launch(_queue_reset,dim=(self.num_envs,6),inputs=[
            self.state,self.packet_id,self.dst,self.size_bytes,
            self.kind,self.created_us,self.ttl_us,self.serial,
            self.next_serial,self.duplicate,self.dropped_full,
            self.expired,self.mac_acked,self.mac_dropped,
            self.invalid_completion,self.capacity,
        ],device=self.device)

    def enqueue(self,packet_id,dest,frame_bytes,kind,created_us,ttl_us):
        expected=(self.num_envs,6)
        fields=[np.asarray(a) for a in (
            packet_id,dest,frame_bytes,kind,created_us,ttl_us,
        )]
        if any(a.shape!=expected for a in fields):
            raise ValueError("all enqueue fields must be [E,6]")
        if any(not np.issubdtype(a.dtype,np.integer) for a in fields):
            raise ValueError("enqueue fields must use integer dtypes")
        pid,dst,size,k,created,ttl=fields
        active=pid>=0
        if np.any(pid<-1):
            raise ValueError("packet_id must be >=0 or NO_PACKET=-1")
        if (np.any((dst[active]<0)|(dst[active]>=6)) or
            np.any(active&(dst==np.arange(6)[None,:]))):
            raise ValueError("invalid destination")
        data=active&(k==0)
        app_ack=active&(k==1)
        if np.any(data&((size<70)|(size>566)|((size-70)%8!=0))):
            raise ValueError("DATA frame size must be 70+8*N bytes, N<=62")
        if np.any(app_ack&(size!=90)):
            raise ValueError("E2E ACK frame must be 90 bytes")
        if np.any((k[active]<0)|(k[active]>1)):
            raise ValueError("kind must be DATA=0 or E2E_ACK=1")
        if np.any(created[active]<0):
            raise ValueError("created_us must be nonnegative")
        if np.any(ttl[active]<=0):
            raise ValueError("ttl_us must be positive")
        n=self.num_envs
        result=wp.zeros((n,6),dtype=wp.int32,device=self.device)
        wp.launch(_queue_enqueue,dim=(n,6),inputs=[
            self._device_array(pid.astype(np.int64),wp.int64),
            self._device_array(dst.astype(np.int32),wp.int32),
            self._device_array(size.astype(np.int32),wp.int32),
            self._device_array(k.astype(np.int32),wp.int32),
            self._device_array(created.astype(np.int64),wp.int64),
            self._device_array(ttl.astype(np.int64),wp.int64),
            self.state,self.packet_id,self.dst,self.size_bytes,
            self.kind,self.created_us,self.ttl_us,self.serial,
            self.next_serial,self.duplicate,self.dropped_full,
            result,self.capacity,
        ],device=self.device)
        return result.numpy()

    def _clock(self,now_us):
        n=self.num_envs
        if isinstance(now_us,(int,np.integer)):
            if int(now_us)<0:
                raise ValueError("time must be nonnegative")
            times=np.full((n,),int(now_us),np.int64)
        else:
            times=np.asarray(now_us)
            if times.shape!=(n,) or not np.issubdtype(times.dtype,np.integer):
                raise ValueError("time must be integer scalar or [E] array")
            if np.any(times<0):
                raise ValueError("time must be nonnegative")
            times=times.astype(np.int64)
        return self._device_array(times,wp.int64)

    def dispatch(self,now_us,*,radio_enabled=None):
        now=self._clock(now_us)
        n=self.num_envs
        if radio_enabled is None:
            radio=np.ones((n,6),np.int32)
        else:
            radio=np.asarray(radio_enabled)
            if radio.shape!=(n,6) or np.any((radio!=0)&(radio!=1)):
                raise ValueError("radio_enabled must be binary [E,6]")
            radio=radio.astype(np.int32)
        wp.launch(_queue_expire,dim=(n,6),inputs=[
            now,self.state,self.created_us,self.ttl_us,self.expired,
            self.capacity,
        ],device=self.device)
        outs=[
            wp.zeros((n,6),dtype=dtype,device=self.device)
            for dtype in (wp.int64,wp.int32,wp.int32,wp.int32,wp.int64,wp.int64)
        ]
        wp.launch(_queue_dispatch,dim=(n,6),inputs=[
            now,self._device_array(radio,wp.int32),
            self.state,self.packet_id,self.dst,self.size_bytes,
            self.kind,self.created_us,self.serial,
            *outs,self.ttl_us,self.capacity,
        ],device=self.device)
        return dict(zip((
            "packet_id","dest","frame_bytes","kind","created_us","ttl_us",
        ),(x.numpy() for x in outs)))

    def resolve(self,completed_id,outcome):
        ids=np.asarray(completed_id)
        statuses=np.asarray(outcome)
        n=self.num_envs
        if ids.shape!=(n,6) or statuses.shape!=(n,6):
            raise ValueError("completion arrays must be [E,6]")
        if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(statuses.dtype,np.integer):
            raise ValueError("completion arrays must be integers")
        if np.any(~np.isin(statuses,[0,1,2,3])):
            raise ValueError("unsupported completion outcome")
        if np.any((ids>=0)&(statuses==0)):
            raise ValueError("active completion requires 1/2/3")
        result=wp.zeros((n,6),dtype=wp.int32,device=self.device)
        wp.launch(_queue_resolve,dim=(n,6),inputs=[
            self._device_array(ids.astype(np.int64),wp.int64),
            self._device_array(statuses.astype(np.int32),wp.int32),
            self.state,self.packet_id,self.mac_acked,self.mac_dropped,
            self.invalid_completion,result,self.capacity,
        ],device=self.device)
        return result.numpy()

    def snapshot(self):
        state=self.state.numpy()
        return {
            "queue_length":np.count_nonzero(state!=0,axis=2).astype(np.int32),
            "inflight":np.count_nonzero(state==2,axis=2).astype(np.int32),
            "duplicate":self.duplicate.numpy(),
            "dropped_full":self.dropped_full.numpy(),
            "expired":self.expired.numpy(),
            "mac_acked":self.mac_acked.numpy(),
            "mac_dropped":self.mac_dropped.numpy(),
            "invalid_completion":self.invalid_completion.numpy(),
        }
