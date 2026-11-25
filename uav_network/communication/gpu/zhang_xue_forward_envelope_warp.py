"""Device-resident application forwarding envelope accompanying each MAC packet.

The link-layer queue holds next-hop, original creation time and TTL.
This slot-bound sidecar retains application origin/destination and visited
path across relays. Actual received DATA, not a MAC ACK, drives forwarding.
The six-UAV bitmask bounds length and rejects revisiting a node even when
routing information is stale. No target truth or global link oracle enters.
"""
from __future__ import annotations
import numpy as np
import warp as wp


@wp.kernel
def _reset(
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    original:wp.array(dtype=wp.int32,ndim=3),
    terminal:wp.array(dtype=wp.int32,ndim=3),
    hops:wp.array(dtype=wp.int32,ndim=3),
    visited:wp.array(dtype=wp.int32,ndim=3),
    capacity:int,
):
    e,u=wp.tid()
    for idx in range(capacity):
        slot_id[e,u,idx]=wp.int64(-1)
        original[e,u,idx]=-1
        terminal[e,u,idx]=-1
        hops[e,u,idx]=-1
        visited[e,u,idx]=0


@wp.kernel
def _attach(
    new_id:wp.array(dtype=wp.int64,ndim=2),
    origin:wp.array(dtype=wp.int32,ndim=2),
    final:wp.array(dtype=wp.int32,ndim=2),
    count:wp.array(dtype=wp.int32,ndim=2),
    seen:wp.array(dtype=wp.int32,ndim=2),
    q_state:wp.array(dtype=wp.int32,ndim=3),
    q_id:wp.array(dtype=wp.int64,ndim=3),
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    original:wp.array(dtype=wp.int32,ndim=3),
    terminal:wp.array(dtype=wp.int32,ndim=3),
    hops:wp.array(dtype=wp.int32,ndim=3),
    visited:wp.array(dtype=wp.int32,ndim=3),
    result:wp.array(dtype=wp.int32,ndim=2),
    capacity:int,
):
    e,u=wp.tid()
    result[e,u]=0
    pid=new_id[e,u]
    if pid<wp.int64(0):
        return
    for slot in range(capacity):
        if q_state[e,u,slot]!=0 and q_id[e,u,slot]==pid:
            if slot_id[e,u,slot]==pid:
                result[e,u]=-2
                return
            slot_id[e,u,slot]=pid
            original[e,u,slot]=origin[e,u]
            terminal[e,u,slot]=final[e,u]
            hops[e,u,slot]=count[e,u]
            visited[e,u,slot]=seen[e,u]
            result[e,u]=1
            return
    result[e,u]=-1


@wp.kernel
def _read_dispatch(
    sent_id:wp.array(dtype=wp.int64,ndim=2),
    q_state:wp.array(dtype=wp.int32,ndim=3),
    q_id:wp.array(dtype=wp.int64,ndim=3),
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    original:wp.array(dtype=wp.int32,ndim=3),
    terminal:wp.array(dtype=wp.int32,ndim=3),
    hops:wp.array(dtype=wp.int32,ndim=3),
    visited:wp.array(dtype=wp.int32,ndim=3),
    out_origin:wp.array(dtype=wp.int32,ndim=2),
    out_final:wp.array(dtype=wp.int32,ndim=2),
    out_hops:wp.array(dtype=wp.int32,ndim=2),
    out_seen:wp.array(dtype=wp.int32,ndim=2),
    capacity:int,
):
    e,u=wp.tid()
    out_origin[e,u]=-1
    out_final[e,u]=-1
    out_hops[e,u]=-1
    out_seen[e,u]=0
    pid=sent_id[e,u]
    if pid<wp.int64(0):
        return
    for idx in range(capacity):
        if (q_state[e,u,idx]==2 and q_id[e,u,idx]==pid and
            slot_id[e,u,idx]==pid):
            out_origin[e,u]=original[e,u,idx]
            out_final[e,u]=terminal[e,u,idx]
            out_hops[e,u]=hops[e,u,idx]
            out_seen[e,u]=visited[e,u,idx]
            return


@wp.kernel
def _classify(
    ids:wp.array(dtype=wp.int64,ndim=2),
    receiver:wp.array(dtype=wp.int32,ndim=2),
    physically_received:wp.array(dtype=wp.int32,ndim=2),
    q_state:wp.array(dtype=wp.int32,ndim=3),
    q_id:wp.array(dtype=wp.int64,ndim=3),
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    original:wp.array(dtype=wp.int32,ndim=3),
    terminal:wp.array(dtype=wp.int32,ndim=3),
    hops:wp.array(dtype=wp.int32,ndim=3),
    visited:wp.array(dtype=wp.int32,ndim=3),
    out_relay:wp.array(dtype=wp.int32,ndim=2),
    out_final:wp.array(dtype=wp.int32,ndim=2),
    out_loop:wp.array(dtype=wp.int32,ndim=2),
    out_ttl_hops:wp.array(dtype=wp.int32,ndim=2),
    out_origin:wp.array(dtype=wp.int32,ndim=2),
    out_ultimate:wp.array(dtype=wp.int32,ndim=2),
    out_hops:wp.array(dtype=wp.int32,ndim=2),
    out_visited:wp.array(dtype=wp.int32,ndim=2),
    capacity:int,
):
    e,u=wp.tid()
    out_relay[e,u]=0
    out_final[e,u]=0
    out_loop[e,u]=0
    out_ttl_hops[e,u]=0
    out_origin[e,u]=-1
    out_ultimate[e,u]=-1
    out_hops[e,u]=-1
    out_visited[e,u]=0
    if physically_received[e,u]==0 or ids[e,u]<wp.int64(0):
        return
    rx=receiver[e,u]
    for idx in range(capacity):
        if (q_state[e,u,idx]!=2 or q_id[e,u,idx]!=ids[e,u] or
            slot_id[e,u,idx]!=ids[e,u]):
            continue
        previous=visited[e,u,idx]
        hopped=hops[e,u,idx]+1
        out_origin[e,u]=original[e,u,idx]
        out_ultimate[e,u]=terminal[e,u,idx]
        out_hops[e,u]=hopped
        out_visited[e,u]=previous
        if (previous & (wp.int32(1)<<rx))!=0:
            out_loop[e,u]=1
            return
        out_visited[e,u]=previous|(wp.int32(1)<<rx)
        if rx==terminal[e,u,idx]:
            if hopped>5:
                out_ttl_hops[e,u]=1
            else:
                out_final[e,u]=1
        elif hopped>=5:
            out_ttl_hops[e,u]=1
        else:
            out_relay[e,u]=1
        return


class ZhangXueWarpForwardingEnvelope:
    """Immutable per-queue-slot app destination and loop-safe hop metadata."""

    def __init__(self,queue):
        self.q=queue
        self.e=queue.num_envs
        self.device=queue.device
        self.capacity=queue.capacity
        shape=(self.e,6,self.capacity)
        self.slot_id=wp.zeros(shape,dtype=wp.int64,device=self.device)
        self.original=wp.zeros(shape,dtype=wp.int32,device=self.device)
        self.terminal=wp.zeros(shape,dtype=wp.int32,device=self.device)
        self.hops=wp.zeros(shape,dtype=wp.int32,device=self.device)
        self.visited=wp.zeros(shape,dtype=wp.int32,device=self.device)
        self.reset()

    def reset(self):
        wp.launch(_reset,dim=(self.e,6),inputs=[
            self.slot_id,self.original,self.terminal,self.hops,
            self.visited,self.capacity,
        ],device=self.device)

    def _array(self,value,dtype):
        return wp.array(np.asarray(value,dtype=np.int64 if dtype==wp.int64
                                   else np.int32),dtype=dtype,device=self.device)

    def _check(self,a,label):
        x=np.asarray(a)
        if x.shape!=(self.e,6) or not np.issubdtype(x.dtype,np.integer):
            raise ValueError(label+" must be integer [E,6]")
        return x

    def attach(self,packet_ids,original_source,ultimate_destination,
               hop_count,visited_mask):
        ids=self._check(packet_ids,"packet_id")
        origin=self._check(original_source,"original_source")
        terminal=self._check(ultimate_destination,"ultimate_destination")
        hops=self._check(hop_count,"hop_count")
        seen=self._check(visited_mask,"visited_mask")
        active=ids>=0
        if np.any(ids<-1):
            raise ValueError("invalid packet identifier")
        if np.any((terminal[active]<0)|(terminal[active]>=6)):
            raise ValueError("ultimate destination must be 0..5")
        if np.any((origin[active]<0)|(origin[active]>=6)):
            raise ValueError("original source must be 0..5")
        if np.any(origin[active]==terminal[active]):
            raise ValueError("destination cannot equal application origin")
        if np.any((hops[active]<0)|(hops[active]>=6)):
            raise ValueError("hop count must be in [0,5]")
        if np.any((seen[active]<1)|(seen[active]>63)):
            raise ValueError("invalid visited bitmask")
        holder=np.broadcast_to(np.arange(6)[None,:],(self.e,6))
        if np.any(active&((seen & (1<<holder))==0)):
            raise ValueError("visited must contain current holder UAV")
        if np.any(active&((seen & (1<<np.maximum(origin,0)))==0)):
            raise ValueError("visited must include original source")
        # A single hop can visit exactly one additional unique UAV.
        # Reject forged path masks before these bytes enter a Warp kernel.
        bit_count=np.zeros_like(seen,dtype=np.int32)
        for bit in range(6):
            bit_count+=(seen>>bit)&1
        if np.any(active&(bit_count!=hops+1)):
            raise ValueError("visited bitcount must equal hop_count+1")
        out=wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
        wp.launch(_attach,dim=(self.e,6),inputs=[
            self._array(ids,wp.int64),self._array(origin,wp.int32),
            self._array(terminal,wp.int32),self._array(hops,wp.int32),
            self._array(seen,wp.int32),
            self.q.state,self.q.packet_id,
            self.slot_id,self.original,self.terminal,self.hops,
            self.visited,out,self.capacity,
        ],device=self.device)
        return out.numpy()

    def read_dispatched(self,ids):
        source=self._check(ids,"sent_id")
        out=[wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
             for _ in range(4)]
        wp.launch(_read_dispatch,dim=(self.e,6),inputs=[
            self._array(source,wp.int64),self.q.state,self.q.packet_id,
            self.slot_id,self.original,self.terminal,self.hops,
            self.visited,*out,self.capacity,
        ],device=self.device)
        return {
            "original_source":out[0].numpy(),
            "ultimate_destination":out[1].numpy(),
            "hop_count":out[2].numpy(),
            "visited_mask":out[3].numpy(),
        }

    def classify_receipt(self,packet_ids,mac_receiver,data_received):
        ids=self._check(packet_ids,"packet_id")
        rx=self._check(mac_receiver,"MAC receiver")
        physically_received=self._check(data_received,"physical DATA")
        if np.any((physically_received!=0)&(physically_received!=1)):
            raise ValueError("physical DATA flags must be binary")
        active=(ids>=0)&(physically_received==1)
        if np.any((rx[active]<0)|(rx[active]>=6)):
            raise ValueError("invalid physical receiver")
        outputs=[wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
                 for _ in range(8)]
        wp.launch(_classify,dim=(self.e,6),inputs=[
            self._array(ids,wp.int64),
            self._array(rx,wp.int32),
            self._array(physically_received,wp.int32),
            self.q.state,self.q.packet_id,
            self.slot_id,self.original,self.terminal,self.hops,
            self.visited,*outputs,self.capacity,
        ],device=self.device)
        names=(
            "relay","final","loop_rejected","hop_budget_exceeded",
            "original_source","ultimate_destination","hop_count",
            "visited_mask",
        )
        return dict(zip(names,(x.numpy() for x in outputs)))
