"""Immutable application evidence sidecar for ZhangXueWarpPacketQueue.

Persistent on device, with slots keyed by packet ID. A queued payload stays
frozen while a source goes offline; stale slots cannot attach to new packets.
This is origin-tagged DATA (62 entries × 8B) not global target truth. Payload
not yet routed and not yet copied to downstream relay queue.
"""
from __future__ import annotations
import numpy as np
import warp as wp


@wp.kernel
def _stage(
    ids:wp.array(dtype=wp.int64,ndim=2),
    entry_origin:wp.array(dtype=wp.int32,ndim=3),
    entry_cell:wp.array(dtype=wp.int32,ndim=3),
    entry_pos:wp.array(dtype=wp.int32,ndim=3),
    entry_neg:wp.array(dtype=wp.int32,ndim=3),
    count:wp.array(dtype=wp.int32,ndim=2),
    queue_state:wp.array(dtype=wp.int32,ndim=3),
    queue_id:wp.array(dtype=wp.int64,ndim=3),
    queue_kind:wp.array(dtype=wp.int32,ndim=3),
    queue_frame:wp.array(dtype=wp.int32,ndim=3),
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    slot_origin:wp.array(dtype=wp.int32,ndim=4),
    slot_cell:wp.array(dtype=wp.int32,ndim=4),
    slot_pos:wp.array(dtype=wp.int32,ndim=4),
    slot_neg:wp.array(dtype=wp.int32,ndim=4),
    slot_length:wp.array(dtype=wp.int32,ndim=3),
    result:wp.array(dtype=wp.int32,ndim=2),
    capacity:int,
):
    e,u=wp.tid()
    result[e,u]=0
    pid=ids[e,u]
    if pid<wp.int64(0):
        return
    found=wp.int32(-1)
    for s in range(capacity):
        if queue_state[e,u,s]!=0 and queue_id[e,u,s]==pid:
            found=s
            break
    if found<0:
        result[e,u]=-1
        return
    if slot_id[e,u,found]==pid:
        result[e,u]=-2
        return
    if queue_kind[e,u,found]!=0 or queue_frame[e,u,found]!=70+8*count[e,u]:
        result[e,u]=-3
        return
    slot_id[e,u,found]=pid
    slot_length[e,u,found]=count[e,u]
    for k in range(62):
        if k<count[e,u]:
            slot_origin[e,u,found,k]=entry_origin[e,u,k]
            slot_cell[e,u,found,k]=entry_cell[e,u,k]
            slot_pos[e,u,found,k]=entry_pos[e,u,k]
            slot_neg[e,u,found,k]=entry_neg[e,u,k]
        else:
            slot_origin[e,u,found,k]=-1
            slot_cell[e,u,found,k]=-1
            slot_pos[e,u,found,k]=0
            slot_neg[e,u,found,k]=0
    result[e,u]=1


@wp.kernel
def _reset_payload(
    slot_id:wp.array(dtype=wp.int64,ndim=3),
    slot_length:wp.array(dtype=wp.int32,ndim=3),
    capacity:int,
):
    e,u=wp.tid()
    for s in range(capacity):
        slot_id[e,u,s]=wp.int64(-1)
        slot_length[e,u,s]=0


@wp.kernel
def _select_slot(
    ids:wp.array(dtype=wp.int64,ndim=2),
    q_state:wp.array(dtype=wp.int32,ndim=3),
    q_ids:wp.array(dtype=wp.int64,ndim=3),
    payload_ids:wp.array(dtype=wp.int64,ndim=3),
    payload_count:wp.array(dtype=wp.int32,ndim=3),
    slots:wp.array(dtype=wp.int32,ndim=2),
    valid:wp.array(dtype=wp.int32,ndim=2),
    capacity:int,
):
    e,u=wp.tid()
    slots[e,u]=-1
    valid[e,u]=0
    if ids[e,u]<wp.int64(0):
        return
    for s in range(capacity):
        if q_state[e,u,s]==2 and q_ids[e,u,s]==ids[e,u]:
            if payload_ids[e,u,s]==ids[e,u]:
                slots[e,u]=s
                valid[e,u]=1
            else:
                valid[e,u]=-1
            return
    valid[e,u]=-2


@wp.kernel
def _read_selected(
    slot:wp.array(dtype=wp.int32,ndim=2),
    length:wp.array(dtype=wp.int32,ndim=3),
    origin:wp.array(dtype=wp.int32,ndim=4),
    cell:wp.array(dtype=wp.int32,ndim=4),
    positive:wp.array(dtype=wp.int32,ndim=4),
    negative:wp.array(dtype=wp.int32,ndim=4),
    count_out:wp.array(dtype=wp.int32,ndim=2),
    origin_out:wp.array(dtype=wp.int32,ndim=3),
    cell_out:wp.array(dtype=wp.int32,ndim=3),
    pos_out:wp.array(dtype=wp.int32,ndim=3),
    neg_out:wp.array(dtype=wp.int32,ndim=3),
):
    e,u,k=wp.tid()
    s=slot[e,u]
    if s<0:
        if k==0:
            count_out[e,u]=0
        origin_out[e,u,k]=-1
        cell_out[e,u,k]=-1
        pos_out[e,u,k]=0
        neg_out[e,u,k]=0
        return
    if k==0:
        count_out[e,u]=length[e,u,s]
    origin_out[e,u,k]=origin[e,u,s,k]
    cell_out[e,u,k]=cell[e,u,s,k]
    pos_out[e,u,k]=positive[e,u,s,k]
    neg_out[e,u,k]=negative[e,u,s,k]


@wp.kernel
def _merge_queue_payload(
    receiver:wp.array(dtype=wp.int32,ndim=2),
    arrived:wp.array(dtype=wp.int32,ndim=2),
    slots:wp.array(dtype=wp.int32,ndim=2),
    count:wp.array(dtype=wp.int32,ndim=3),
    origin:wp.array(dtype=wp.int32,ndim=4),
    cell:wp.array(dtype=wp.int32,ndim=4),
    positive:wp.array(dtype=wp.int32,ndim=4),
    negative:wp.array(dtype=wp.int32,ndim=4),
    belief_pos:wp.array(dtype=wp.int32,ndim=4),
    belief_neg:wp.array(dtype=wp.int32,ndim=4),
):
    e,u,k=wp.tid()
    s=slots[e,u]
    if arrived[e,u]==0 or s<0 or k>=count[e,u,s]:
        return
    dst=receiver[e,u]
    src=origin[e,u,s,k]
    cid=cell[e,u,s,k]
    wp.atomic_max(belief_pos,e,dst,src,cid,positive[e,u,s,k])
    wp.atomic_max(belief_neg,e,dst,src,cid,negative[e,u,s,k])


class ZhangXueWarpPayloadStore:
    """Persistent immutable, per-queue-slot DATA entry arrays on Warp."""
    def __init__(self,queue):
        self.queue=queue
        self.e=queue.num_envs
        self.capacity=queue.capacity
        self.device=queue.device
        q=(self.e,6,queue.capacity)
        shape=(*q,62)
        self.slot_id=wp.zeros(q,dtype=wp.int64,device=self.device)
        self.slot_length=wp.zeros(q,dtype=wp.int32,device=self.device)
        def alloc():return wp.zeros(shape,dtype=wp.int32,device=self.device)
        self.origin=alloc()
        self.cell=alloc()
        self.pos=alloc()
        self.neg=alloc()
        self.reset()

    def reset(self):
        wp.launch(_reset_payload,dim=(self.e,6),
                  inputs=[self.slot_id,self.slot_length,self.capacity],
                  device=self.device)

    def _fields(self,ids,origin,cell,positive,negative,length):
        arrays=[np.asarray(x) for x in (ids,origin,cell,positive,negative,length)]
        if arrays[0].shape!=(self.e,6) or arrays[-1].shape!=(self.e,6):
            raise ValueError("payload IDs/count shape must be [E,6]")
        if any(x.shape!=(self.e,6,62) for x in arrays[1:5]):
            raise ValueError("payload entries must be [E,6,62]")
        if any(not np.issubdtype(x.dtype,np.integer) for x in arrays):
            raise ValueError("payload must be integer")
        ids,org,cid,p,n,cnt=arrays
        if np.any((cnt<0)|(cnt>62)):
            raise ValueError("payload entry count must be 0..62")
        active=(np.arange(62)[None,None,:]<cnt[:,:,None])&(ids[:,:,None]>=0)
        if np.any((org[active]<0)|(org[active]>=6)):
            raise ValueError("invalid payload origin")
        if np.any((cid[active]<0)|(cid[active]>=400)):
            raise ValueError("invalid payload cell")
        if (np.any((p[active]<0)|(p[active]>65535)) or
            np.any((n[active]<0)|(n[active]>65535))):
            raise ValueError("invalid evidence count")
        return arrays

    def _wa(self,x,dtype):
        return wp.array(np.asarray(x).astype(np.int32 if dtype==wp.int32 else np.int64),
                        dtype=dtype,device=self.device)

    def attach(self,ids,origin,cell,positive,negative,length):
        ids,org,cid,p,n,cnt=self._fields(ids,origin,cell,positive,negative,length)
        out=wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
        wp.launch(_stage,dim=(self.e,6),inputs=[
            self._wa(ids,wp.int64),self._wa(org,wp.int32),
            self._wa(cid,wp.int32),self._wa(p,wp.int32),
            self._wa(n,wp.int32),self._wa(cnt,wp.int32),
            self.queue.state,self.queue.packet_id,self.queue.kind,
            self.queue.size_bytes,self.slot_id,
            self.origin,self.cell,self.pos,self.neg,self.slot_length,
            out,self.capacity,
        ],device=self.device)
        return out.numpy()

    def _slots(self,ids):
        arr=np.asarray(ids)
        if arr.shape!=(self.e,6) or not np.issubdtype(arr.dtype,np.integer):
            raise ValueError("packet IDs must be [E,6] integer")
        slots=wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
        valid=wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
        wp.launch(_select_slot,dim=(self.e,6),inputs=[
            self._wa(arr,wp.int64),self.queue.state,
            self.queue.packet_id,self.slot_id,self.slot_length,
            slots,valid,self.capacity,
        ],device=self.device)
        return slots,valid.numpy()

    def read_dispatched(self,packet_ids):
        slots,_=self._slots(packet_ids)
        count=wp.zeros((self.e,6),dtype=wp.int32,device=self.device)
        o=[wp.zeros((self.e,6,62),dtype=wp.int32,device=self.device)
           for _ in range(4)]
        wp.launch(_read_selected,dim=(self.e,6,62),inputs=[
            slots,self.slot_length,self.origin,self.cell,self.pos,self.neg,
            count,*o,
        ],device=self.device)
        return {
            "entry_count":count.numpy(),
            "origin":o[0].numpy(),
            "cell":o[1].numpy(),
            "positive":o[2].numpy(),
            "negative":o[3].numpy(),
        }

    def merge_receipts(self,packet_ids,receivers,arrived,belief):
        ids=np.asarray(packet_ids)
        rx=np.asarray(receivers)
        success=np.asarray(arrived)
        if (ids.shape!=(self.e,6) or rx.shape!=(self.e,6)
            or success.shape!=(self.e,6)):
            raise ValueError("MAC receipt inputs must be [E,6]")
        if (not np.issubdtype(ids.dtype,np.integer) or
            not np.issubdtype(rx.dtype,np.integer) or
            not np.issubdtype(success.dtype,np.integer)):
            raise ValueError("MAC receipt fields must be integers")
        if np.any((success!=0)&(success!=1)):
            raise ValueError("MAC DATA receipt flags must be binary")
        if np.any((success!=0)&((rx<0)|(rx>=6))):
            raise ValueError("invalid receiver")
        if belief.num_envs!=self.e or belief.device!=self.device:
            raise ValueError("belief must share world/device")
        slots,valid=self._slots(ids)
        if np.any((success==1)&(valid!=1)):
            raise ValueError("missing or stale packet payload on received DATA")
        wp.launch(_merge_queue_payload,dim=(self.e,6,62),inputs=[
            self._wa(rx,wp.int32),self._wa(success,wp.int32),
            slots,self.slot_length,self.origin,self.cell,self.pos,self.neg,
            belief.pos,belief.neg,
        ],device=self.device)
        belief._belief()
        return int(np.count_nonzero(success))
