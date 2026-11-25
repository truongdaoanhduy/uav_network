"""Device-resident, per-origin/per-peer application E2E ACK watermarks.

PHY/MAC ACK does NOT mark evidence synchronized. Only a matching unique
application-level ACK permits advancing counts from its immutable snapshot.
No target truth, other UAV's undisclosed measurements or global routing
information is available here.

Supports independent [E,6] source senders and five remote peers/source.
Each source-peer pair may have at most one outstanding application sequence;
a lost E2E ACK can be abandoned after TTL, leaving evidence dirty for
retransmission. ACK duplicates never double reward or advance watermark.
"""
from __future__ import annotations
import numpy as np
import warp as wp

N=6
C=400
K=62


@wp.kernel
def _reset_pending(
    pending_id:wp.array(dtype=wp.int64,ndim=3),
    pending_count:wp.array(dtype=wp.int32,ndim=3),
    successful:wp.array(dtype=wp.int32,ndim=2),
    stale:wp.array(dtype=wp.int32,ndim=2),
):
    e,u,peer=wp.tid()
    pending_id[e,u,peer]=wp.int64(-1)
    pending_count[e,u,peer]=0
    if peer==0:
        successful[e,u]=0
        stale[e,u]=0


@wp.kernel
def _dirty(
    own_p:wp.array(dtype=wp.int32,ndim=3),
    own_n:wp.array(dtype=wp.int32,ndim=3),
    peers:wp.array(dtype=wp.int32,ndim=2),
    acked_p:wp.array(dtype=wp.int32,ndim=4),
    acked_n:wp.array(dtype=wp.int32,ndim=4),
    pending_id:wp.array(dtype=wp.int64,ndim=3),
    cells:wp.array(dtype=wp.int32,ndim=3),
    selected_p:wp.array(dtype=wp.int32,ndim=3),
    selected_n:wp.array(dtype=wp.int32,ndim=3),
    selected_count:wp.array(dtype=wp.int32,ndim=2),
):
    e,u=wp.tid()
    selected_count[e,u]=0
    for k in range(K):
        cells[e,u,k]=-1
        selected_p[e,u,k]=0
        selected_n[e,u,k]=0
    peer=peers[e,u]
    if peer<0 or peer==u:
        return
    if pending_id[e,u,peer]>=wp.int64(0):
        return
    count=wp.int32(0)
    for cid in range(C):
        if (own_p[e,u,cid]>acked_p[e,u,peer,cid] or
            own_n[e,u,cid]>acked_n[e,u,peer,cid]):
            if count<K:
                cells[e,u,count]=cid
                selected_p[e,u,count]=own_p[e,u,cid]
                selected_n[e,u,count]=own_n[e,u,cid]
                count+=1
            else:
                break
    selected_count[e,u]=count


@wp.kernel
def _register(
    packet_id:wp.array(dtype=wp.int64,ndim=2),
    peers:wp.array(dtype=wp.int32,ndim=2),
    cells:wp.array(dtype=wp.int32,ndim=3),
    p:wp.array(dtype=wp.int32,ndim=3),
    n:wp.array(dtype=wp.int32,ndim=3),
    count:wp.array(dtype=wp.int32,ndim=2),
    pending_id:wp.array(dtype=wp.int64,ndim=3),
    pending_count:wp.array(dtype=wp.int32,ndim=3),
    pending_cell:wp.array(dtype=wp.int32,ndim=4),
    pending_p:wp.array(dtype=wp.int32,ndim=4),
    pending_n:wp.array(dtype=wp.int32,ndim=4),
    status:wp.array(dtype=wp.int32,ndim=2),
):
    e,u=wp.tid()
    status[e,u]=0
    if packet_id[e,u]<wp.int64(0):
        return
    peer=peers[e,u]
    if peer<0 or peer==u or count[e,u]<1:
        status[e,u]=-1
        return
    if pending_id[e,u,peer]>=wp.int64(0):
        status[e,u]=-2
        return
    pending_id[e,u,peer]=packet_id[e,u]
    pending_count[e,u,peer]=count[e,u]
    for k in range(K):
        pending_cell[e,u,peer,k]=cells[e,u,k]
        pending_p[e,u,peer,k]=p[e,u,k]
        pending_n[e,u,peer,k]=n[e,u,k]
    status[e,u]=1


@wp.kernel
def _ack(
    packet_id:wp.array(dtype=wp.int64,ndim=2),
    peers:wp.array(dtype=wp.int32,ndim=2),
    physical_arrival:wp.array(dtype=wp.int32,ndim=2),
    pending_id:wp.array(dtype=wp.int64,ndim=3),
    pending_count:wp.array(dtype=wp.int32,ndim=3),
    pending_cell:wp.array(dtype=wp.int32,ndim=4),
    pending_p:wp.array(dtype=wp.int32,ndim=4),
    pending_n:wp.array(dtype=wp.int32,ndim=4),
    acked_p:wp.array(dtype=wp.int32,ndim=4),
    acked_n:wp.array(dtype=wp.int32,ndim=4),
    successful:wp.array(dtype=wp.int32,ndim=2),
    stale:wp.array(dtype=wp.int32,ndim=2),
    status:wp.array(dtype=wp.int32,ndim=2),
):
    e,u=wp.tid()
    status[e,u]=0
    if physical_arrival[e,u]==0 or packet_id[e,u]<wp.int64(0):
        return
    peer=peers[e,u]
    if peer<0 or peer==u:
        status[e,u]=-1
        stale[e,u]+=1
        return
    if pending_id[e,u,peer]!=packet_id[e,u]:
        status[e,u]=-1
        stale[e,u]+=1
        return
    for k in range(K):
        if k<pending_count[e,u,peer]:
            cid=pending_cell[e,u,peer,k]
            acked_p[e,u,peer,cid]=wp.max(
                acked_p[e,u,peer,cid],pending_p[e,u,peer,k]
            )
            acked_n[e,u,peer,cid]=wp.max(
                acked_n[e,u,peer,cid],pending_n[e,u,peer,k]
            )
    pending_id[e,u,peer]=wp.int64(-1)
    pending_count[e,u,peer]=0
    successful[e,u]+=1
    status[e,u]=1


@wp.kernel
def _abandon(
    ids:wp.array(dtype=wp.int64,ndim=2),
    peers:wp.array(dtype=wp.int32,ndim=2),
    pending_id:wp.array(dtype=wp.int64,ndim=3),
    pending_count:wp.array(dtype=wp.int32,ndim=3),
    status:wp.array(dtype=wp.int32,ndim=2),
):
    e,u=wp.tid()
    status[e,u]=0
    if ids[e,u]<wp.int64(0):
        return
    peer=peers[e,u]
    if peer<0 or peer==u or pending_id[e,u,peer]!=ids[e,u]:
        status[e,u]=-1
        return
    pending_id[e,u,peer]=wp.int64(-1)
    pending_count[e,u,peer]=0
    status[e,u]=1


class ZhangXueWarpAckWatermarks:
    """Stable source-peer evidence watermark advanced only by real E2E ACK."""
    def __init__(self,num_envs:int,*,device="cpu"):
        if not isinstance(num_envs,int) or num_envs<1:
            raise ValueError("positive integer num_envs needed")
        self.num_envs=num_envs
        self.device=device
        wp.init()
        e=num_envs
        p4=(e,N,N,C)
        pend=(e,N,N,K)
        triple=(e,N,N)
        pair=(e,N)
        self.acked_p=wp.zeros(p4,dtype=wp.int32,device=device)
        self.acked_n=wp.zeros(p4,dtype=wp.int32,device=device)
        self.pending_id=wp.zeros(triple,dtype=wp.int64,device=device)
        self.pending_count=wp.zeros(triple,dtype=wp.int32,device=device)
        self.pending_cell=wp.zeros(pend,dtype=wp.int32,device=device)
        self.pending_p=wp.zeros(pend,dtype=wp.int32,device=device)
        self.pending_n=wp.zeros(pend,dtype=wp.int32,device=device)
        self.successful=wp.zeros(pair,dtype=wp.int32,device=device)
        self.stale=wp.zeros(pair,dtype=wp.int32,device=device)
        self.reset()

    def reset(self):
        self.acked_p.zero_()
        self.acked_n.zero_()
        wp.launch(_reset_pending,dim=(self.num_envs,N,N),inputs=[
            self.pending_id,self.pending_count,self.successful,self.stale
        ],device=self.device)

    def _two(self,values,field):
        v=np.asarray(values)
        if v.shape!=(self.num_envs,N) or not np.issubdtype(v.dtype,np.integer):
            raise ValueError(field+" must be integer [E,6]")
        return v

    def _wa(self,values,dtype=wp.int32):
        return wp.array(values.astype(np.int64 if dtype==wp.int64 else np.int32),
                        dtype=dtype,device=self.device)

    def _check_peer(self,peer,active):
        if (np.any((peer[active]<0)|(peer[active]>=N)) or
            np.any(active&(peer==np.arange(N)[None,:]))):
            raise ValueError("invalid peer or self-target")

    def dirty(self,positive,negative,peer):
        pos=np.asarray(positive)
        neg=np.asarray(negative)
        p=self._two(peer,"peer")
        if (pos.shape!=(self.num_envs,N,C) or neg.shape!=(self.num_envs,N,C)):
            raise ValueError("own sensor counts must be [E,6,400]")
        for a in (pos,neg):
            if (not np.issubdtype(a.dtype,np.integer) or
                np.any(a<0) or np.any(a>65535)):
                raise ValueError("sensor counts must be in 0..65535")
        self._check_peer(p,p>=0)
        cells=wp.zeros((self.num_envs,N,K),dtype=wp.int32,device=self.device)
        selected_p=wp.zeros((self.num_envs,N,K),dtype=wp.int32,device=self.device)
        selected_n=wp.zeros((self.num_envs,N,K),dtype=wp.int32,device=self.device)
        count=wp.zeros((self.num_envs,N),dtype=wp.int32,device=self.device)
        wp.launch(_dirty,dim=(self.num_envs,N),inputs=[
            self._wa(pos),self._wa(neg),self._wa(p),
            self.acked_p,self.acked_n,self.pending_id,
            cells,selected_p,selected_n,count
        ],device=self.device)
        return {
            "cell":cells.numpy(),"positive":selected_p.numpy(),
            "negative":selected_n.numpy(),"count":count.numpy(),
        }

    def register(self,packet_ids,peers,dirty):
        ids=self._two(packet_ids,"packet_ids")
        p=self._two(peers,"peer")
        active=ids>=0
        if np.any(ids<-1):
            raise ValueError("invalid packet_id")
        self._check_peer(p,active)
        cols=[np.asarray(dirty[k]) for k in ("cell","positive","negative")]
        counts=self._two(dirty["count"],"count")
        if (any(a.shape!=(self.num_envs,N,K) for a in cols) or
            any(not np.issubdtype(a.dtype,np.integer) for a in cols)):
            raise ValueError("dirty entries must be integer [E,6,62]")
        if np.any((counts<0)|(counts>K)):
            raise ValueError("dirty count outside [0,62]")
        valid=(np.arange(K)[None,None,:]<counts[:,:,None])&active[:,:,None]
        if np.any((cols[0][valid]<0)|(cols[0][valid]>=C)):
            raise ValueError("invalid dirty cell")
        if any(np.any((a[valid]<0)|(a[valid]>65535)) for a in cols[1:]):
            raise ValueError("invalid dirty positive/negative")
        out=wp.zeros((self.num_envs,N),dtype=wp.int32,device=self.device)
        wp.launch(_register,dim=(self.num_envs,N),inputs=[
            self._wa(ids,wp.int64),self._wa(p),*[self._wa(a) for a in cols],
            self._wa(counts),
            self.pending_id,self.pending_count,self.pending_cell,
            self.pending_p,self.pending_n,out
        ],device=self.device)
        return out.numpy()

    def acknowledge(self,packet_ids,peers,physical_arrival):
        ids=self._two(packet_ids,"packet_ids")
        p=self._two(peers,"peer")
        flags=self._two(physical_arrival,"physical_arrival")
        if np.any((flags!=0)&(flags!=1)):
            raise ValueError("E2E arrival flags must be binary")
        self._check_peer(p,(flags==1)&(ids>=0))
        out=wp.zeros((self.num_envs,N),dtype=wp.int32,device=self.device)
        wp.launch(_ack,dim=(self.num_envs,N),inputs=[
            self._wa(ids,wp.int64),self._wa(p),self._wa(flags),
            self.pending_id,self.pending_count,self.pending_cell,
            self.pending_p,self.pending_n,self.acked_p,self.acked_n,
            self.successful,self.stale,out,
        ],device=self.device)
        return out.numpy()

    def abandon(self,packet_ids,peers):
        ids=self._two(packet_ids,"packet_ids")
        p=self._two(peers,"peer")
        out=wp.zeros((self.num_envs,N),dtype=wp.int32,device=self.device)
        wp.launch(_abandon,dim=(self.num_envs,N),inputs=[
            self._wa(ids,wp.int64),self._wa(p),
            self.pending_id,self.pending_count,out
        ],device=self.device)
        return out.numpy()

    def read(self):
        return {
            "acked_positive":self.acked_p.numpy(),
            "acked_negative":self.acked_n.numpy(),
            "pending_id":self.pending_id.numpy(),
            "successful_e2e_acks":self.successful.numpy(),
            "stale_ack_count":self.stale.numpy(),
        }
