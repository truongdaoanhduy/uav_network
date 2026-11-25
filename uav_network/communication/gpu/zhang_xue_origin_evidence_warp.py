"""Device-resident per-receiver, per-origin DS evidence with MAX-plus-SUM.

No truth is accepted here. The only inputs are local sensor counts and
origin-labelled evidence entries of packets that actually reached receiver.
All merges run on native Warp arrays, including on CUDA when available.
Duplicate/relay retransmissions use per-origin atomic max, never sum twice.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from uav_network.environment.gpu.zhang_xue_map_warp import _ds_from_counts

N_UAV=6
N_CELL=400
N_ENTRY=62


@wp.kernel
def _copy_own(
    local_pos:wp.array(dtype=wp.int32,ndim=3),
    local_neg:wp.array(dtype=wp.int32,ndim=3),
    pos:wp.array(dtype=wp.int32,ndim=4),
    neg:wp.array(dtype=wp.int32,ndim=4),
):
    e,u,c=wp.tid()
    # Only own locally sensed counts may be written here.
    pos[e,u,u,c]=local_pos[e,u,c]
    neg[e,u,u,c]=local_neg[e,u,c]


@wp.kernel
def _merge_delivered(
    receivers:wp.array(dtype=wp.int32,ndim=2),
    origin:wp.array(dtype=wp.int32,ndim=3),
    cell:wp.array(dtype=wp.int32,ndim=3),
    count_pos:wp.array(dtype=wp.int32,ndim=3),
    count_neg:wp.array(dtype=wp.int32,ndim=3),
    num_entries:wp.array(dtype=wp.int32,ndim=2),
    received:wp.array(dtype=wp.int32,ndim=2),
    pos:wp.array(dtype=wp.int32,ndim=4),
    neg:wp.array(dtype=wp.int32,ndim=4),
):
    e,sender,idx=wp.tid()
    if received[e,sender]==0 or idx>=num_entries[e,sender]:
        return
    dest=receivers[e,sender]
    src=origin[e,sender,idx]
    cid=cell[e,sender,idx]
    # Validation performed before copying to device.
    wp.atomic_max(pos,e,dest,src,cid,count_pos[e,sender,idx])
    wp.atomic_max(neg,e,dest,src,cid,count_neg[e,sender,idx])


@wp.kernel
def _belief_from_local_origins(
    pos:wp.array(dtype=wp.int32,ndim=4),
    neg:wp.array(dtype=wp.int32,ndim=4),
    mf:wp.array(dtype=wp.float32,ndim=3),
    me:wp.array(dtype=wp.float32,ndim=3),
    mu:wp.array(dtype=wp.float32,ndim=3),
):
    e,u,c=wp.tid()
    total_pos=wp.int32(0)
    total_neg=wp.int32(0)
    for origin in range(N_UAV):
        total_pos+=pos[e,u,origin,c]
        total_neg+=neg[e,u,origin,c]
    m=_ds_from_counts(total_pos,total_neg)
    mf[e,u,c]=m[0]
    mu[e,u,c]=m[1]
    me[e,u,c]=wp.max(wp.float32(0.),wp.float32(1.)-m[0]-m[1])


@wp.kernel
def _init_unknown(
    mf:wp.array(dtype=wp.float32,ndim=3),
    me:wp.array(dtype=wp.float32,ndim=3),
    mu:wp.array(dtype=wp.float32,ndim=3),
):
    e,u,c=wp.tid()
    mf[e,u,c]=0.
    me[e,u,c]=0.
    mu[e,u,c]=1.


class ZhangXueWarpOriginBelief:
    """Origin-aware local evidence [E, receiver, origin, 400] on Warp.

    The actor/critic must NEVER inspect remote evidence outside the local
    receiver dimension. No target truth array exists in this module.
    """
    def __init__(self,num_envs:int,*,device="cpu"):
        if not isinstance(num_envs,int) or num_envs<1:
            raise ValueError("num_envs must be positive integer")
        self.num_envs=num_envs
        self.device=device
        wp.init()
        dims=(num_envs,N_UAV,N_UAV,N_CELL)
        obs_dims=(num_envs,N_UAV,N_CELL)
        self.pos=wp.zeros(dims,dtype=wp.int32,device=device)
        self.neg=wp.zeros(dims,dtype=wp.int32,device=device)
        self.mass_target=wp.zeros(obs_dims,dtype=wp.float32,device=device)
        self.mass_empty=wp.zeros(obs_dims,dtype=wp.float32,device=device)
        self.unknown=wp.zeros(obs_dims,dtype=wp.float32,device=device)
        self.reset()

    def reset(self):
        self.pos.zero_()
        self.neg.zero_()
        wp.launch(_init_unknown,dim=(self.num_envs,N_UAV,N_CELL),
                  inputs=[self.mass_target,self.mass_empty,self.unknown],
                  device=self.device)

    def _belief(self):
        wp.launch(
            _belief_from_local_origins,
            dim=(self.num_envs,N_UAV,N_CELL),
            inputs=[self.pos,self.neg,self.mass_target,
                    self.mass_empty,self.unknown],
            device=self.device,
        )

    def set_own_evidence(self,positive,negative):
        """The local sensors' cumulative counts, or Warp arrays directly.

        Sensor map has no packet barrier for *own* evidence. It has one
        for every other (origin, receiver) pair.
        """
        shape=(self.num_envs,N_UAV,N_CELL)
        if isinstance(positive,wp.array) and isinstance(negative,wp.array):
            if positive.shape!=shape or negative.shape!=shape:
                raise ValueError("own evidence requires [E,6,400]")
            p,n=positive,negative
        else:
            pval=np.asarray(positive)
            nval=np.asarray(negative)
            if pval.shape!=shape or nval.shape!=shape:
                raise ValueError("own evidence requires [E,6,400]")
            for x in (pval,nval):
                if (not np.issubdtype(x.dtype,np.integer) or
                    np.any(x<0) or np.any(x>65535)):
                    raise ValueError("sensor counts must be integers in [0,65535]")
            p=wp.array(pval.astype(np.int32),dtype=wp.int32,device=self.device)
            n=wp.array(nval.astype(np.int32),dtype=wp.int32,device=self.device)
        wp.launch(_copy_own,dim=shape,
                  inputs=[p,n,self.pos,self.neg],device=self.device)
        self._belief()

    def merge_delivered(self,receivers,origin,cell,positive,negative,
                        num_entries,received):
        """Apply only valid, successfully *received* packet entries.

        One packet per sender, up to 62 origin-tagged cumulative snapshots.
        The caller must authenticate packet provenance and router identity;
        this kernel never derives truth or fabricates packet receptions.
        """
        e=self.num_envs
        fields=[np.asarray(a) for a in (receivers,origin,cell,positive,negative,
                                         num_entries,received)]
        two=(e,N_UAV)
        three=(e,N_UAV,N_ENTRY)
        if any(x.shape!=two for x in (fields[0],fields[5],fields[6])):
            raise ValueError("receivers/length/received must be [E,6]")
        if any(x.shape!=three for x in fields[1:5]):
            raise ValueError("entries must be [E,6,62]")
        if any(not np.issubdtype(x.dtype,np.integer) for x in fields):
            raise ValueError("payload fields must be integer-valued")
        rx,orig,cid,p,n,counts,good=fields
        if (np.any((counts<0)|(counts>N_ENTRY)) or
            np.any((good!=0)&(good!=1))):
            raise ValueError("entry length or received flag invalid")
        sending=good==1
        if np.any((rx[sending]<0)|(rx[sending]>=N_UAV)):
            raise ValueError("invalid receiver")
        entry_active=(np.arange(N_ENTRY)[None,None,:]<counts[:,:,None])&sending[:,:,None]
        if np.any((orig[entry_active]<0)|(orig[entry_active]>=N_UAV)):
            raise ValueError("invalid packet origin")
        if np.any((cid[entry_active]<0)|(cid[entry_active]>=N_CELL)):
            raise ValueError("invalid cell index")
        if (np.any((p[entry_active]<0)|(p[entry_active]>65535)) or
            np.any((n[entry_active]<0)|(n[entry_active]>65535))):
            raise ValueError("invalid count range [0,65535]")
        def w(a):return wp.array(a.astype(np.int32),dtype=wp.int32,device=self.device)
        wp.launch(_merge_delivered,dim=three,inputs=[
            w(rx),w(orig),w(cid),w(p),w(n),w(counts),w(good),
            self.pos,self.neg,
        ],device=self.device)
        self._belief()
        return int(np.count_nonzero(sending&(counts>0)))

    def read(self):
        return {
            "pos":self.pos.numpy(),
            "neg":self.neg.numpy(),
            "mass_target":self.mass_target.numpy(),
            "mass_empty":self.mass_empty.numpy(),
            "unknown":self.unknown.numpy(),
        }
