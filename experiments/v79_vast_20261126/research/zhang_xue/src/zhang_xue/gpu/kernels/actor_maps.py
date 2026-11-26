"""Warp-parallel packet-local DS/AoI spatial Actor map projection (Gate 3 subcomponent).

This module accelerates 5x10x10 map projection for E independent 7-radio worlds.
It DOES NOT implement packet MAC/PHY on CUDA. Full env CPU-Warp parity remains BLOCKED.
"""
from __future__ import annotations
import math

try:
    import warp as wp
    from zhang_xue.gpu.kernels.scaffold import counts_to_canonical_ds_wp
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

if HAS_WARP:
    @wp.kernel
    def _actor_maps_7radio_kernel(
        ledger_pos: wp.array4d(dtype=wp.uint16),
        ledger_neg: wp.array4d(dtype=wp.uint16),
        ledger_ts: wp.array4d(dtype=wp.int32),
        rx_delay: wp.array4d(dtype=wp.float32),
        pos_m: wp.array3d(dtype=wp.float32),
        episode_step: wp.array(dtype=wp.int32),
        out: wp.array3d(dtype=wp.float32),
        logone_mf: float,logone_me: float,eta: float,age_ref: float,
        fusion_mode: int, uq88: int, semantic: int,
    ):
        """1 Warp thread computes pooled 2x2 cells for 1 actor in 1 env."""
        idx=wp.tid()
        e=idx//600
        a=(idx//100)%6
        binid=idx%100
        cy=binid//10
        cx=binid%10
        sm_t=0.0;sm_e=0.0;sm_u=0.0;sm_age=0.0
        t_max=0.0;f_min=1.0;age_max=0.0
        for oy in range(2):
            for ox in range(2):
                cid=(2*cy+oy)*20+2*cx+ox
                latest=-1
                tt=0.0;ee=0.0;uu=1.0
                total_p=0;total_n=0
                max_p=0;max_n=0
                for src in range(6):
                    p=wp.int32(ledger_pos[e,a,src,cid])
                    n=wp.int32(ledger_neg[e,a,src,cid])
                    if p+n>0:
                        latest=wp.max(latest,ledger_ts[e,a,src,cid])
                    if uq88==0:
                        total_p += p
                        total_n += n
                        max_p=wp.max(max_p,p)
                        max_n=wp.max(max_n,n)
                        if fusion_mode==1 and p+n>0:
                            src_m=counts_to_canonical_ds_wp(p,n,logone_mf,logone_me)
                            age=wp.max(0.0,rx_delay[e,a,src,cid])
                            weight=wp.exp(-eta*age)
                            st=src_m[0]*weight
                            se=src_m[1]*weight
                            su=1.0-weight*(src_m[0]+src_m[1])
                            conflict=tt*se+ee*st
                            den=wp.max(1e-10,1.0-conflict)
                            next_t=(tt*st+tt*su+uu*st)/den
                            next_e=(ee*se+ee*su+uu*se)/den
                            next_u=(uu*su)/den
                            tt=next_t;ee=next_e;uu=next_u
                if uq88!=0:
                    # Transport stores effective counts in UQ8.8; interpreting
                    # raw integer 256 as 256 observations incorrectly collapses DS.
                    ep=float(0.0);en=float(0.0)
                    for src in range(6):
                        p=wp.int32(ledger_pos[e,a,src,cid])
                        n=wp.int32(ledger_neg[e,a,src,cid])
                        if p+n==0 or ledger_ts[e,a,src,cid]<0:
                            continue
                        weight=1.0
                        if fusion_mode!=2:
                            age=wp.max(0.0,float(episode_step[e]-ledger_ts[e,a,src,cid]))
                            weight=wp.exp(-eta*age)
                        sp=float(p)*(1.0/256.0)*weight
                        sn=float(n)*(1.0/256.0)*weight
                        if fusion_mode==2:
                            ep=wp.max(ep,sp)
                            en=wp.max(en,sn)
                        else:
                            ep+=sp
                            en+=sn
                    if ep+en>0.0:
                        from_ds_a=ep*logone_mf
                        from_ds_b=en*logone_me
                        shift=wp.max(from_ds_a,from_ds_b)
                        pa=wp.exp(from_ds_a-shift)
                        pb=wp.exp(from_ds_b-shift)
                        pu=wp.exp(wp.min(from_ds_a,from_ds_b))
                        t_num=pb*(1.0-wp.exp(from_ds_a))
                        f_num=pa*(1.0-wp.exp(from_ds_b))
                        z=wp.max(1.0e-20,t_num+f_num+pu)
                        tt=t_num/z;ee=f_num/z;uu=pu/z
                elif fusion_mode==0 and total_p+total_n>0:
                    r=counts_to_canonical_ds_wp(total_p,total_n,logone_mf,logone_me)
                    tt=r[0];ee=r[1];uu=r[2]
                elif fusion_mode==2 and max_p+max_n>0:
                    r=counts_to_canonical_ds_wp(max_p,max_n,logone_mf,logone_me)
                    tt=r[0];ee=r[1];uu=r[2]
                sm_t+=tt*0.25;sm_e+=ee*0.25;sm_u+=uu*0.25
                t_max=wp.max(t_max,tt)
                f_min=wp.min(f_min,ee)
                if latest>=0:
                    age=wp.clamp((wp.float32(episode_step[e])-wp.float32(latest))/age_ref,0.0,1.0)
                else:
                    age=1.0
                sm_age+=age*0.25
                age_max=wp.max(age_max,age)
        out[e,a,binid]=t_max if semantic!=0 else sm_t
        out[e,a,100+binid]=f_min if semantic!=0 else sm_e
        out[e,a,200+binid]=sm_u
        out[e,a,300+binid]=age_max if semantic!=0 else sm_age
        px=wp.int32(wp.clamp(pos_m[e,a,0]/50.0,0.0,19.0))//2
        py=wp.int32(wp.clamp(pos_m[e,a,1]/50.0,0.0,19.0))//2
        own=0.0
        if px==cx and py==cy:
            own=1.0
        # Own-position is a binary tile indicator, not a pooled probability.
        out[e,a,400+binid]=own


def project_actor_maps_warp(ledger_pos,ledger_neg,ledger_ts,rx_delay,
                             pos_m,episode_step,*,mf=.8,me=.8,eta=.02,
                             age_ref=60.0,mode='sum',device='cpu',uq88=False,pool_mode='mean'):
    """Returns (E,6,500) Float32; E worlds run in parallel on CUDA when available.

    mode sum+aoi uses *per-origin* discounted fusion; raw sum (eta=0) uses
    canonical stable counts for parity. `zhang_max` follows Zhang Eq16.
    """
    if not HAS_WARP: raise RuntimeError('warp-lang is required')
    import numpy as np
    allowed={'sum':1 if eta>0 else 0,'zhang_max':2}
    if mode not in allowed: raise ValueError('invalid fusion mode')
    if pool_mode not in ('mean','semantic'): raise ValueError('invalid pooling')
    lp=np.asarray(ledger_pos,dtype=np.uint16)
    ln=np.asarray(ledger_neg,dtype=np.uint16)
    ts=np.asarray(ledger_ts,dtype=np.int32)
    delay=np.asarray(rx_delay,dtype=np.float32)
    pos=np.asarray(pos_m,dtype=np.float32)
    steps=np.asarray(episode_step,dtype=np.int32)
    e=lp.shape[0]
    if (lp.shape!=(e,7,6,400) or ln.shape!=lp.shape or ts.shape!=lp.shape
            or delay.shape!=lp.shape or pos.shape!=(e,7,3) or steps.shape!=(e,)):
        raise ValueError('invalid Warp actor map shapes')
    if not(0<mf<1 and 0<me<1 and 0<=eta and age_ref>0):
        raise ValueError('invalid DS/AoI parameters')
    args=(wp.array(lp,dtype=wp.uint16,device=device),
          wp.array(ln,dtype=wp.uint16,device=device),
          wp.array(ts,dtype=wp.int32,device=device),
          wp.array(delay,dtype=wp.float32,device=device),
          wp.array(pos,dtype=wp.float32,device=device),
          wp.array(steps,dtype=wp.int32,device=device))
    out=wp.zeros((e,6,500),dtype=wp.float32,device=device)
    wp.launch(_actor_maps_7radio_kernel,dim=e*600,inputs=[*args,out,
              float(math.log1p(-mf)),float(math.log1p(-me)),
              float(eta),float(age_ref),int(allowed[mode]),int(bool(uq88)),int(pool_mode=='semantic')],device=device)
    wp.synchronize_device(device)
    return out.numpy()
