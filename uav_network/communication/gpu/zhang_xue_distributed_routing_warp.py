"""Distributed distance-vector routing on actual Warp arrays, no oracle.

This module has no access to UAV ground truth, map obstacles, coordinates or
global active PHY links. A path can only be learned after actual received
HELLO packets on both directions, followed by an actual successfully
received advertised distance vector. Stale HELLO and DV advertisements expire. A source advertises unreachable
for a destination if its next hop is the very peer receiving the distance
vector, which prevents two-node count-to-infinity after a link failure.

The link-layer simulator is responsible for determining HELLO/DV successes.
The route module never assumes a neighbor can be heard from its coordinates.
"""
from __future__ import annotations
import numpy as np
import warp as wp

N=6
INF=99


@wp.kernel
def _reset(
    heard:wp.array(dtype=wp.int64,ndim=3),
    adv_stamp:wp.array(dtype=wp.int64,ndim=3),
    adv:wp.array(dtype=wp.int32,ndim=4),
    hop:wp.array(dtype=wp.int32,ndim=3),
    next_hop:wp.array(dtype=wp.int32,ndim=3),
):
    e,u,d=wp.tid()
    heard[e,u,d]=wp.int64(-1)
    adv_stamp[e,u,d]=wp.int64(-1)
    hop[e,u,d]=0 if u==d else INF
    next_hop[e,u,d]=u if u==d else -1
    for dst in range(N):
        adv[e,u,d,dst]=INF


@wp.kernel
def _receive_hello(
    receivers:wp.array(dtype=wp.int32,ndim=2),
    successes:wp.array(dtype=wp.int32,ndim=2),
    now:wp.array(dtype=wp.int64,ndim=1),
    heard:wp.array(dtype=wp.int64,ndim=3),
):
    e,sender=wp.tid()
    if successes[e,sender]!=0:
        rx=receivers[e,sender]
        heard[e,rx,sender]=now[e]


@wp.kernel
def _receive_distance_vector(
    receivers:wp.array(dtype=wp.int32,ndim=2),
    successes:wp.array(dtype=wp.int32,ndim=2),
    now:wp.array(dtype=wp.int64,ndim=1),
    heard:wp.array(dtype=wp.int64,ndim=3),
    sender_hop:wp.array(dtype=wp.int32,ndim=3),
    sender_next_hop:wp.array(dtype=wp.int32,ndim=3),
    adv:wp.array(dtype=wp.int32,ndim=4),
    adv_stamp:wp.array(dtype=wp.int64,ndim=3),
    hello_ttl:int,
):
    e,sender,dst=wp.tid()
    if successes[e,sender]==0:
        return
    rx=receivers[e,sender]
    if heard[e,rx,sender]<0 or heard[e,sender,rx]<0:
        return
    if (now[e]-heard[e,rx,sender]>wp.int64(hello_ttl) or
        now[e]-heard[e,sender,rx]>wp.int64(hello_ttl)):
        return
    # Split horizon with poisoned reverse. Source cannot advertise
    # a path to a neighbor if that path already relies on the neighbor:
    # otherwise two nodes can route to each other after a link expires.
    # The physical DV frame's success is still the entry barrier.
    advertised=sender_hop[e,sender,dst]
    if sender_next_hop[e,sender,dst]==rx:
        advertised=wp.int32(INF)
    adv[e,rx,sender,dst]=advertised
    adv_stamp[e,rx,sender]=now[e]


@wp.kernel
def _compute_routes(
    heard:wp.array(dtype=wp.int64,ndim=3),
    adv_stamp:wp.array(dtype=wp.int64,ndim=3),
    adv:wp.array(dtype=wp.int32,ndim=4),
    now:wp.array(dtype=wp.int64,ndim=1),
    hop:wp.array(dtype=wp.int32,ndim=3),
    next_hop:wp.array(dtype=wp.int32,ndim=3),
    links:wp.array(dtype=wp.int32,ndim=3),
    hello_ttl:int,
):
    e,u,dst=wp.tid()
    if u==dst:
        hop[e,u,dst]=0
        next_hop[e,u,dst]=u
    else:
        best=wp.int32(INF)
        via=wp.int32(-1)
        for peer in range(N):
            if peer==u:
                continue
            h1=heard[e,u,peer]
            h2=heard[e,peer,u]
            alive=(h1>=0 and h2>=0 and
                   now[e]-h1<=wp.int64(hello_ttl) and
                   now[e]-h2<=wp.int64(hello_ttl))
            if alive:
                if peer==dst and 1<best:
                    best=1
                    via=peer
                elif adv_stamp[e,u,peer]>=0 and (
                    now[e]-adv_stamp[e,u,peer]<=wp.int64(hello_ttl)
                ):
                    candidate=adv[e,u,peer,dst]+1
                    if candidate<best and candidate<N:
                        best=candidate
                        via=peer
        hop[e,u,dst]=best
        next_hop[e,u,dst]=via
    h=heard[e,u,dst]
    rev=heard[e,dst,u]
    links[e,u,dst]=0
    if (u!=dst and h>=0 and rev>=0 and
        now[e]-h<=wp.int64(hello_ttl) and
        now[e]-rev<=wp.int64(hello_ttl)):
        links[e,u,dst]=1


class ZhangXueWarpDistanceVector:
    """DV routes derived ONLY from successful bidirectional HELLO and DV packets."""
    def __init__(self,num_envs:int,*,hello_ttl_us=5_000_000,device="cpu"):
        if not isinstance(num_envs,int) or num_envs<1:
            raise ValueError("positive num_envs required")
        if not isinstance(hello_ttl_us,int) or hello_ttl_us<=0:
            raise ValueError("positive integer hello TTL required")
        self.num_envs=num_envs
        self.device=device
        self.hello_ttl_us=hello_ttl_us
        self.now_us=0
        wp.init()
        three=(num_envs,N,N)
        four=(num_envs,N,N,N)
        self.heard=wp.zeros(three,dtype=wp.int64,device=device)
        self.adv_stamp=wp.zeros(three,dtype=wp.int64,device=device)
        self.adv=wp.zeros(four,dtype=wp.int32,device=device)
        self.hop=wp.zeros(three,dtype=wp.int32,device=device)
        self.next_hop=wp.zeros(three,dtype=wp.int32,device=device)
        self.links=wp.zeros(three,dtype=wp.int32,device=device)
        self.reset()

    def reset(self):
        wp.launch(_reset,dim=(self.num_envs,N,N),inputs=[
            self.heard,self.adv_stamp,self.adv,self.hop,self.next_hop,
        ],device=self.device)
        self.now_us=0
        self._compute()

    def _stamp(self,now_us):
        if not isinstance(now_us,(int,np.integer)) or now_us<0:
            raise ValueError("time must be nonnegative integer microseconds")
        if int(now_us)<self.now_us:
            raise ValueError("cannot reverse route time")
        self.now_us=int(now_us)
        return wp.array(np.full(self.num_envs,int(now_us),np.int64),
                        dtype=wp.int64,device=self.device)

    def _fields(self,dst,good):
        receiver=np.asarray(dst)
        successes=np.asarray(good)
        if receiver.shape!=(self.num_envs,N) or successes.shape!=(self.num_envs,N):
            raise ValueError("receiver/success must be [E,6]")
        if (not np.issubdtype(receiver.dtype,np.integer) or
            not np.issubdtype(successes.dtype,np.integer)):
            raise ValueError("routing frame fields must be integers")
        if np.any((successes!=0)&(successes!=1)):
            raise ValueError("success must be binary")
        active=successes!=0
        if np.any((receiver[active]<0)|(receiver[active]>=N)):
            raise ValueError("bad receiver")
        if np.any(active&(receiver==np.arange(N)[None,:])):
            raise ValueError("no HELLO to sender self")
        return (wp.array(receiver.astype(np.int32),dtype=wp.int32,
                         device=self.device),
                wp.array(successes.astype(np.int32),dtype=wp.int32,
                         device=self.device))

    def _compute(self):
        t=wp.array(np.full(self.num_envs,self.now_us,np.int64),
                   dtype=wp.int64,device=self.device)
        wp.launch(_compute_routes,dim=(self.num_envs,N,N),inputs=[
            self.heard,self.adv_stamp,self.adv,t,self.hop,
            self.next_hop,self.links,self.hello_ttl_us,
        ],device=self.device)

    def receive_hello(self,receivers,successful,*,now_us):
        dst,good=self._fields(receivers,successful)
        time=self._stamp(now_us)
        wp.launch(_receive_hello,dim=(self.num_envs,N),inputs=[
            dst,good,time,self.heard,
        ],device=self.device)
        self._compute()

    def receive_advertisement(self,receivers,successful,*,now_us):
        dst,good=self._fields(receivers,successful)
        time=self._stamp(now_us)
        wp.launch(_receive_distance_vector,dim=(self.num_envs,N,N),inputs=[
            dst,good,time,self.heard,self.hop,self.next_hop,
            self.adv,self.adv_stamp,self.hello_ttl_us,
        ],device=self.device)
        self._compute()

    def tick(self,now_us):
        self._stamp(now_us)
        self._compute()

    def read_routes(self):
        return {
            "next_hop":self.next_hop.numpy(),
            "hop_count":self.hop.numpy(),
            "bidirectional_links":self.links.numpy(),
        }
