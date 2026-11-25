"""Zhang–Xue 1km 20×20 map-only batched Warp sensor and DS belief backend.

Real Warp kernels on CPU or GPU. Separate from packet transport, RL learner
and old 800m/25-target Warp environment. Mission reward/evaluation cannot
substitute this map-only proof for full Warp environment parity.
"""
from __future__ import annotations

import math
import numpy as np
import warp as wp

NUM_AGENTS=6
GRID_WIDTH=20
NUM_CELLS=400


@wp.func
def _ds_from_counts(pos: wp.int32,neg: wp.int32) -> wp.vec2f:
    """Equal evidence masses 0.8: stable integer-difference log-space DS.

    Returns (mass_target, mass_unknown). m_empty=1-m_target-m_unknown.
    Intermediate powers are normalized by max(loglikelihood) to avoid
    catastrophic cancellation when pos and neg are similar.
    """
    logq=wp.float32(-1.6094379124341003)
    dn=pos-neg
    exp_a=wp.exp(wp.float32(pos)*logq)
    ea=wp.float32(1.)
    eb=wp.float32(1.)
    if dn>=wp.int32(0):
        ea=wp.exp(wp.float32(dn)*logq)
    else:
        eb=wp.exp(wp.float32(-dn)*logq)
    q=wp.exp(wp.float32(wp.max(pos,neg))*logq)
    denom=ea+eb-q
    mf=(eb*(wp.float32(1.)-exp_a))/denom
    mu=q/denom
    return wp.vec2f(mf,mu)


@wp.kernel
def _sense_with_draws(
    poses: wp.array(dtype=wp.vec2f,ndim=2),
    truth: wp.array(dtype=wp.int32,ndim=2),
    draws: wp.array(dtype=wp.float32,ndim=3),
    pos: wp.array(dtype=wp.int32,ndim=3),
    neg: wp.array(dtype=wp.int32,ndim=3),
    mass_target: wp.array(dtype=wp.float32,ndim=3),
    unknown: wp.array(dtype=wp.float32,ndim=3),
    radius_sq: wp.float32,
    pd: wp.float32,
    pf: wp.float32,
):
    env,u,cid=wp.tid()
    x=(wp.float32(cid%GRID_WIDTH)+wp.float32(.5))*wp.float32(50.)
    y=(wp.float32(cid//GRID_WIDTH)+wp.float32(.5))*wp.float32(50.)
    dp=x-poses[env,u][0]
    dq=y-poses[env,u][1]
    if dp*dp+dq*dq>radius_sq+wp.float32(.00001):
        return
    prob=pf
    if truth[env,cid]!=0:
        prob=pd
    if draws[env,u,cid]<prob:
        pos[env,u,cid]=wp.min(pos[env,u,cid]+1,wp.int32(65535))
    else:
        neg[env,u,cid]=wp.min(neg[env,u,cid]+1,wp.int32(65535))
    m=_ds_from_counts(pos[env,u,cid],neg[env,u,cid])
    mass_target[env,u,cid]=m[0]
    unknown[env,u,cid]=m[1]


@wp.kernel
def _sense_with_rng(
    poses: wp.array(dtype=wp.vec2f,ndim=2),
    truth: wp.array(dtype=wp.int32,ndim=2),
    pos: wp.array(dtype=wp.int32,ndim=3),
    neg: wp.array(dtype=wp.int32,ndim=3),
    mass_target: wp.array(dtype=wp.float32,ndim=3),
    unknown: wp.array(dtype=wp.float32,ndim=3),
    radius_sq: wp.float32,
    pd: wp.float32,
    pf: wp.float32,
    seed: wp.int32,
    step_index: wp.int32,
):
    env,u,cid=wp.tid()
    x=(wp.float32(cid%GRID_WIDTH)+wp.float32(.5))*wp.float32(50.)
    y=(wp.float32(cid//GRID_WIDTH)+wp.float32(.5))*wp.float32(50.)
    dp=x-poses[env,u][0]
    dq=y-poses[env,u][1]
    if dp*dp+dq*dq>radius_sq+wp.float32(.00001):
        return
    # Warp deterministic per-thread random state; independent of thread order.
    state=wp.rand_init(seed+step_index*wp.int32(1000003),env*wp.int32(2400)+u*wp.int32(400)+cid)
    draw=wp.randf(state)
    prob=pf
    if truth[env,cid]!=0:
        prob=pd
    if draw<prob:
        pos[env,u,cid]=wp.min(pos[env,u,cid]+1,wp.int32(65535))
    else:
        neg[env,u,cid]=wp.min(neg[env,u,cid]+1,wp.int32(65535))
    m=_ds_from_counts(pos[env,u,cid],neg[env,u,cid])
    mass_target[env,u,cid]=m[0]
    unknown[env,u,cid]=m[1]


@wp.kernel
def _fuse_team_counts(
    pos: wp.array(dtype=wp.int32,ndim=3),
    neg: wp.array(dtype=wp.int32,ndim=3),
    mf: wp.array(dtype=wp.float32,ndim=2),
    mu: wp.array(dtype=wp.float32,ndim=2),
):
    env,cid=wp.tid()
    p=wp.int32(0)
    n=wp.int32(0)
    for u in range(NUM_AGENTS):
        p+=pos[env,u,cid]
        n+=neg[env,u,cid]
    m=_ds_from_counts(p,n)
    mf[env,cid]=m[0]
    mu[env,cid]=m[1]


class ZhangXueWarpMapBatch:
    """Map-state device arrays for E independent 6-UAV scenarios.

    Test mode accepts external uniform_draws[E,6,400] for precise CPU
    parity. Run mode uses Warp per-thread seeded sampling on the same device
    as belief counters, so it avoids large host→device random matrices.
    The simulator does not implement no-fly motion, packet/network or RL.
    """
    def __init__(self,num_envs:int,*,device="cpu",
                 altitude_m=100.,camera_fov_degrees=90.,
                 pd=.85,pf=.03):
        self.num_envs=int(num_envs)
        if self.num_envs<1:
            raise ValueError("num_envs must be positive")
        if not (0.<camera_fov_degrees<180. and altitude_m>0.):
            raise ValueError("Invalid camera FoV or altitude")
        if not 0.<=pf<pd<=1.:
            raise ValueError("Invalid Pd/Pf")
        wp.init()
        self.device=device
        self.radius_sq=float(altitude_m*math.tan(math.radians(camera_fov_degrees/2)))**2
        self.pd=float(pd)
        self.pf=float(pf)
        shape=(self.num_envs,NUM_AGENTS,NUM_CELLS)
        self.pos=wp.zeros(shape,dtype=wp.int32,device=device)
        self.neg=wp.zeros(shape,dtype=wp.int32,device=device)
        self.mass_target=wp.zeros(shape,dtype=wp.float32,device=device)
        self.unknown=wp.ones(shape,dtype=wp.float32,device=device)
        team=(self.num_envs,NUM_CELLS)
        self.team_mass_target=wp.zeros(team,dtype=wp.float32,device=device)
        self.team_unknown=wp.ones(team,dtype=wp.float32,device=device)

    def reset(self):
        self.pos.zero_()
        self.neg.zero_()
        self.mass_target.zero_()
        self.team_mass_target.zero_()
        wp.launch(
            _initialize_unknown,dim=(self.num_envs,NUM_AGENTS,NUM_CELLS),
            inputs=[self.unknown],device=self.device,
        )
        wp.launch(
            _initialize_team_unknown,dim=(self.num_envs,NUM_CELLS),
            inputs=[self.team_unknown],device=self.device,
        )

    def step(self,positions_xy,truth_cells,*,uniform_draws=None,seed=44,step_index=0):
        positions=np.asarray(positions_xy,dtype=np.float32)
        truth=np.asarray(truth_cells,dtype=np.int32)
        if positions.shape!=(self.num_envs,NUM_AGENTS,2):
            raise ValueError("positions_xy must be [E,6,2]")
        if truth.shape!=(self.num_envs,NUM_CELLS):
            raise ValueError("truth_cells must be [E,400]")
        if not np.isfinite(positions).all() or np.any(truth<0) or np.any(truth>1):
            raise ValueError("Invalid position/truth values")
        poses_wp=wp.array(positions,dtype=wp.vec2f,device=self.device)
        truth_wp=wp.array(truth,dtype=wp.int32,device=self.device)
        inputs=[poses_wp,truth_wp]
        kernel=_sense_with_rng
        if uniform_draws is not None:
            draws=np.asarray(uniform_draws,dtype=np.float32)
            if draws.shape!=(self.num_envs,NUM_AGENTS,NUM_CELLS):
                raise ValueError("uniform_draws must be [E,6,400]")
            if not np.isfinite(draws).all() or np.any(draws<0.) or np.any(draws>=1.):
                raise ValueError("uniform_draws must be in [0,1)")
            inputs.append(wp.array(draws,dtype=wp.float32,device=self.device))
            kernel=_sense_with_draws
        inputs.extend([
            self.pos,self.neg,self.mass_target,self.unknown,
            self.radius_sq,self.pd,self.pf,
        ])
        if uniform_draws is None:
            inputs.extend([int(seed),int(step_index)])
        wp.launch(kernel,dim=(self.num_envs,NUM_AGENTS,NUM_CELLS),
                  inputs=inputs,device=self.device)
        wp.launch(
            _fuse_team_counts,dim=(self.num_envs,NUM_CELLS),
            inputs=[self.pos,self.neg,self.team_mass_target,self.team_unknown],
            device=self.device,
        )

    def read(self):
        return {
            "pos":self.pos.numpy(),
            "neg":self.neg.numpy(),
            "mass_target":self.mass_target.numpy(),
            "unknown":self.unknown.numpy(),
            "team_mass_target":self.team_mass_target.numpy(),
            "team_unknown":self.team_unknown.numpy(),
        }


@wp.kernel
def _initialize_unknown(out:wp.array(dtype=wp.float32,ndim=3)):
    e,u,cid=wp.tid()
    out[e,u,cid]=wp.float32(1.)


@wp.kernel
def _initialize_team_unknown(out:wp.array(dtype=wp.float32,ndim=2)):
    e,cid=wp.tid()
    out[e,cid]=wp.float32(1.)
