"""Reproducible Zhang–Xue 1km scene sampling, shared CPU and Warp map reset.

Pure NumPy, no network, SimPy, actor or learner dependencies. Identical RNG
draw sequence to the v3.5 CPU pilot's original _generate_scenario routine.
"""
from __future__ import annotations

import math
import numpy as np


def sample_scene(config,rng):
    """Sample one scene with 6 starts, 2 no-fly disks, 2 blackout diamonds, 10 targets.

    Deterministic for a supplied numpy.random.Generator.
    No-fly disks restrict UAV motion, and ground target cell *centers*
    are excluded from these disks; blackout diamonds affect radio only.
    """
    c=config
    center=np.array([c.map_size_meter/2]*2)
    positions=np.zeros((c.uav_count,2),dtype=np.float32)
    for u in range(c.uav_count):
        angle=2*math.pi*u/c.uav_count
        positions[u]=center+110.*np.array([math.cos(angle),math.sin(angle)])
    forbidden=[]
    no_fly_zones=[]
    blackout_zones=[]
    for radius,count,is_blackout in (
        (c.no_fly_radius_meter,c.no_fly_zone_count,False),
        ((c.blackout_radius_cells+.5)*c.cell_size_meter,c.blackout_zone_count,True),
    ):
        for _ in range(count):
            for _attempt in range(5000):
                if is_blackout:
                    gx,gy=(int(q) for q in rng.integers(4,16,size=2))
                    x,y=(gx+.5)*c.cell_size_meter,(gy+.5)*c.cell_size_meter
                else:
                    x,y=(float(q) for q in rng.uniform(100,900,size=2))
                if math.hypot(x-center[0],y-center[1])<radius+180:
                    continue
                if any(math.hypot(x-ox,y-oy)<radius+orr+20
                       for ox,oy,orr in forbidden):
                    continue
                forbidden.append((x,y,radius))
                if is_blackout:
                    blackout_zones.append((gx,gy))
                else:
                    no_fly_zones.append((x,y,radius))
                break
            else:
                raise RuntimeError("Unable to sample nonoverlapping pilot zones")

    eligible=[]
    for cell in range(c.grid_width*c.grid_height):
        gx,gy=cell%c.grid_width,cell//c.grid_width
        x,y=(gx+.5)*c.cell_size_meter,(gy+.5)*c.cell_size_meter
        if all(math.hypot(x-ox,y-oy)>radius for ox,oy,radius in no_fly_zones):
            eligible.append(cell)
    mask=np.zeros(c.grid_width*c.grid_height,dtype=bool)
    selected=rng.choice(eligible,size=c.target_count,replace=False)
    mask[selected]=True
    return positions,mask,no_fly_zones,blackout_zones


def sample_scenario_batch(config,seeds):
    """Materialize per-env scenes for batched Warp map (CPU sampling at reset).

    CPU scene generation remains one-time per episode; the 4096-environment
    sensor updates and DS posterior are Warp kernels. Each seed is unique
    to avoid duplicate scenes when training samples independent rollouts.
    """
    ids=[int(x) for x in seeds]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError("Scenario seeds must be a nonempty unique sequence")
    e=len(ids)
    positions=np.empty((e,config.uav_count,2),np.float32)
    target_mask=np.empty((e,config.grid_height*config.grid_width),bool)
    no_fly=np.empty((e,config.no_fly_zone_count,3),np.float64)
    blackout=np.empty((e,config.blackout_zone_count,2),np.int32)
    for k,seed in enumerate(ids):
        pos,truth,fly,radio=sample_scene(config,np.random.default_rng(seed))
        positions[k]=pos
        target_mask[k]=truth
        no_fly[k]=fly
        blackout[k]=radio
    return {
        "positions":positions,
        "target_mask":target_mask,
        "no_fly_zones":no_fly,
        "blackout_zones":blackout,
    }
