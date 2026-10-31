"""Bounded XYZ dynamic-window motion with swept cylinder/peer certificates."""
import numpy as np
from .avoidance import reachable_velocities
from .xyz_geometry import cylinder_hit
from .cbf import filter_xyz_acceleration


def _peer_clear(a,b,distance):
    start=a[:-1]-b[:-1];delta=np.diff(a,axis=0)-np.diff(b,axis=0)
    fraction=np.clip(-(start*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-24),0,1)
    return bool((np.linalg.norm(start+fraction[:,None]*delta,axis=-1)>=distance-1e-9).all())


def move_xyz(positions,velocities,acceleration,obstacles,heights,config,disabled=None,diagnostics=None):
    w=config.world;n=len(positions);step=w.max_accel_mps2*w.dt_s
    if w.motion_safety=="dynamic_window":
        raise NotImplementedError("XYZ dynamic-window safety was retired; use its pinned historical source")
    disabled=np.zeros(n,bool) if disabled is None else disabled
    nominal=reachable_velocities(velocities+acceleration*step,velocities,w);nominal[disabled]=0
    lo=np.array([0.,0.,w.altitude_min_m]);hi=np.array([w.map_size_m,w.map_size_m,w.altitude_max_m])
    cylinders=[np.r_[o,h] for o,h in zip(obstacles,heights)]
    counts={f"shield_{k}":0 for k in ("interventions","obstacle_interventions","peer_interventions","boundary_interventions","infeasible")}
    chosen=nominal.copy()
    if w.motion_safety=="cbf":
        physical_nominal=np.clip(acceleration,-1.,1.)*w.max_accel_mps2
        physical_nominal[disabled]=0.
        safe_accel,cbf=filter_xyz_acceleration(positions,velocities,physical_nominal,obstacles,heights,w,disabled)
        nominal=reachable_velocities(velocities+physical_nominal*w.dt_s,velocities,w);nominal[disabled]=0
        chosen=reachable_velocities(velocities+safe_accel*w.dt_s,velocities,w);chosen[disabled]=0
        counts.update(
            shield_interventions=cbf["cbf_interventions"],
            shield_obstacle_interventions=cbf["cbf_obstacle_interventions"],
            shield_peer_interventions=cbf["cbf_peer_interventions"],
            shield_boundary_interventions=cbf["cbf_boundary_interventions"],
            shield_infeasible=cbf["cbf_infeasible"],
        )
    end=np.clip(positions+chosen*w.dt_s,lo,hi);end[disabled]=positions[disabled]
    clipped=np.any(end!=positions+chosen*w.dt_s,axis=1)
    blocked=np.array([any(cylinder_hit(a,b,o,w.obstacle_clearance_m) for o in cylinders)
                      for a,b in zip(positions,end)])
    obstacle_blocked=blocked.copy()
    for _ in range(n+1):
        resolved=np.where(blocked[:,None],positions,end);old=blocked.copy()
        for i in range(n):
            for j in range(i):
                if not _peer_clear(np.stack([positions[i],resolved[i]]),
                                   np.stack([positions[j],resolved[j]]),w.safety_distance_m):
                    blocked[i]=blocked[j]=True
        if np.array_equal(old,blocked):break
    resolved=np.where(blocked[:,None],positions,end)
    actual=(resolved-positions)/w.dt_s
    if diagnostics is not None:
        diagnostics.update(counts)
        diagnostics.update(obstacle_blocks=int((obstacle_blocked&~disabled).sum()),
            peer_blocks=int((blocked&~obstacle_blocked&~disabled).sum()),
            commanded_stalls=int(((np.linalg.norm(acceleration,axis=1)>.1)&(np.linalg.norm(actual,axis=1)<1e-8)&~disabled).sum()),
            shield_correction=float(np.linalg.norm(chosen-nominal,axis=1).mean()/w.max_speed_mps))
    return resolved,actual,int((blocked&~disabled).sum()),int((clipped&~disabled).sum())
