"""Bounded XYZ dynamic-window motion with swept cylinder/peer certificates."""
from itertools import product
import numpy as np
from .avoidance import reachable_velocities, braking_paths
from .xyz_geometry import cylinder_hit


def _peer_clear(a,b,distance):
    start=a[:-1]-b[:-1];delta=np.diff(a,axis=0)-np.diff(b,axis=0)
    fraction=np.clip(-(start*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-24),0,1)
    return bool((np.linalg.norm(start+fraction[:,None]*delta,axis=-1)>=distance-1e-9).all())


def move_xyz(positions,velocities,acceleration,obstacles,heights,config,disabled=None,diagnostics=None):
    w=config.world;n=len(positions);step=w.max_accel_mps2*w.dt_s
    disabled=np.zeros(n,bool) if disabled is None else disabled
    nominal=reachable_velocities(velocities+acceleration*step,velocities,w);nominal[disabled]=0
    lo=np.array([0.,0.,w.altitude_min_m]);hi=np.array([w.map_size_m,w.map_size_m,w.altitude_max_m])
    cylinders=[np.r_[o,h] for o,h in zip(obstacles,heights)]
    def static(path):
        return bool(((path>=lo)&(path<=hi)).all() and
                    not any(cylinder_hit(a,b,o,w.obstacle_clearance_m)
                            for a,b in zip(path,path[1:]) for o in cylinders))
    speed=np.linalg.norm(velocities,axis=1,keepdims=True)
    chosen=velocities*np.maximum(0,1-step/np.maximum(speed,1e-12));chosen[disabled]=0
    paths=braking_paths(positions,chosen,w)
    counts={f"shield_{k}":0 for k in ("interventions","obstacle_interventions","peer_interventions","boundary_interventions","infeasible")}
    if w.motion_safety=="dynamic_window":
        # Pass the policy through unchanged when the complete joint nominal
        # braking trajectories are already safe.  Without this fast path, an
        # early-ID UAV is compared with later peers' fallback braking paths,
        # which can create a correction even though all peers following their
        # nominal commands would remain collision-free.
        nominal_paths=braking_paths(positions,nominal,w)
        joint_safe=all(static(path) for path in nominal_paths)
        if joint_safe:
            joint_safe=all(_peer_clear(nominal_paths[i],nominal_paths[j],w.safety_distance_m)
                           for i in range(n) for j in range(i))
        if joint_safe:
            chosen=nominal.copy();paths=nominal_paths
        else:
            grid=np.array(list(product((-1.,0.,1.),repeat=3)))*step
            for i in range(n):
                if disabled[i]:continue
                candidates=reachable_velocities(np.vstack([nominal[i],chosen[i],velocities[i]+grid]),velocities[i],w)
                order=np.argsort(np.square(candidates-nominal[i]).sum(1),kind="stable")
                selected=False
                for j in order:
                    path=braking_paths(positions[i:i+1],candidates[j:j+1],w)[0]
                    if static(path) and all(_peer_clear(path,paths[k],w.safety_distance_m) for k in range(n) if k!=i):
                        chosen[i]=candidates[j];paths[i]=path;selected=True;break
                counts["shield_infeasible"]+=int(not selected)
                counts["shield_interventions"]+=int(not np.allclose(chosen[i],nominal[i],rtol=0,atol=1e-9))
    else: chosen=nominal.copy()
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
