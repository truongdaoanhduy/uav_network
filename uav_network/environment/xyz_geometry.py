"""Swept XYZ cylinder geometry, without target/report knowledge."""
import numpy as np


def cylinder_hit(start, end, obstacle, margin=0.0):
    a,b,o = np.asarray(start,float),np.asarray(end,float),np.asarray(obstacle,float)
    d=b-a
    low,high=0.,1.
    top=o[3]+margin
    if abs(d[2])<1e-15:
        if a[2]<0 or a[2]>top: return False
    else:
        u,v=(0-a[2])/d[2],(top-a[2])/d[2]
        low,high=max(low,min(u,v)),min(high,max(u,v))
    q=a[:2]-o[:2]
    aa=float(d[:2]@d[:2]); cc=float(q@q)-(o[2]+margin)**2
    if aa<1e-24:
        if cc>0: return False
    else:
        bb=2*float(q@d[:2]); disc=bb*bb-4*aa*cc
        if disc<0: return False
        root=np.sqrt(max(0.,disc))
        low=max(low,(-bb-root)/(2*aa));high=min(high,(-bb+root)/(2*aa))
    return low<=high


def camera_clear(start, ground_xy, obstacles, heights):
    end=np.r_[ground_xy,0.]
    return not any(cylinder_hit(start,end,np.r_[o,h]) for o,h in zip(obstacles,heights))


def prepare_scene(scene, config, seed, options=None):
    from .bayesian import sensor_uniform
    w=config.world
    if scene.positions.shape[1]==2:
        scene.positions=np.column_stack((scene.positions,np.full(w.num_uavs,config.sensor.altitude_m)))
    if options is not None:
        pos=np.asarray(options["positions"],float)
        if pos.shape!=(w.num_uavs,3) or not np.isfinite(pos).all():
            raise ValueError("XYZ positions must be finite Nx3")
        scene.positions=pos.copy()
    scene.gcs=np.r_[scene.gcs[:2],0.]
    heights=np.array([w.obstacle_height_min_m+(w.obstacle_height_max_m-w.obstacle_height_min_m)*
                     sensor_uniform(seed,0,0,o+1610612736) for o in range(w.num_obstacles)])
    if options is not None and len(options["obstacles"]):
        raw=np.asarray(options["obstacles"],float)
        if raw.shape==(w.num_obstacles,4): heights=raw[:,3].copy()
    scene.obstacle_heights=heights
    lo=np.array([0.,0.,w.altitude_min_m]);hi=np.array([w.map_size_m,w.map_size_m,w.altitude_max_m])
    if ((scene.positions<lo).any() or (scene.positions>hi).any()
            or not np.isfinite(heights).all() or (heights<=0).any()):
        raise ValueError("Invalid XYZ scene bounds/heights")
    for i,pos in enumerate(scene.positions):
        if any(cylinder_hit(pos,pos,np.r_[o,h],w.obstacle_clearance_m)
               for o,h in zip(scene.obstacles,heights)):
            raise ValueError("XYZ UAV starts in obstacle clearance")
        if any(np.linalg.norm(pos-other)<w.safety_distance_m for other in scene.positions[:i]):
            raise ValueError("XYZ launch peer safety distance")
    return scene
