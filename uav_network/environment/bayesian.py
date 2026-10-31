"""Local probabilistic camera model adapted from the user's reference project."""
import math
import numpy as np
from .xyz_geometry import camera_clear


def sensor_profile(sensor, altitude):
    z=float(altitude)
    return (float(np.interp(z,sensor.altitude_anchors,sensor.pd)),
            float(np.interp(z,sensor.altitude_anchors,sensor.pf)),
            max(0.,z)*math.tan(math.radians(sensor.full_fov_deg/2)))


def sensor_uniform(seed,tick,agent,cell):
    mask=0xffffffff
    x=(int(seed) ^ (int(tick)*0x9e3779b9) ^ (int(agent)*0x85ebca6b) ^ (int(cell)*0xc2b2ae35))&mask
    x=((x^(x>>16))*0x7feb352d)&mask
    x=((x^(x>>15))*0x846ca68b)&mask
    return ((x^(x>>16))+.5)/4294967296.


def bayes_update(prior, observation, pd, pf):
    p=min(1-1e-8,max(1e-8,float(prior)))
    yes=pd if observation else 1-pd
    no=pf if observation else 1-pf
    return min(1-1e-8,max(1e-8,yes*p/max(yes*p+no*(1-p),1e-8)))


def binary_entropy_bits(probability):
    p=np.clip(np.asarray(probability,dtype=np.float64),1e-12,1-1e-12)
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


def belief_certainty_potential(values):
    entropy=binary_entropy_bits(values)
    return float(np.clip(1-float(np.mean(entropy)),0,1))


def coarse_belief_uncertainty_map(belief_map, output_cells):
    values=np.asarray(belief_map,dtype=np.float64)
    if values.ndim!=2:
        raise ValueError("belief_map must be 2-D")
    count=int(output_cells)
    if count<1 or count>min(values.shape):
        raise ValueError("belief_coarse_cells must fit inside the belief map")
    entropy=binary_entropy_bits(values)
    rows=np.array_split(np.arange(values.shape[0]),count)
    cols=np.array_split(np.arange(values.shape[1]),count)
    return np.asarray([[entropy[np.ix_(r,c)].mean() for c in cols] for r in rows],dtype=np.float32)


def investigation_potential(env):
    s,w=env.config.sensor,env.config.world
    threshold=float(env.config.reward.investigation_belief_threshold)
    if not s.belief_prior < threshold < s.confirmation_threshold:
        return 0.0
    radius=s.belief_patch_cells//2;g=env.belief.shape[-1];width=s.grid_cell_m
    altitude_span=max(w.altitude_max_m-s.fine_altitude_m,1e-6);total=0.0
    for i,pos in enumerate(env.positions):
        if not env.active[i]: continue
        gx=min(max(int(math.floor(pos[0]/width)),0),g-1);gy=min(max(int(math.floor(pos[1]/width)),0),g-1)
        x0,x1=max(0,gx-radius),min(g,gx+radius+1);y0,y1=max(0,gy-radius),min(g,gy+radius+1)
        patch=env.belief[i,y0:y1,x0:x1]
        py,px=np.unravel_index(int(np.argmax(patch)),patch.shape)
        peak=float(patch[py,px]);evidence=float(np.clip((peak-threshold)/max(s.confirmation_threshold-threshold,1e-6),0,1))
        peak_xy=np.array([(x0+px+.5)*width,(y0+py+.5)*width])
        max_distance=max(math.sqrt(2)*(radius+.5)*width,1e-6)
        proximity=float(np.clip(1-np.linalg.norm(peak_xy-pos[:2])/max_distance,0,1))
        descent=float(np.clip((w.altitude_max_m-pos[2])/altitude_span,0,1))
        total+=evidence*.5*(proximity+descent)
    return float(total/max(1,w.num_uavs))


def _integral(x,r):
    x=min(r,max(-r,x))
    return .5*(x*math.sqrt(max(0.,r*r-x*x))+r*r*math.asin(x/r))


def cell_coverage(gx,gy,xy,r,width,map_size):
    if r<=0: return 0.
    x0,x1=gx*width-xy[0],min((gx+1)*width,map_size)-xy[0]
    y0,y1=gy*width-xy[1],min((gy+1)*width,map_size)-xy[1]
    left,right=max(x0,-r),min(x1,r)
    if right<=left or y1<=-r or y0>=r: return 0.
    cuts=[left,right]
    for y in (y0,y1):
        if abs(y)<r:
            q=math.sqrt(max(0.,r*r-y*y));cuts.extend(x for x in (-q,q) if left<x<right)
    cuts=sorted(set(cuts));area=0.
    for a,b in zip(cuts,cuts[1:]):
        h=math.sqrt(max(0.,r*r-((a+b)/2)**2))
        if min(y1,h)<=max(y0,-h): continue
        arc=_integral(b,r)-_integral(a,r)
        area+=(y1*(b-a) if y1<h else arc)-(y0*(b-a) if y0>-h else -arc)
    total=(x1-x0)*(y1-y0)
    return min(1.,max(0.,area/total))


def initialize_sensor(env):
    c=env.config;g=int(math.ceil(c.world.map_size_m/c.sensor.grid_cell_m))
    env.belief=np.full((c.world.num_uavs,g,g),c.sensor.belief_prior,dtype=np.float64)
    env.visited=np.zeros_like(env.belief,dtype=bool);env.positive=np.zeros_like(env.belief,dtype=bool)
    env.sensor_tick=0
    env.sensor_counts=dict(measurements=0,positive=0,false_positive=0,false_negative=0,false_confirmation=0,confirmations=0)


def sense_bayesian(env):
    c,s,w=env.config,env.config.sensor,env.config.world;g=env.belief.shape[-1];width=s.grid_cell_m
    env.positive.fill(False)
    for i,pos in enumerate(env.positions):
        if not env.active[i] or pos[2]<=0: continue
        pd,pf,r=sensor_profile(s,pos[2]);x0=max(0,int(math.floor((pos[0]-r)/width)));x1=min(g-1,int(math.floor((pos[0]+r)/width)))
        y0=max(0,int(math.floor((pos[1]-r)/width)));y1=min(g-1,int(math.floor((pos[1]+r)/width)))
        for gy in range(y0,y1+1):
            for gx in range(x0,x1+1):
                q=cell_coverage(gx,gy,pos[:2],r,width,w.map_size_m)
                if q<=0: continue
                center=np.array([min((gx+.5)*width,w.map_size_m),min((gy+.5)*width,w.map_size_m)])
                if s.obstacle_occlusion and not camera_clear(pos,center,env.obstacles,env.obstacle_heights): continue
                ids=[t for t,target in enumerate(env.targets) if min(int(target[0]/width),g-1)==gx and min(int(target[1]/width),g-1)==gy and np.linalg.norm(target-pos[:2])<=r and (not s.obstacle_occlusion or camera_clear(pos,target,env.obstacles,env.obstacle_heights))]
                ef=1-(1-pf)**q;ed=q*pd+(1-q)*ef
                yes=sensor_uniform(env.seed,env.sensor_tick,i,gy*g+gx)<(pd if ids else ef)
                p=bayes_update(env.belief[i,gy,gx],yes,ed,ef)
                env.belief[i,gy,gx]=p;env.visited[i,gy,gx]=True;env.positive[i,gy,gx]=yes
                for name,value in (("measurements",1),("positive",yes),("false_positive",yes and not ids),("false_negative",not yes and bool(ids))): env.sensor_counts[name]+=int(value)
                if yes and pos[2]<=s.fine_altitude_m and p>=s.confirmation_threshold:
                    false=not ids;env.sensor_counts["false_confirmation"]+=int(false)
                    if false and s.verified_empty_belief>0: env.belief[i,gy,gx]=s.verified_empty_belief
                    for t in ids: env.sensor_counts["confirmations"]+=int(env.reports.discover(i,t,env.time_s))
    env.sensor_tick+=1


def belief_patch(env,i):
    s=env.config.sensor;k=s.belief_patch_cells;g=env.belief.shape[-1];half=k//2
    x=min(int(env.positions[i,0]/s.grid_cell_m),g-1);y=min(int(env.positions[i,1]/s.grid_cell_m),g-1)
    result=np.full((k,k),s.belief_prior)
    for j in range(k):
        for a in range(k):
            xx,yy=x+a-half,y+j-half
            if 0<=xx<g and 0<=yy<g: result[j,a]=env.belief[i,yy,xx]
    return result.ravel()
