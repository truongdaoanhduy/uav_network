"""Real Warp NoOracle CPU trajectory hash; use separate source dirs for A/B."""
import argparse,hashlib,json,statistics,time,os
from pathlib import Path
import numpy as np,torch,warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def run(config_file,envs,steps,seed,autonomous):
    c=load_config(config_file)
    c.runtime.device='cpu'
    c.runtime.num_envs=envs
    c.runtime.deterministic=True
    c.runtime.replay_device='cpu'
    c.tracking.enabled=False
    c.validate()
    rng=np.random.default_rng(seed+7654)
    v=WarpVectorEnv(c,device='cpu')
    times=[];reward_sum=0.;states={}
    try:
        observations,_=v.reset(seed=list(range(seed,seed+envs)))
        # Set up hard, reproducible sensor-confirmation conditions in four
        # differently high worlds WITHOUT changing sensor physics/ground truth.
        altitudes=wp.to_torch(v.s.altitudes)
        positions=wp.to_torch(v.s.positions)
        targets=wp.to_torch(v.s.targets)
        belief=wp.to_torch(v.s.belief)
        for e,z in enumerate((25.,75.,100.,150.)[:envs]):
            positions[e,0]=targets[e,0]
            altitudes[e,0]=z
            gx=min(c.world.map_size_m//c.sensor.grid_cell_m-1,int(targets[e,0,0]/c.sensor.grid_cell_m))
            gy=min(c.world.map_size_m//c.sensor.grid_cell_m-1,int(targets[e,0,1]/c.sensor.grid_cell_m))
            belief[e,0,int(gy*(c.world.map_size_m//c.sensor.grid_cell_m)+gx)]=.99999
        for tick in range(steps):
            continuous=rng.uniform(-.35,.35,(envs,c.world.num_uavs,c.continuous_dim)).astype(np.float32)
            continuous[:,:,-1]=.4
            destinations=np.zeros((envs,c.world.num_uavs),np.int64)
            if tick%2:
                mask=observations['observation'][...,-(c.world.num_uavs+1):].cpu().numpy()
                destinations[:,0]=np.where(mask[:,0,-1]>.5,c.world.num_uavs,0)
            start=time.perf_counter()
            observations,reward,done,_,infos=v.step({'continuous':continuous,'destination':destinations})
            if tick>=2:times.append(time.perf_counter()-start)
            reward_sum+=float(reward.double().sum().item())
            if bool(done.any()):
                mask=(done.cpu().numpy()).astype(bool)
                v.reset(seed=[seed+1000+tick*envs+e if flag else None for e,flag in enumerate(mask)], options={'reset_mask':mask})
                observations=v._buffers[v._buffer]
        h=hashlib.sha256()
        arrays={}
        for name in v.snapshot_state()['arrays']:
            value=wp.to_torch(getattr(v.s,name)).detach().contiguous().cpu().numpy().tobytes()
            arrays[name]=hashlib.sha256(value).hexdigest()
            h.update(name.encode());h.update(value)
        info=dict(kind='real_nooracle_warp_cpu_sensor_and_radio_AB',config_name=config_file,
            num_envs=envs,vector_steps=steps,seed=seed,
            median_vector_step_ms=1000*statistics.median(times),
            transitions_per_second=envs/statistics.median(times),
            reward_sum=reward_sum,
            report_cells=int((wp.to_torch(v.s.report_cell)>=0).sum().item()),
            known_reports=int(wp.to_torch(v.s.known).sum().item()),
            delivered_reports=int(wp.to_torch(v.s.delivered).sum().item()),
            world_sha256=h.hexdigest(),field_sha256=arrays)
        print('NOORACLE_FOV_AB_JSON',json.dumps(info,sort_keys=True),flush=True)
    finally:v.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/matd3-nooracle-adaptive-altitude-lr4-h256x256-20261123.yaml')
    ap.add_argument('--envs',type=int,default=4)
    ap.add_argument('--steps',type=int,default=24)
    ap.add_argument('--seed',type=int,default=44)
    ap.add_argument('--autonomous',action='store_true')
    args=ap.parse_args()
    run(args.config,args.envs,args.steps,args.seed,args.autonomous)
