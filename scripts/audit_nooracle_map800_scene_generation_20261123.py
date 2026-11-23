"""Deterministic 800m NoOracle scene reset stress across train/validation/test seeds."""
from collections import Counter
import argparse,json,time
import numpy as np
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv

CFG='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml'


def audit(*,envs=128,per_bank=384):
    c=load_config(CFG)
    c.runtime.device='cpu'
    c.runtime.num_envs=envs
    v=WarpVectorEnv(c,device='cpu')
    counts=Counter()
    total=0
    max_retries=0
    first_failures=[]
    began=time.perf_counter()
    try:
        for base in (44,1000000044,2000000044):
            for start in range(base,base+per_bank,envs):
                seeds=list(range(start,min(start+envs,base+per_bank)))
                if len(seeds)!=envs:
                    raise ValueError('Seed bank should divide the vector size for this audit')
                try:v.reset(seed=seeds)
                except RuntimeError:
                    pass
                errors=wp.to_torch(v.s.errors).cpu().numpy().tolist()
                attempts=wp.to_torch(v.s.generation_attempts).cpu().numpy().tolist()
                for seed,code in zip(seeds,errors):
                    counts[int(code)]+=1
                    if code and len(first_failures)<16:first_failures.append([int(seed),int(code)])
                max_retries=max(max_retries,max(attempts));total+=envs
        result=dict(kind='map800_scene_generation_nooracle_cpu',num_worlds=total,
          counts=dict(counts),first_failures=first_failures,
          max_scene_generation_attempts=max_retries,
          wall_s=round(time.perf_counter()-began,3),grid_cells=v.p.grid_n**2,
          map_size=c.world.map_size_m,
          obstacles=c.world.num_obstacles,
          targets=c.world.num_targets,
          obstacle_radius_range_m=[c.world.obstacle_radius_min_m,c.world.obstacle_radius_max_m])
        print('MAP800_SCENE_STRESS_JSON '+json.dumps(result,sort_keys=True),flush=True)
        if first_failures:raise RuntimeError('Map800 scene generation failed for seeds '+str(first_failures))
        return result
    finally:v.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--envs',type=int,default=128);p.add_argument('--per-bank',type=int,default=384);a=p.parse_args();audit(envs=a.envs,per_bank=a.per_bank)
