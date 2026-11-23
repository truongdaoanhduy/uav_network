"""CPU Warp mission-horizon integrity stress (does not claim CUDA verification)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def one_batch(config, first_seed):
    count=config.runtime.num_envs
    env=WarpVectorEnv(config,device='cpu')
    seeds=list(range(first_seed,first_seed+count))
    start=time.perf_counter()
    try:
        obs,_=env.reset(seed=seeds)
        finished=torch.zeros(count,dtype=torch.bool)
        last_infos=[None]*count
        for step in range(config.world.max_steps):
            action=env.random_actions(obs['observation'])
            action.continuous[finished]=0
            action.destination[finished]=0
            obs,reward,terminated,truncated,infos=env.step({
                'continuous':action.continuous,'destination':action.destination})
            assert torch.isfinite(torch.as_tensor(reward)).all(), ('nonfinite_reward',step)
            newly=(terminated|truncated)&(~finished)
            for e in torch.nonzero(newly).flatten().tolist():
                last_infos[e]=infos['device_episode_infos'][e]
            finished|=(terminated|truncated)
            if step%150==0:
                for name in ('positions','battery','belief'):
                    assert torch.isfinite(wp.to_torch(getattr(env.s,name))).all(), ('nonfinite',name,step)
                assert not bool((wp.to_torch(env.s.battery)<-1e-4).any())
            if finished.all():break
        assert finished.all(),f'Incomplete worlds at {step}'
        assert all(entry is not None for entry in last_infos)
        record=[]
        for seed,info in zip(seeds,last_infos):
            found=int(info['targets_found']); delivered=int(info['targets_delivered'])
            assert 0<=delivered<=found<=config.world.num_targets
            assert bool(info['success'])==(delivered==config.world.num_targets)
            assert 0<=int(info['step'])<=config.world.max_steps
            assert 0<=float(info['coverage_percent'])<=100
            network=info['network']; assert network['generated_packets']>=network['received_packets']>=0
            record.append({'seed':seed,'steps':int(info['step']),
                 'found':found,'delivered':delivered,'false_confirmation':int(info['sensing']['false_confirmation']),
                 'generated_packets':int(network['generated_packets']),
                 'received_packets':int(network['received_packets']),
                 'coverage_percent':round(float(info['coverage_percent']),2)})
        return {'elapsed_seconds':round(time.perf_counter()-start,3),'results':record}
    finally:env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--envs',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('/tmp/uav_v7_horizon_stress_20261122.json'))
    a=parser.parse_args()
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=a.envs;c.tracking.enabled=False
    assert c.network.enabled and c.sensor.report_cell_hypotheses
    assert c.world.max_steps==1500 and c.world.num_targets==35 and c.world.num_uavs==6
    result=[]
    for i in range(a.repeats):
        entry={'round':i+1,**one_batch(c,44+100*i)}
        result.append(entry)
        print('HORIZON_ROUND',i+1,json.dumps(entry,ensure_ascii=False),flush=True)
        a.output.write_text(json.dumps({'ok':True,'rounds':result},indent=2))
    print('HORIZON_STRESS_PASSED',len(result),a.envs,c.world.max_steps,flush=True)

if __name__=='__main__':main()
