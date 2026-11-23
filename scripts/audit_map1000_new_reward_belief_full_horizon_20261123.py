"""Reproducible full-horizon MAP1000/TX0.6 NoOracle mission audit; no training."""
from pathlib import Path
import argparse
import json,time
import numpy as np
import torch,warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/home/aduy/Downloads/MAP1000_TX06_PRIOR0021875_REWARD02_16WORLD_FULL_HORIZON_20261123.json')
    parser.add_argument('--compare',help='Previously saved reference report (compares all per-world mission metrics, not wall time)')
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError('Refuse to overwrite mission evidence: '+str(path))
    config=load_config('configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml')
    config.runtime.num_envs=16
    config.runtime.device='cpu'
    config.tracking.enabled=False
    v=WarpVectorEnv(config,device='cpu')
    rng=np.random.default_rng(4250)
    began=time.perf_counter()
    try:
        obs,_=v.reset(seed=list(range(400,416)))
        max_z=0.0
        for step in range(1500):
            z=wp.to_torch(v.s.altitudes).cpu().numpy()
            max_z=max(max_z,float(z.max()))
            desired=np.tile(np.asarray([25,50,75,100,125,145])[None,:],(16,1))
            # Move toward different altitude bands and cover random XY areas.
            c=rng.uniform(-.9,.9,(16,6,4)).astype(np.float32)
            c[:,:,2]=np.where(z<desired-2,.7,np.where(z>desired+2,-.7,0))
            c[:,:,3]=.85
            mask=obs['observation'][...,-7:].detach().cpu().numpy()
            destination=np.where(mask[:,:,-1]>.5,6,0).astype(np.int64)
            obs,reward,term,_,_=v.step({'continuous':c,'destination':destination})
            if not bool(torch.isfinite(reward).all()):
                raise FloatingPointError(f'Non-finite reward at {step}')
        records=[]
        for i in range(config.runtime.num_envs):
            info=world_info(v.s,v.p,i)
            records.append(dict(seed=400+i,steps=info['step'],
              targets_found=info['targets_found'],targets_delivered=info['targets_delivered'],
              total_hypotheses=info.get('total_hypotheses',0),
              false_confirmed_cells=info.get('false_confirmed_cells',0),
              false_reports_delivered=info.get('false_reports_delivered',0),
              termination_reason=info['termination_reason'],
              confirmation_source=info.get('confirmation_source')))
        errors=wp.to_torch(v.s.errors).tolist()
        result=dict(kind='real_warp_MAP1000_PRIOR0021875_REWARD02_16world_x1500_stress_not_training',
            scenario_version=10,map_size_m=1000,grid_cell_width_m=25,tx_power_max_w=0.6,seeds='400..415',max_altitude_seen_m=max_z,
            max_hypotheses=max(r['total_hypotheses'] for r in records),
            report_slot_capacity=int(v.p.report_slots),
            total_true_found=sum(r['targets_found'] for r in records),
            total_true_delivered=sum(r['targets_delivered'] for r in records),
            total_false_confirmed_cells=sum(r['false_confirmed_cells'] for r in records),
            any_internal_error=any(x!=0 for x in errors),
            wall_seconds=time.perf_counter()-began,
            records=records)
        result["new_science"]={"belief_prior":float(config.sensor.belief_prior),
             "communication_efficiency":float(config.reward.communication_efficiency),
             "actor_output_normalization":bool(config.matd3.output_normalization),
             "actor_output_regularization":float(config.matd3.actor_output_regularization)}
        if args.compare:
            ref=json.loads(Path(args.compare).read_text())
            fields=('scenario_version','map_size_m','grid_cell_width_m','tx_power_max_w',
                'seeds','max_altitude_seen_m','max_hypotheses','report_slot_capacity',
                'total_true_found','total_true_delivered','total_false_confirmed_cells',
                'any_internal_error','records')
            unequal={k for k in fields if result[k]!=ref[k]}
            if unequal:raise AssertionError('Scenario changes vs original MAP1000 baseline: '+repr(sorted(unequal)))
            result['reference_science_metric_parity']=True
        path.write_text(json.dumps(result,indent=2,allow_nan=False))
        print('MAP1000_NEW_MDP_16x1500_MISSION_OK',json.dumps({k:v for k,v in result.items() if k!='records'},allow_nan=False),flush=True)
        if result['any_internal_error']:raise RuntimeError('Warp internal errors='+repr(errors))
    finally:v.close()


if __name__=='__main__':main()
