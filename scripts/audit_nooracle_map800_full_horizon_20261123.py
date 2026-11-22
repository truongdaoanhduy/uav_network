"""16 full-horizon real Warp CPU NoOracle missions; no training or cloud action."""
from pathlib import Path
import json,time
import numpy as np
import torch,warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info


def main():
    config=load_config('configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml')
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
        result=dict(kind='real_warp_map800_16_world_x_1500_step_nooracle_privacy_stress_no_training',
            scenario_version=10,map_size_m=800,grid_cell_width_m=20,seeds='400..415',max_altitude_seen_m=max_z,
            max_hypotheses=max(r['total_hypotheses'] for r in records),
            report_slot_capacity=int(v.p.report_slots),
            total_true_found=sum(r['targets_found'] for r in records),
            total_true_delivered=sum(r['targets_delivered'] for r in records),
            total_false_confirmed_cells=sum(r['false_confirmed_cells'] for r in records),
            any_internal_error=any(x!=0 for x in errors),
            wall_seconds=time.perf_counter()-began,
            records=records)
        filename=Path('/home/aduy/Downloads/MAP800_ORTHOGONAL_COMPILE_16WORLD_FULL_HORIZON_STRESS_20261123.json')
        filename.write_text(json.dumps(result,indent=2,allow_nan=False))
        print('STRESS_JSON',json.dumps(result,allow_nan=False),flush=True)
        if result['any_internal_error']:raise RuntimeError('Warp internal errors='+repr(errors))
    finally:v.close()


if __name__=='__main__':main()
