"""Exercise two seeded full-horizon 3D NoOracle worlds; never trains policy."""
import json,time,math
from pathlib import Path
import numpy as np,torch,warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info


def main():
    c=load_config('configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml')
    c.runtime.num_envs=2;c.runtime.device='cpu'
    c.tracking.enabled=False
    c.validate()
    env=WarpVectorEnv(c,device='cpu')
    started=time.perf_counter()
    top=0.;bottom=150.;entries=[]
    rng=np.random.default_rng(55668)
    try:
        obs,_=env.reset(seed=[44,45])
        for t in range(c.world.max_steps):
            z=wp.to_torch(env.s.altitudes).cpu().numpy()
            top=max(top,float(z.max()));bottom=min(bottom,float(z.min()))
            desired=130. if t<300 else 80. if t<900 else 55.
            actions=rng.uniform(-.55,.55,(2,6,4)).astype(np.float32)
            actions[:,:,2]=np.where(z<desired-3, .7,np.where(z>desired+3,-.7,0.))
            destinations=np.zeros((2,6),dtype=np.int64)
            mask=obs['observation'][...,-7:].detach().cpu().numpy()
            destinations[:,:]=np.where(mask[:,:,-1]>.5,6,0)
            obs,reward,term,_,_=env.step({'continuous':actions,'destination':destinations})
            assert torch.isfinite(reward).all()
            if bool(term.any()):
                break
        for e in range(2):
            info=world_info(env.s,env.p,e)
            entries.append({ 'seed':44+e,'steps':info['step'],
              'targets_found':info['targets_found'],
              'targets_delivered':info['targets_delivered'],
              'hypotheses':info.get('total_hypotheses'),
              'false_confirmed_cells':info.get('false_confirmed_cells'),
              'false_reports_delivered':info.get('false_reports_delivered'),
              'sensor_counts':info['sensing'],
              'altitudes_final':info['altitudes_m'],
              'termination':info['termination_reason'],
              'report_identity':info.get('report_identity')})
        report={'kind':'2_worlds_real_nooracle_warp_full_horizon_scenario_audit_not_training',
                'sciences':'adaptive_lr4_ratio_posterior_099',
                'max_altitude_observed_m':top,'min_altitude_observed_m':bottom,
                'wall_seconds':time.perf_counter()-started,'worlds':entries}
        p=Path('/home/aduy/Downloads/V9_ADAPTIVE_ALTITUDE_LONG_HORIZON_2WORLD_CPU_20261123.json')
        p.write_text(json.dumps(report,indent=2,allow_nan=False))
        print('LONG_HORIZON_CPU_AUDIT',json.dumps(report,allow_nan=False),flush=True)
    finally:env.close()


if __name__=='__main__':main()
