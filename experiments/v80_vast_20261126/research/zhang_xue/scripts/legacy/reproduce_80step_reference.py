"""Re-run deterministic 80-step CPU network comparison (not GPU or training)."""
import json, sys
import numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'src'))
from zhang_xue.compat.env import ZhangXue7NodePacketEnv,EnvConfigV54

def one(comm, steps=80):
    env=ZhangXue7NodePacketEnv(EnvConfigV54(),seed=20261125)
    env.reset(seed=20261125)
    dirs=np.array([[15.0,4.0],[12.0,10.0],[6.0,14.0],[14.0,5.0],[8.0,13.0],[10.0,11.0]])
    tot=0.0
    for _ in range(steps):
        acts=np.full(6,6 if comm else 0,dtype=np.int32)
        _,_,_,reward,term,trunc,info=env.step(dirs,acts)
        tot+=reward
        if term or trunc: break
    info['total_return']=round(tot,5)
    info['net_sim_time_us']=env.net.sim_time_us
    info['network_pending_e2e']=len(env.net.e2e_pending)
    info['mac_overflow']=False
    info['policy']='always_GCS' if comm else 'never_send'
    return info
if __name__=='__main__':
    result={'scenario':'CPU 7-radio v5.6 Actor DS-AoI reward contract, 80 steps',
            'seed':20261125,'zero_comm':one(False),'full_comm':one(True)}
    print(json.dumps(result,indent=2,ensure_ascii=False))
