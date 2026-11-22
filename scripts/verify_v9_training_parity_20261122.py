"""Emit deterministic checksum of genuine NoOracle vector trajectory for commit comparison.

Run in separate interpreters with PYTHONPATH pointing to each checked-out repo:
PYTHONPATH=<repo> python scripts/verify_v9_training_parity_20261122.py --steps 100
The script never bypasses PHY/MAC/sensing or simplifies the map.
"""
from __future__ import annotations
import argparse
import hashlib
import json

import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--envs',type=int,default=4)
    args=ap.parse_args()
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=args.envs
    torch.manual_seed(44)
    vector=WarpVectorEnv(c,device='cpu')
    checksum=hashlib.sha256()
    observed={}
    try:
        obs,_=vector.reset(seed=list(range(44,44+args.envs)))
        def digest(label):
            # Only source-model, scientific simulator fields; ignore new cache.
            names=('positions','targets','obstacles','belief','belief_coarse',
                'visited','positive','sensor_counts','report_cell','visible','known',
                'report_present','report_created','report_hops','delivered',
                'flight_energy','communication_energy','battery','steps',
                'reward','coverage','coverage_auc_weight_sum','event_sequence',
                'packet_counts','diagnostic_counts','observation','state',
                'report_counts','first_delivery','true_report_delay_sum')
            for name in names:
                x=wp.to_torch(getattr(vector.s,name)).contiguous().numpy()
                checksum.update(name.encode());checksum.update(x.tobytes())
            observed[str(label)]=checksum.hexdigest()
        digest('reset')
        for step in range(args.steps):
            action=vector.random_actions(obs['observation'])
            obs,_,done,trunc,_=vector.step({'continuous':action.continuous,'destination':action.destination})
            if step in (0,1,2,5,10,20,40,args.steps-1):digest(step+1)
        print('TRAJECTORY_PARITY_JSON',json.dumps({'envs':args.envs,'steps':args.steps,'hashes':observed},sort_keys=True))
    finally:vector.close()
if __name__=='__main__':main()
