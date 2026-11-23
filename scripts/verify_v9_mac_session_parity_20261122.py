"""Bitwise cross-branch MAC parity after injecting realistic occupied report slots.

Run independently with PYTHONPATH set to the old optimized source and to this
new version; all report/event internals are hashed, not just success metrics.
"""
from __future__ import annotations
import argparse,hashlib,json
import torch,warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=40)
    ap.add_argument('--envs',type=int,default=4);ap.add_argument('--device',default='cpu',choices=['cpu','cuda']);args=ap.parse_args()
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.num_envs=args.envs;c.runtime.device=args.device
    torch.manual_seed(44);v=WarpVectorEnv(c,device=args.device)
    h=hashlib.sha256();samples={}
    try:
        obs,_=v.reset(seed=list(range(440,440+args.envs)))
        p=v.p;s=v.s
        cell=wp.to_torch(s.report_cell);present=wp.to_torch(s.report_present)
        origin=wp.to_torch(s.report_origin);created=wp.to_torch(s.report_created)
        order=wp.to_torch(s.report_order);known=wp.to_torch(s.known)
        # Legitimate NoOracle observed-cell FIFO prefix (false-positive
        # hypotheses are allowed; never use true target IDs for reports).
        for t in range(3):
            cell[:,t]=t+1
            present[:,t,t]=1;origin[:,t,t]=t;created[:,t,t]=0.0;order[:,t,t]=t+1
            known[:,t,t]=1
        # An active fragmented source session is already pending at the
        # beginning of the next policy tick, and one invalid session must be
        # evicted. This exercises clear_pending's bookkeeping, not just empty.
        sess=wp.to_torch(s.session_active);sess_created=wp.to_torch(s.session_created)
        sess_origin=wp.to_torch(s.session_origin);sess_pending=wp.to_torch(s.session_pending_mask)
        sess_hops=wp.to_torch(s.session_hops)
        for dest in range(p.n,p.n+1):
            idx=(0*p.report_slots+0)*(p.n+1)+dest
            sess[:,idx]=1;sess_created[:,idx]=0.0;sess_origin[:,idx]=0;sess_hops[:,idx]=0
            sess_pending[:,idx]=3
            invalid=(2*p.report_slots+1)*(p.n+1)+dest
            sess[:,invalid]=1;sess_created[:,invalid]=0.0;sess_origin[:,invalid]=2;sess_hops[:,invalid]=0
            sess_pending[:,invalid]=7
        def digest(key):
            # Entire *original* scientific state, including report/session
            # queue, RF events and so on; ignore only cache outside WorldState.
            for name,val in sorted(vars(s).items()):
                if isinstance(val,wp.array):
                    arr=wp.to_torch(val).contiguous().cpu().numpy();h.update(name.encode());h.update(arr.tobytes())
            samples[str(key)]=h.hexdigest()
        digest('initial')
        for it in range(args.steps):
            action=v.random_actions(obs['observation'])
            destination=torch.full((args.envs,p.n),p.n,dtype=torch.int64,device=v.device)
            obs,_,_,_,_=v.step({'continuous':action.continuous,'destination':destination})
            if it in (0,1,2,4,9,19,args.steps-1):digest(it+1)
        print('MAC_SESSION_PARITY_JSON',json.dumps({'steps':args.steps,'envs':args.envs,'hashes':samples},sort_keys=True))
    finally:v.close()
if __name__=='__main__':main()
