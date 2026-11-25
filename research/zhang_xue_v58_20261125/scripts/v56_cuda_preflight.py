#!/usr/bin/env python3
"""Experimental CUDA allocator/7-radio Warp smoke gate; exit 2 when no CUDA/T4.

Does NOT train, access user credentials, or claim CPU-Warp parity. Review
experimental Warp event kernel and private environment provenance before use.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from v56_warp_7radio_mac import WarpSevenRadioNetwork,backing_bytes_per_environment

def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--envs',type=int,default=32)
    p.add_argument('--allow-experimental',action='store_true',help='explicit authorization for experimental MAC kernel')
    a=p.parse_args()
    out={'stage':'experimental_v56_cuda_gate3','device':a.device,'num_envs':a.envs,
         'soa_bytes_per_env':backing_bytes_per_environment(),'cpu_warp_parity':'NOT_VERIFIED'}
    try:
        import torch,warp as wp
        if a.envs not in (1,2,32,256,1024,2048):raise RuntimeError('Unsupported batch size; use 1,2,32,256,1024,2048')
        if not a.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA driver/GPU unavailable; not a Tesla T4 validation')
        dev=int(a.device.split(':',1)[1]) if ':' in a.device else 0
        props=torch.cuda.get_device_properties(dev)
        out['gpu']=props.name;out['total_memory']=props.total_memory
        if not a.allow_experimental:raise RuntimeError('Add --allow-experimental ONLY for a controlled CUDA smoke test')
        free_start,total=torch.cuda.mem_get_info(dev)
        start=time.perf_counter()
        env=WarpSevenRadioNetwork(a.envs,device=a.device,seed=44)
        gain=np.zeros((a.envs,7,7),np.float32)
        # Only a single direct link; zero-channel worlds verify world separation.
        gain[0,0,6]=gain[0,6,0]=1e-8
        env.set_channel_gain(gain)
        env.run(500_000,experimental=True)
        snap=env.snapshot(('overflow','event_count','now','route_hop'))
        if np.any(snap['overflow']):raise RuntimeError('Warp max-events overflow')
        if snap['route_hop'][0,0,6]!=6:raise RuntimeError('Direct HELLO route unavailable')
        if a.envs>1 and np.any(snap['route_hop'][1:,:,6]>=0):raise RuntimeError('Cross-environment route leakage')
        free_end,_=torch.cuda.mem_get_info(dev)
        out.update(status='CUDA_WARP_SMOKE_PASSED_NOT_FULL_PARITY',wall_seconds=round(time.perf_counter()-start,3),
            mem_used_delta_bytes=max(0,free_start-free_end),event_total=int(snap['event_count'].sum()))
        print(json.dumps(out,indent=2));return 0
    except Exception as ex:
        out.update(status='GATE_BLOCKED',reason=str(ex))
        print(json.dumps(out,indent=2));return 2

if __name__=='__main__':raise SystemExit(run())
