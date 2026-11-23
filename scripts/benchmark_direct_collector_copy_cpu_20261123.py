"""Same-device CPU proxy for policy synchronization, not a GPU P2P claim."""
from __future__ import annotations
import hashlib,json,statistics,time
from unittest.mock import patch
import torch
from uav_network.configuration import load_config
from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy
from uav_network.experiments.training import refresh_base_collector


def main():
    torch.set_num_threads(1)
    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=1
    torch.manual_seed(44)
    source=GPUActorOnlyPolicy(c,'cpu')
    old=GPUActorOnlyPolicy(c,'cpu');direct=GPUActorOnlyPolicy(c,'cpu')
    for name, tensor in source.actors.state_dict().items():
        with torch.no_grad():
            if tensor.is_floating_point():tensor.uniform_(-0.23,0.34)
            else:tensor.add_(3)
    times=[]
    for rep in range(5):
        for mode in (('old','direct') if rep%2==0 else ('direct','old')):
            target=old if mode=='old' else direct
            with patch.dict('os.environ', {'UAV_DIRECT_GPU_COLLECTOR_COPY':'true' if mode=='direct' else 'false'}):
                for _ in range(8):refresh_base_collector(target,source)
                start=time.perf_counter()
                for _ in range(250):refresh_base_collector(target,source)
                elapsed=time.perf_counter()-start
            for key,value in source.actors.state_dict().items():
                assert torch.equal(target.actors.state_dict()[key],value),(mode,key)
            times.append((mode,rep,elapsed))
            print('ACTOR_SYNC_AB',mode,rep,elapsed,flush=True)
    med={mode:statistics.median(t[2] for t in times if t[0]==mode) for mode in ('old','direct')}
    print('ACTOR_SYNC_CPU_JSON '+json.dumps({'runs':times,'median_seconds_250_copies':med,
        'copy_speedup_percent':100*(med['old']/med['direct']-1),
        'all_actor_params_and_bn_buffers_equal':True,'caveat':'CPU copies only; GPU interdevice throughput unmeasured'}),flush=True)
if __name__=='__main__':main()
