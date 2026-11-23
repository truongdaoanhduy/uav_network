"""Identical-sample pinned/non-pinned 75/25 hybrid replay A/B on T4 learner GPU.

For a real decision run CUDA:1 with B8192 and capacity16384 or greater.
Full scientific replay in production remains 1M entries; the smaller pilot
contains *the same shapes*, 75/25 storage, and batch size but is NOT a full
training throughput measure. Includes full minibatch SHA256 and RNG parity.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time,statistics
import torch
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim,state_dim
from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay

CFG='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--staging',choices=['off','on'],required=True)
    p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--capacity',type=int,default=16384)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--samples',type=int,default=6)
    a=p.parse_args()
    if a.batch<1 or a.capacity<a.batch or a.samples<3 or a.warmup<1:
        raise ValueError('Invalid bounded replay pilot shape')
    device=torch.device(a.device)
    if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()<2):
        raise RuntimeError('Requires Kaggle T4x2 for CUDA benchmark')
    torch.set_num_threads(1)
    os.environ['UAV_HYBRID_PINNED_STAGING']='1' if a.staging=='on' else '0'
    c=load_config(CFG)
    r=HybridMeasuredReplay(a.capacity,(c.world.num_uavs,observation_dim(c)),state_dim(c),
                           seed=44,hot_device=device,cpu_fraction=.75,continuous_dim=c.continuous_dim)
    # Data is deterministic, nonzero and differs across rows. Global RNG
    # belongs exclusively to the replay and is identically seeded per process.
    for group,start in ((r.cpu_data,0),(r.hot_data,r.cpu_capacity)):
        for name,t in group.items():
            t.zero_()
            values=torch.arange(start,start+len(t),device=t.device,dtype=torch.int64)%191
            t.reshape(len(t),-1)[:,0].copy_(values.to(t.dtype))
    r.size=r.inserted=r.capacity;r.collections=1;r.index=0
    r.born[:]=1;r.known_rows=r.size
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for _ in range(a.warmup):
        _=r.sample(a.batch)
        sync()
    samples=[];digest=None
    for step in range(a.samples):
        sync()
        t0=time.perf_counter()
        batch=r.sample(a.batch)
        sync()
        samples.append(time.perf_counter()-t0)
        if step==0:
            # Hash *all* B8192 fields including 75/25 mixed CPU and GPU
            # samples. D2H hashing is outside the sample timing measurement.
            h=hashlib.sha256()
            for key in sorted(batch):
                h.update(key.encode())
                h.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
            digest=h.hexdigest()
    rng_sha=hashlib.sha256(r.rng.get_state().numpy().tobytes()).hexdigest()
    seen_sha=hashlib.sha256(r.seen.numpy().tobytes()).hexdigest()
    output={'kind':'real_MAP1000_B8192_hybrid_replay_cuda_staging_AB',
        'device':str(device),'staging':a.staging,'batch':a.batch,'capacity':a.capacity,
        'cpu_fraction':.75,'iterations':a.samples,'warmup':a.warmup,
        'sample_seconds':samples,'median_sample_seconds':statistics.median(samples),
        'min_sample_seconds':min(samples),'max_sample_seconds':max(samples),
        'all_sampled_tensor_sha256':digest,'rng_state_sha256':rng_sha,
        'seen_bitmap_sha256':seen_sha,'replay_allocated_bytes':r.allocated_bytes,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
    }
    print('HYBRID_REPLAY_CUDA_AB_JSON',json.dumps(output,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
