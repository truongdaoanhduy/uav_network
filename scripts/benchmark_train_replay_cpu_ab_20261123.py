"""True B8192 hybrid replay read/write parity and paired CPU sample A/B."""
import argparse,hashlib,json,os,subprocess,sys,time,statistics
from pathlib import Path
import torch
from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
from uav_network.algorithms.replay import transition_shapes

REFERENCE=Path('/home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123')
CANDIDATE=Path('/home/aduy/Documents/NCKH/uav_network_train_speed_opt_20261123')

def worker(n,batch_size,samples):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    # Same shape as NoOracle V9 CT095 (6 UAV, actor obs 228, critic state 830).
    shape=(6,228);state=830;cont=4
    r=HybridMeasuredReplay(n,shape,state,seed=44,hot_device='cpu',cpu_fraction=.75,continuous_dim=cont)
    tensors={}
    for name,dim in transition_shapes(shape,state,cont).items():
        tensors[name]=torch.randn((4096,*dim)) if name!='destination' else torch.randint(0,7,(4096,*dim),dtype=torch.int64)
    class Act:
        continuous=tensors['continuous']
        destination=tensors['destination']
    for i in range((n+4095)//4096):
        count=min(4096,n-i*4096)
        r.add_batch(tensors['observation'][:count],tensors['state'][:count],Act,
                    tensors['reward'][:count],tensors['next_observation'][:count],
                    tensors['next_state'][:count],tensors['terminated'][:count])
    times=[]
    h=hashlib.sha256()
    for i in range(samples):
        t=time.perf_counter()
        batch=r.sample(batch_size)
        times.append(time.perf_counter()-t)
        # Hash outside measured sample() call.
        for key in sorted(batch):
            h.update(key.encode())
            h.update(batch[key].contiguous().numpy().tobytes())
        del batch
    stats=r.stats()
    h.update(r.seen.numpy().tobytes())
    h.update(r.rng.get_state().numpy().tobytes())
    h.update(json.dumps(stats,sort_keys=True,allow_nan=False).encode())
    return {'hash':h.hexdigest(),'times':times,'total_sample_s':sum(times),'stats':stats,
            'sample_count':samples,'batch_size':batch_size,'capacity':n}

def invoke(which,n,batch_size,samples):
    root=REFERENCE if which=='old' else CANDIDATE
    env=os.environ.copy();env['PYTHONPATH']=str(root)
    args=[sys.executable,str(CANDIDATE/'scripts/benchmark_train_replay_cpu_ab_20261123.py'),
          '--worker','--capacity',str(n),'--batch',str(batch_size),'--samples',str(samples)]
    p=subprocess.run(args,cwd=root,env=env,text=True,capture_output=True,timeout=90)
    if p.returncode:raise RuntimeError(which+' failed: '+p.stderr[-1800:])
    rows=[x.split(' ',1)[1] for x in p.stdout.splitlines() if x.startswith('HYBRID_REPLAY_WORKER_JSON ')]
    if len(rows)!=1:raise RuntimeError('Missing worker report '+p.stdout[-1200:])
    return json.loads(rows[0])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--capacity',type=int,default=12288)
    p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--output')
    a=p.parse_args()
    if a.worker:
        print('HYBRID_REPLAY_WORKER_JSON '+json.dumps(worker(a.capacity,a.batch,a.samples)),flush=True);return
    trials=[]
    for i in range(a.pairs):
        for which in (('old','new') if i%2==0 else ('new','old')):
            d=invoke(which,a.capacity,a.batch,a.samples)
            trials.append({'which':which,'repeat':i,**d})
            print('REPLAY_AB',which,i,'sample_s',round(d['total_sample_s'],5),'digest',d['hash'],flush=True)
    hashes={t['hash'] for t in trials}
    if len(hashes)!=1:raise AssertionError('Replay tensor / seen / RNG mismatch: reject')
    med={q:statistics.median(t['total_sample_s'] for t in trials if t['which']==q) for q in ('old','new')}
    result={'kind':'full-shape B8192 CPU hybrid replay sample-only benchmark',
            'capacity':a.capacity,'samples':a.samples,'same_replay_samples_state_rng':True,
            'digest':next(iter(hashes)),'measurements':trials,
            'median_sample_seconds':med,'speedup_percent':(med['old']/med['new']-1)*100}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print('REPLAY_CPU_AB_RESULT',json.dumps({k:result[k] for k in ('same_replay_samples_state_rng','median_sample_seconds','speedup_percent')}),flush=True)

if __name__=='__main__':main()
