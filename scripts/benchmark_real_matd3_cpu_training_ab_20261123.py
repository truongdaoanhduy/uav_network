"""Bounded full MATD3 CPU train A/B: exact env/replay/learner output parity."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,statistics,time,uuid
from pathlib import Path
OLD=Path('/home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123')
NEW=Path('/home/aduy/Documents/NCKH/uav_network_train_speed_opt_20261123')

def worker(which,steps,output):
    import torch
    from uav_network.configuration import load_config
    from uav_network.experiments.training import train
    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=8;c.world.max_steps=steps
    c.runtime.episodes=16 if steps > 12 else 24;c.matd3.batch_size=16;c.matd3.learning_starts=16
    c.matd3.replay_capacity=512;c.matd3.updates_per_step=1
    c.runtime.output_dir=str(output);c.runtime.checkpoint_interval=256
    c.runtime.log_interval_steps=128;c.runtime.max_train_seconds=0
    c.runtime.eval_episodes=4;c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=False;c.tracking.online_gpu_eval_enabled=False
    c.tracking.evaluation_interval=0;c.tracking.evaluation_interval_steps=0
    c.tracking.periodic_evaluation_interval_episodes=0
    c.tracking.resumable_checkpoint_interval_seconds=0
    c.validate()
    start=time.perf_counter()
    report=train(c)
    elapsed=time.perf_counter()-start
    saved=torch.load(report['checkpoint'],map_location='cpu',weights_only=True)
    model=saved.get('model',{})
    h=hashlib.sha256()
    for k,v in sorted(model.items()):
        h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
    epis=Path(c.runtime.output_dir,'episodes.jsonl')
    records=[json.loads(x) for x in epis.read_text().splitlines() if x.strip()]
    # Data-based mission metrics are compared exactly, except clock timestamps.
    for x in records:
        for key in tuple(x):
            if 'time' in key.lower() and key!='mission_time_s':x.pop(key,None)
    sig=hashlib.sha256(json.dumps(records,sort_keys=True,default=str).encode()).hexdigest()
    return {'which':which,'elapsed_seconds':elapsed,
      'run_throughput':report['joint_transitions_per_s'],
      'collection_wait_seconds':report['collection_wait_s'],
      'learner_seconds':report['learner_s'],
      'sample_seconds':report['replay_sample_host_s'],
      'optimizer_seconds':report['optimizer_update_host_s'],
      'global_step':report['global_step'],'updates':report['updates'],
      'episodes':report['episodes'],'model_sha256':h.hexdigest(),
      'episodes_sha256':sig,'final_checkpoint':str(report['checkpoint'])}

def invoke(which,args,rep):
    root=OLD if which=='old' else NEW
    output=Path('/home/aduy/Downloads/V9_TRAIN_CPU_AB_REPLAYONLY_WORK_20261123')/(which+'_rep'+str(rep)+'_envs8_h'+str(args.steps))
    if output.exists():raise RuntimeError('Refuse to reuse existing training output path '+str(output))
    env=os.environ.copy();env['PYTHONPATH']=str(root)
    env.update(UAV_DIAGNOSTIC_STORE='off',UAV_FORENSIC_ENABLED='false',
               UAV_TRAINING_ONLY='true',UAV_FAST_TRAINING='false',
               WANDB_MODE='disabled',UAV_DEFERRED_EXPORT_HF='false',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,str(NEW/'scripts/benchmark_real_matd3_cpu_training_ab_20261123.py'),
         '--worker',which,'--output',str(output),'--steps',str(args.steps)]
    p=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=100)
    if p.returncode:
        raise RuntimeError(which+' full-training failed: '+p.stdout[-1200:]+'\n'+p.stderr[-1500:])
    rows=[x.split(' ',1)[1] for x in p.stdout.splitlines() if x.startswith('CPU_REAL_MATD3_WORKER_JSON ')]
    if len(rows)!=1:raise RuntimeError('missing report '+p.stdout[-1000:])
    return json.loads(rows[0])

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['old','new']);p.add_argument('--steps',type=int,default=32);p.add_argument('--pairs',type=int,default=2);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.worker:
        print('CPU_REAL_MATD3_WORKER_JSON '+json.dumps(worker(a.worker,a.steps,a.output)),flush=True)
        return
    rows=[]
    for rep in range(a.pairs):
        for which in (('old','new') if rep%2==0 else ('new','old')):
            data=invoke(which,a,rep)
            rows.append(data)
            print('CPU_REAL_MATD3_AB',rep,which,'tps',round(data['run_throughput'],2),'model',data['model_sha256'][:15],'episodes',data['episodes_sha256'][:12],flush=True)
    model_hashes={r['model_sha256'] for r in rows}
    episode_hashes={r['episodes_sha256'] for r in rows}
    outcome={'kind':'full CPU MATD3 training pipeline smoke (8 env, reduced horizon/batch)',
             'same_model_weights':len(model_hashes)==1,'same_episode_records':len(episode_hashes)==1,
             'runs':rows,'median_tps':{i:statistics.median(r['run_throughput'] for r in rows if r['which']==i) for i in ('old','new')}}
    outcome['speedup_percent']=100*(outcome['median_tps']['new']/outcome['median_tps']['old']-1)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(outcome,indent=2))
    print('CPU_REAL_MATD3_SUMMARY',json.dumps({k:outcome[k] for k in ('same_model_weights','same_episode_records','median_tps','speedup_percent')}),flush=True)
    if not outcome['same_model_weights'] or not outcome['same_episode_records']:
        raise AssertionError('Real learner weights or mission records differ; reject production candidate')
if __name__=='__main__':main()
