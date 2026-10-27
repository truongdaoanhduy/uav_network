import gc,json,os,resource,shutil,sys,time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
import warp as wp
root=Path.cwd();sys.path.insert(0,str(root/'tests'));sys.path.insert(0,str(root))
import test_recovery_checkpoints as checks
from uav_network.experiments.training import train as real_train
result={'gpu_cases':[]}
layout='hybrid'
def gpu_train(c):
 c.runtime.device='cuda';c.runtime.amp_mode='fp16';c.runtime.compile_models=False
 c.runtime.vectorize_critics=True
 c.runtime.replay_device='cuda'if layout=='fullgpu'else'cpu'
 c.runtime.replay_cpu_fraction=0. if layout=='fullgpu'else.75
 if c.pql.enabled:
  c.pql.collector_device='cuda:0';c.pql.learner_device=c.pql.policy_device='cuda:1'
 return real_train(c)
checks.train=gpu_train
for layout in ['hybrid','fullgpu']:
 for algorithm in ['masac','matd3']:
  for pql in [False,True]:
   start=time.perf_counter()
   with TemporaryDirectory(prefix='small-recovery-',dir='/kaggle/working')as folder:
    checks.test_periodic_full_state_before_first_episode_exactly_resumes_and_leaves_no_staging(Path(folder),algorithm,pql)
   row={'layout':layout,'algorithm':algorithm,'pql':pql,'amp':'fp16','compile':False,'seconds':time.perf_counter()-start,'status':'passed'}
   result['gpu_cases'].append(row);print('GPU_CASE',json.dumps(row),flush=True)
   gc.collect();torch.cuda.empty_cache()
# Capacity-only serialization probe: synthetic replay, never a learning result or uploaded artifact.
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim,state_dim
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.algorithms.checkpoints import make_learner
from uav_network.experiments.pql_replay import make_replay
from uav_network.experiments.fixed_checkpoints import save_fixed_checkpoint
from uav_network.experiments.environment_checkpoints import snapshot_environment
from uav_network.experiments.recovery_checkpoints import PeriodicRecovery
from uav_network.experiments.checkpoint_store import file_sha256
c=load_config('configs/comparison-20k-hybrid75-1kb.yaml')
c.runtime.device='cuda';c.runtime.replay_device='cuda';c.runtime.replay_cpu_fraction=0
c.runtime.max_train_seconds=0;c.runtime.algorithm='masac'
c.tracking.resumable_checkpoint_interval_seconds=1e-12;c.tracking.resumable_checkpoint_max_uploads=1
vector=learner=replay=None
with TemporaryDirectory(prefix='capacity-recovery-',dir='/kaggle/working')as folder:
 output=Path(folder)
 try:
  gc.collect();torch.cuda.empty_cache()
  vector=WarpVectorEnv(c,device='cuda:0');obs,_=vector.reset(seed=c.runtime.seed)
  learner=make_learner(c,observation_dim(c),state_dim(c),'cuda:1')
  replay=make_replay(c,tuple(obs['observation'].shape[1:]),state_dim(c),learner.device)
  for value in replay.data.values():value.zero_()
  replay.size=replay.capacity;replay.index=0
  torch.cuda.synchronize()
  n=c.runtime.num_envs;next_seed=c.runtime.seed+n
  progress={'episodes':0,'global_step':0,'collection_steps':0,'next_seed':next_seed,'records':[]}
  measurements={'kind':'synthetic full-capacity serialization only; not a valid learned checkpoint','replay_bytes':replay.allocated_bytes,'disk_free_before':shutil.disk_usage(output).free,'published_to_hf':False}
  started=time.perf_counter()
  def save(policy,*,resume_path):
   learner.save(policy,progress=progress)
   environment=snapshot_environment(vector,torch.zeros(n,dtype=torch.float64,device='cuda:0'),torch.zeros(n,dtype=torch.int64,device='cuda:0'),np.arange(c.runtime.seed,next_seed),next_seed)
   save_fixed_checkpoint(learner,resume_path,progress,replay,np.random.default_rng(c.runtime.seed+1),
     warmup_origin=0,random_warmup=True,next_metrics_at=0,latest={},pending_learning=False,
     action_generator=vector.action_generator,environment=environment)
   return progress
  class Verifier:
   def checkpoint(self,policy,progress,*,resumable=False):
    assert resumable
    path=Path(progress['resume_checkpoint'])
    measurements.update(resume_bytes=path.stat().st_size,policy_bytes=Path(policy).stat().st_size,disk_free_during=shutil.disk_usage(path).free,serialize_seconds=time.perf_counter()-started)
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert payload['training']['replay']['size']==1000000
    assert payload['training']['environment']['world']['arrays']
    del payload;gc.collect()
    hash_start=time.perf_counter();measurements['sha256']=file_sha256(path);measurements['checksum_seconds']=time.perf_counter()-hash_start
    return 'local/verified-capacity:v0'
   def diagnostic(self,values,step):measurements.update(values)
  PeriodicRecovery(c,output,started=started).maybe_save(save,Verifier())
  assert measurements.get('recovery/status')=='uploaded',measurements
  assert not list(output.iterdir()),'Owned15GB stage must be released before any future snapshot'
  measurements['disk_free_after_cleanup']=shutil.disk_usage(output).free
  measurements['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  result['capacity_probe']=measurements;print('CAPACITY_PROBE',json.dumps(measurements),flush=True)
 finally:
  if vector is not None:vector.close()
  del vector,learner,replay
  gc.collect();torch.cuda.empty_cache()
Path('/kaggle/working/recovery-gpu-results.json').write_text(json.dumps(result,indent=2))
print('GPU_RECOVERY_ALL_PASS',flush=True)
