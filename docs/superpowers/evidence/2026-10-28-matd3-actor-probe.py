import json,os,time
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import torch,wandb
from torch.utils._python_dispatch import TorchDispatchMode
from uav_network.configuration import load_config,Config,_apply
from uav_network.algorithms.checkpoints import make_learner
from uav_network.algorithms.optimization import compile_models
from uav_network.environment.vector import make_vector
from uav_network.environment.observations import observation_dim,state_dim
from uav_network.experiments.pql_replay import make_replay
from uav_network.experiments.pql_workers import inference_copy
from uav_network.experiments.training import seed_runtime
from uav_network.experiments.overflow_capture import attach_overflow_capture,replay_overflow,tensor_stats
from uav_network.experiments.evidence import learner_telemetry
spec=json.loads(Path('/kaggle/working/diagnostic-spec.json').read_text());out=Path('/kaggle/working/actor-late-probe');out.mkdir(exist_ok=True)
run=wandb.init(entity=spec['entity'],project=spec['project'],id=spec['wandb_run_id'],resume='never',mode='online',name='CHECK-MATD3-ACTOR-LATE-'+spec['wandb_run_id'],group='actor-overflow-investigation-20261028',job_type='actor-overflow-diagnostic',tags=['diagnostic','not-production','actor','compiled-comparison'],config={'stage':'diagnostic',**spec})
report={**spec,'historical_failed_batch_replayed':False,'optimizer_restored_from_historical_run':False,'attempts':[],'comparisons':[],'trace':None,'found_actor_overflow':False,'status':'initializing','policy_rollout_learning':False}
started=time.monotonic();vector=None

def persist():
 (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False));run.save(str(out/'result.json'),base_path=str(out),policy='now')

class FirstNonfinite(TorchDispatchMode):
 def __init__(self):super().__init__();self.first=None;self.uninspected=0
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  result=func(*args,**(kwargs or {}))
  def tensors(v):
   if isinstance(v,torch.Tensor):return [v]
   if isinstance(v,(list,tuple)):return [t for x in v for t in tensors(x)]
   if isinstance(v,dict):return [t for x in v.values() for t in tensors(x)]
   return []
  if self.first is None:
   try:
    bad=[x for x in tensors(result) if x.is_floating_point() and not bool(torch.isfinite(x).all())]
    if bad:self.first={'op':str(func),'inputs':[tensor_stats(x) for x in tensors((args,kwargs or {}))],'nonfinite_outputs':[tensor_stats(x) for x in bad]}
   except RuntimeError:self.uninspected+=1
  return result

try:
 if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Two CUDA devices required')
 report['hardware']=[torch.cuda.get_device_name(i) for i in range(2)]
 source=wandb.Api().run(spec['entity']+'/'+spec['project']+'/'+spec['source_run_id'])
 path=source.file('failure_state.pt').download(root=str(out/'source'),replace=True).name
 saved=torch.load(path,map_location='cpu',weights_only=True);report['source_model_finite']=all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
 config=load_config('configs/comparison-20k-hybrid75-1kb.yaml');config.runtime.algorithm='matd3';config.runtime.amp_mode='fp16';config.runtime.compile_models=False;config.runtime.vectorize_critics=True;config.tracking.enabled=False;seed_runtime(config)
 learners={}
 for compiled in [False,True]:
  cc=deepcopy(config);cc.runtime.compile_models=compiled
  learner=make_learner(cc,observation_dim(cc),state_dim(cc),'cuda:1');learner.load_state_dict(saved['model']);learner.updates=saved['updates'];learner.actor_updates=saved['updates']//2
  if compiled:compile_models(learner)
  learners['compiled' if compiled else 'eager']=learner
 collector=inference_copy(learners['eager'],'cuda:0')
 vector=make_vector(config,device='cuda:0');count=config.runtime.num_envs;next_seed=44+count;obs,_=vector.reset(seed=list(range(44,next_seed)))
 replay=make_replay(config,(config.world.num_uavs,observation_dim(config)),state_dim(config),torch.device('cuda:1'))
 milestones={32,128,256,512,768,1024,1536,1792};snapshots=[]
 for step in range(1,1793):
  if time.monotonic()-started>=1200:break
  action=collector.act_batch(obs['observation'],step=14188544+(step-1)*count,return_tensors=True)
  following,reward,terminated,truncated,infos=vector.step({'continuous':action.continuous,'destination':action.destination})
  replay.add_batch(obs['observation'],obs['state'],action,reward,following['observation'],following['state'],terminated);obs=following
  report['rollout_steps']=step;report['fresh_transitions']=len(replay);report['elapsed_seconds']=time.monotonic()-started
  if step in milestones:
   report['status']='probing_actor';persist()
   for batch_index in range(2):
    batch=replay.sample(config.algorithm_config.batch_size)
    for scale in [32768.,65536.]:
     for mode,learner in learners.items():
      learner.load_state_dict(saved['model']);learner.actor_updates=saved['updates']//2
      precision=learner.precision.state_dict();precision['scalers']['actor']['scale']=scale;precision['scalers']['actor']['_growth_tracker']=0;precision['counts']['actor']={'attempted':0,'applied':0,'skipped':0};learner.precision.load_state_dict(precision)
      dest=out/(str(step)+'-'+str(batch_index)+'-'+mode+'-'+str(int(scale)));observer=attach_overflow_capture(learner,dest,'actor-probe')
      metrics=learner.update_actor(batch)
      row={'rollout_step':step,'batch_index':batch_index,'compiled':mode=='compiled','scale_before':scale,'active_fraction':float((batch['observation'][:,:,9]>.5).float().mean()),'input_finite':all(tensor_stats(v)['finite'] for v in batch.values()),'metrics':learner_telemetry(metrics)};report['attempts'].append(row)
      run.log({'diagnostic/rollout_step':step,'diagnostic/compiled':int(row['compiled']),'diagnostic/scale':scale,'diagnostic/actor_skipped':learner.precision.counts['actor']['skipped'],'diagnostic/active_fraction':row['active_fraction']})
      if observer.events:
       report['found_actor_overflow']=True
       for file in dest.glob('*'):
        if file.suffix in {'.pt','.json'}:run.save(str(file),base_path=str(out),policy='now')
       snapshots+=list(dest.glob('*.pt'))
     if report['found_actor_overflow']:break
    if report['found_actor_overflow']:break
   persist()
   if report['found_actor_overflow']:break
  done=terminated|truncated;indices=list(infos['device_episode_infos'])
  if indices:
   seeds=[None]*count
   for index in indices:seeds[index]=next_seed;next_seed+=1
   obs,_=vector.reset(seed=seeds,options={'reset_mask':done.cpu().numpy()})
  if step%64==0:
   report['status']='collecting_late_policy_data';persist();run.log({'diagnostic/rollout_step':step,'diagnostic/elapsed_seconds':report['elapsed_seconds']})
 if vector is not None:vector.close();vector=None
 del replay,collector,learners
 torch.cuda.empty_cache()
 for snapshot in snapshots:
  comparison=replay_overflow(snapshot,device='cuda:1');comparison['snapshot_file']=str(snapshot.relative_to(out));report['comparisons'].append(comparison)
 if snapshots:
  payload=torch.load(snapshots[0],map_location='cpu',weights_only=True);cc=Config();_apply(cc,payload['config']);cc.runtime.compile_models=False
  learner=make_learner(cc,payload['batch']['observation'].shape[-1],payload['batch']['state'].shape[-1],'cuda:1');learner.load_state_dict(payload['model']);learner.precision.load_state_dict(payload['precision_before']);learner.updates=payload['updates'];learner.actor_updates=payload['actor_updates'];batch={k:v.to('cuda:1') for k,v in payload['batch'].items()}
  torch.set_rng_state(payload['rng']['cpu'])
  if 'cuda' in payload['rng']:torch.cuda.set_rng_state(payload['rng']['cuda'],'cuda:1')
  trace=FirstNonfinite();backward=torch.autograd.backward
  def traced_backward(*args,**kwargs):
   with trace:return backward(*args,**kwargs)
  with patch('torch.autograd.backward',traced_backward):metrics=learner.update_actor(batch)
  report['trace']={'mode':'eager_backward_only','first_observed_nonfinite_op':trace.first,'uninspected_operations':trace.uninspected,'metrics':learner_telemetry(metrics)}
 report['status']='completed';persist();run.summary.update({'diagnostic/found_actor_overflow':report['found_actor_overflow'],'diagnostic/rollout_steps':report.get('rollout_steps',0),'diagnostic/comparisons':len(report['comparisons']),'diagnostic/production_training':False})
except BaseException as error:
 report['status']='failed';report['error_type']=type(error).__name__;persist();raise
finally:
 if vector is not None:vector.close()
 run.finish(exit_code=1 if report['status']=='failed' else 0)
