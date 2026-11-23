import os,sys,json,time
from pathlib import Path
import torch,wandb
from uav_network.configuration import load_config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.vector import make_vector
from uav_network.environment.observations import observation_dim,state_dim
from uav_network.experiments.pql_replay import make_replay
from uav_network.experiments.overflow_capture import attach_overflow_capture,replay_overflow,tensor_stats
from uav_network.experiments.training import seed_runtime
from uav_network.experiments.evidence import json_safe
spec=json.loads(Path('/kaggle/working/diagnostic-spec.json').read_text());out=Path('/kaggle/working/trained-overflow');out.mkdir(exist_ok=True)
run=wandb.init(entity=spec['entity'],project=spec['project'],id=spec['wandb_run_id'],resume='never',mode='online',name='CHECK-TRAINED-MATD3-OVERFLOW-'+spec['wandb_run_id'],group='overflow-diagnostic-20261028',job_type='overflow-diagnostic',tags=['diagnostic','not-production','trained-weights'],config={'stage':'diagnostic','purpose':spec['purpose'],'source_git_commit':spec['commit'],'source_run_id':spec['source_run_id'],'kaggle_kernel':spec['kernel_ref']})
report={**spec,'historical_failed_batch_replayed':False,'optimizer_state_restored':False,'compile_models':False,'comparisons':[],'attempts':[]}
vector=None
try:
 if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Two CUDA devices required')
 report['hardware']=[torch.cuda.get_device_name(i) for i in range(2)]
 api=wandb.Api();source=api.run(spec['entity']+'/'+spec['project']+'/'+spec['source_run_id'])
 path=source.file('failure_state.pt').download(root=str(out/'source'),replace=True).name
 saved=torch.load(path,map_location='cpu',weights_only=True)
 report['source_model_finite']=all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
 config=load_config('configs/comparison-20k-hybrid75-1kb.yaml');config.runtime.algorithm='matd3';config.runtime.amp_mode='fp16';config.runtime.compile_models=False;config.runtime.vectorize_critics=True;config.tracking.enabled=False
 seed_runtime(config)
 vector=make_vector(config,device='cuda:0');obs,_=vector.reset(seed=list(range(44,44+config.runtime.num_envs)))
 replay=make_replay(config,(config.world.num_uavs,observation_dim(config)),state_dim(config),torch.device('cuda:1'))
 for step in range(32):
  action=vector.random_actions(obs['observation']);following,reward,terminated,truncated,infos=vector.step({'continuous':action.continuous,'destination':action.destination})
  replay.add_batch(obs['observation'],obs['state'],action,reward,following['observation'],following['state'],terminated);obs=following
 vector.close();vector=None
 report['fresh_transitions']=len(replay)
 for group,scale in [('actor',32768.),('critic',8192.)]:
  for index in range(8):
   learner=make_learner(config,observation_dim(config),state_dim(config),'cuda:1');learner.load_state_dict(saved['model']);learner.updates=saved['updates']
   state=learner.precision.state_dict();state['scalers'][group]['scale']=scale;learner.precision.load_state_dict(state)
   dest=out/(group+'-'+str(index));observer=attach_overflow_capture(learner,dest,'trained')
   batch=replay.sample(config.algorithm_config.batch_size)
   method=learner.update_actor if group=='actor' else learner.update_critic
   metrics=method(batch)
   row={'group':group,'batch_index':index,'scale_before':scale,'input_finite':all(tensor_stats(v)['finite'] for v in batch.values()),'metrics':json_safe(metrics),'amp':learner.precision.metrics()};report['attempts'].append(row)
   for file in dest.glob('*'):
    if file.suffix in {'.json','.pt'}:run.save(str(file),base_path=str(out),policy='now')
   paths=list(dest.glob('*.pt'))
   if paths:
    comparison=replay_overflow(paths[0],device='cuda:1');comparison['snapshot_file']=str(paths[0].relative_to(out));report['comparisons'].append(comparison)
    (out/'result.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False));run.save(str(out/'result.json'),base_path=str(out),policy='now')
    run.log({'diagnostic/group':group,'diagnostic/overflow_captured':1,'diagnostic/fp32_skipped':comparison['cases']['off']['skipped'],'diagnostic/fp16_skipped':comparison['cases']['fp16']['skipped']})
    break
   del learner
 report['status']='completed'
 run.summary.update({'diagnostic/production_training':False,'diagnostic/comparisons':len(report['comparisons']),'diagnostic/source_model_finite':report['source_model_finite']})
except BaseException as e:
 report['error_type']=type(e).__name__;report['status']='failed';raise
finally:
 if vector is not None:vector.close()
 (out/'result.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False));run.save(str(out/'result.json'),base_path=str(out),policy='now');run.finish(exit_code=1 if report.get('status')=='failed' else 0)
