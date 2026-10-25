import gc,json,time
from pathlib import Path
import torch
from uav_network.configuration import load_config
from uav_network.cloud.execution_check import check_execution
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate,load_evaluation_policy
out=Path('/kaggle/working');report={'preflights':[],'runtime_smokes':[]}
def save(): (out/'learner-probe.json').write_text(json.dumps(report,indent=2,allow_nan=False))
for algorithm in ('masac','matd3'):
 for pql in (False,True):
  name=algorithm+('-pql' if pql else '-base')
  c=load_config('configs/comparison-20k-hybrid75-1kb.yaml')
  c.runtime.algorithm=algorithm;c.pql.enabled=pql;c.tracking.enabled=False
  c.runtime.output_dir=str(out/(name+'-preflight'))
  proof=check_execution(c,out/(name+'-execution.json'))
  report['preflights'].append({'case':name,'result':proof});save()
  gc.collect();torch.cuda.empty_cache()
  c=load_config('configs/smoke.yaml')
  c.runtime.algorithm=algorithm;c.pql.enabled=pql;c.tracking.enabled=False
  c.runtime.environment_backend='warp_packet';c.runtime.device='cuda';c.runtime.replay_device='cpu'
  c.runtime.replay_cpu_fraction=.75;c.runtime.num_envs=4;c.runtime.episodes=8
  c.runtime.eval_seed=2000000044;c.runtime.output_dir=str(out/(name+'-train'))
  c.pql.warmup_collection_steps=3
  started=time.monotonic();result=train(c)
  assert result['episodes']>=8 and result['episode_budget_complete'] is True
  assert result['stop_reason']=='episode_budget' and result['updates']>0
  learner,progress,source=load_evaluation_policy(result['checkpoint'])
  assert all(torch.isfinite(p).all() for p in learner.parameters())
  evaluation=evaluate(result['checkpoint'],episodes=1,output_dir=str(out/(name+'-evaluation')),seed=2000000044)
  assert evaluation['episodes']==1 and evaluation['records'][0]['seed']==2000000044
  report['runtime_smokes'].append({'case':name,'training':result,'checkpoint_load_finite':True,
      'evaluation':evaluation,'seconds':time.monotonic()-started})
  save();del learner;gc.collect();torch.cuda.empty_cache()
report['status']='pass';save();print('FOUR_CUDA_LEARNERS_PASS',flush=True)
