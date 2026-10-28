import json,os
from pathlib import Path
import torch,wandb
from uav_network.experiments.overflow_capture import replay_overflow
spec=json.loads(Path('/kaggle/working/diagnostic-spec.json').read_text());out=Path('/kaggle/working/gradient-reference');out.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
if not torch.cuda.is_available() or torch.cuda.device_count()!=2:raise RuntimeError('Two CUDA devices required')
run=wandb.init(entity=spec['entity'],project=spec['project'],id=spec['wandb_run_id'],resume='never',mode='online',name='CHECK-OVERFLOW-GRADIENT-REFERENCE-'+spec['wandb_run_id'],group='overflow-diagnostic-20261028',job_type='overflow-diagnostic',tags=['diagnostic','not-production','fp32-reference'],config={'stage':'diagnostic','source_git_commit':spec['commit'],'purpose':spec['purpose'],'kaggle_kernel':spec['kernel_ref']})
results=[]
try:
 api=wandb.Api()
 for source_id,name,label in [('128d227d','matd3-pql/overflow-critic-critic.pt','matd3-pql'),('128d227d','masac-pql/overflow-critic-critic.pt','masac-pql'),('c031f202','critic-0/overflow-trained-critic.pt','trained-matd3-critic')]:
  source=api.run(spec['entity']+'/'+spec['project']+'/'+source_id);path=source.file(name).download(root=str(out/'source'/source_id),replace=True).name
  result=replay_overflow(path,device='cuda:1');result.update(source_wandb_run_id=source_id,source_file=name,label=label)
  for item in result['cases']['off']['parameter_gradient_stats'].values():
   item['exceeds_fp16_at_source_scale']=item.get('abs_max_at_source_scale',0)>65504.
  results.append(result);dest=out/(label+'.json');dest.write_text(json.dumps(result,indent=2,allow_nan=False));run.save(str(dest),base_path=str(out),policy='now')
  run.log({'diagnostic/case':label,'diagnostic/fp32_loss_finite':int(result['cases']['off']['losses_finite']),'diagnostic/fp16_skipped':result['cases']['fp16']['skipped'],'diagnostic/fp32_reference_parameters_exceeding_fp16':sum(x['exceeds_fp16_at_source_scale'] for x in result['cases']['off']['parameter_gradient_stats'].values())})
 (out/'result.json').write_text(json.dumps(results,indent=2,allow_nan=False));run.save(str(out/'result.json'),base_path=str(out),policy='now');run.summary['diagnostic/cases']=len(results)
finally:run.finish()
