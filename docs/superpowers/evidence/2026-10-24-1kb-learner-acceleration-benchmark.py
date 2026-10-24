import gc, json, math, statistics, time
from copy import deepcopy
from pathlib import Path
import torch
from torch._dynamo.utils import counters
from torch._inductor import metrics as inductor_metrics
from uav_network.configuration import load_config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.algorithms.optimization import compile_models
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.environment.vector import make_vector
from uav_network.experiments.pql_replay import make_replay
from uav_network.experiments.training import resolve_base_devices, seed_runtime

OUT=Path('/kaggle/working')
config=load_config('configs/comparison-20k-hybrid75-1kb.yaml',[
    'runtime.algorithm=masac','runtime.device=cuda','tracking.enabled=false'])
assert config.runtime.num_envs==4096 and config.masac.batch_size==8192 and config.masac.replay_capacity==1_000_000
assert config.runtime.replay_device=='cpu' and config.runtime.replay_cpu_fraction==.75
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))
torch.set_num_threads(1); torch.cuda.init(); seed_runtime(config)
collector_device, learner_device, replay_device=resolve_base_devices(config)
assert (str(collector_device),str(learner_device),str(replay_device))==('cuda:0','cuda:1','cpu')
obs_dim=observation_dim(config); sdim=state_dim(config)
result={'source_commit':'fbbb5569ce6520004bf406de5829fd2ac4696875',
        'scope':'MASAC 1KiB production-shape steady-state learner+hybrid-replay benchmark; no training checkpoints',
        'torch':torch.__version__,'cuda_devices':[torch.cuda.get_device_name(i) for i in range(2)],
        'config':{'num_envs':config.runtime.num_envs,'batch_size':config.masac.batch_size,
                  'replay_capacity':config.masac.replay_capacity,'replay_cpu_fraction':config.runtime.replay_cpu_fraction,
                  'report_bytes':config.network.report_bytes,'payload_bytes':config.network.payload_bytes,
                  'updates_per_step':config.masac.updates_per_step},'rows':[]}
def persist():
    (OUT/'acceleration-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
persist()
vector=make_vector(config,device=collector_device)
try:
    obs,_=vector.reset(seed=config.runtime.seed)
    action=vector.random_actions(obs['observation'])
    following,reward,terminated,truncated,infos=vector.step({'continuous':action.continuous,'destination':action.destination})
    replay=make_replay(config,obs['observation'].shape[1:],sdim,learner_device,measured=False)
    values=(obs['observation'],obs['state'],action,reward,following['observation'],following['state'],terminated)
    fill_started=time.perf_counter()
    batches=math.ceil(replay.capacity/config.runtime.num_envs)
    for _ in range(batches):
        replay.add_batch(*values)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    result['replay_fill']={'batches':batches,'size':len(replay),'seconds':time.perf_counter()-fill_started,
                           'allocated_bytes':replay.allocated_bytes,'cpu_bytes':replay.cpu_allocated_bytes,
                           'hot_bytes':replay.hot_allocated_bytes}
    assert len(replay)==replay.capacity and replay.cpu_allocated_bytes>replay.hot_allocated_bytes>0
finally:
    vector.close()
    del vector
rng_state=replay.rng.get_state().clone()
cases={
 'baseline':('off',False,False),
 'amp_only':('fp16',False,False),
 'compile_only':('off',True,False),
 'fused_only':('off',False,True),
 'all_three':('fp16',True,True),
}
order=['baseline','amp_only','compile_only','fused_only','all_three','all_three','fused_only','compile_only','amp_only','baseline']
for ordinal,label in enumerate(order,1):
    amp,compiled,fused=cases[label]
    c=deepcopy(config); c.runtime.amp_mode=amp; c.runtime.compile_models=compiled; c.runtime.fused_optimizers=fused
    seed_runtime(c); replay.rng.set_state(rng_state.clone())
    torch.cuda.reset_peak_memory_stats(1)
    graphs_before=counters['stats']['unique_graphs']; kernels_before=inductor_metrics.generated_kernel_count
    learner=make_learner(c,obs_dim,sdim,learner_device)
    if compiled:
        compile_models(learner)
    optimizers=[learner.critic_optimizer,*learner.actor_optimizers,learner.alpha_optimizer]
    all_fused=all(group.get('fused') is True for opt in optimizers for group in opt.param_groups)
    warm=time.perf_counter()
    for _ in range(3):
        learner.update(replay.sample(c.masac.batch_size),collect_metrics=False)
    torch.cuda.synchronize(1); warm_s=time.perf_counter()-warm
    started=time.perf_counter(); updates=20
    for _ in range(updates):
        learner.update(replay.sample(c.masac.batch_size),collect_metrics=False)
    torch.cuda.synchronize(1); elapsed=time.perf_counter()-started
    finite=all(torch.isfinite(p).all().item() for p in learner.parameters())
    precision=learner.precision.state_dict()
    row={'ordinal':ordinal,'label':label,'amp_mode':amp,'compile_models':compiled,'fused_optimizers':fused,
         'warmup_s':warm_s,'updates':updates,'timed_s':elapsed,'update_ms':1000*elapsed/updates,
         'updates_per_s':updates/elapsed,'all_adam_groups_fused':all_fused,'parameters_finite':finite,
         'precision':precision,'compiled_unique_graphs':counters['stats']['unique_graphs']-graphs_before,
         'inductor_generated_kernels':inductor_metrics.generated_kernel_count-kernels_before,
         'gpu1_peak_allocated_bytes':torch.cuda.max_memory_allocated(1),
         'gpu1_peak_reserved_bytes':torch.cuda.max_memory_reserved(1)}
    if amp=='fp16':
        assert all(precision['counts'][g]['applied']>0 for g in ('critic','actor','alpha'))
    if fused:
        assert all_fused
    if compiled:
        assert row['compiled_unique_graphs']>0
    assert finite
    result['rows'].append(row); persist(); print(json.dumps(row),flush=True)
    del learner; gc.collect(); torch.cuda.empty_cache(); torch._dynamo.reset()
agg={}
for label in cases:
    rows=[r for r in result['rows'] if r['label']==label]
    agg[label]={'median_update_ms':statistics.median(r['update_ms'] for r in rows),
                'median_warmup_s':statistics.median(r['warmup_s'] for r in rows),
                'repetitions':len(rows)}
base=agg['baseline']['median_update_ms']
for label,stats in agg.items():
    stats['speedup_vs_baseline_x']=base/stats['median_update_ms']
    stats['time_reduction_fraction']=1-stats['median_update_ms']/base
result['aggregate']=agg; persist(); print(json.dumps(agg,indent=2),flush=True)
