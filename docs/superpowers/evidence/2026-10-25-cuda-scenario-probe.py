import json,time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.metrics import world_info
from uav_network.experiments.feasibility import oracle_action

out=Path('/kaggle/working'); report={'scope':'Primary 1KiB scenario, no learned policy or scientific retuning','checks':[]}
base=load_config('configs/comparison-20k-hybrid75-1kb.yaml')
base.tracking.enabled=False
def save(): (out/'scenario-probe.json').write_text(json.dumps(report,indent=2,allow_nan=False))
def invariants(v,obs,reward):
 assert obs['observation'].is_cuda and obs['state'].is_cuda
 assert bool(torch.isfinite(obs['observation']).all()) and bool(torch.isfinite(obs['state']).all())
 assert bool(torch.isfinite(reward).all())
 positions=v.s.positions.numpy(); battery=v.s.battery.numpy(); present=v.s.report_present.numpy()
 assert np.isfinite(positions).all() and (positions>=-1e-6).all() and (positions<=v.config.world.map_size_m+1e-6).all()
 assert np.isfinite(battery).all() and (battery>=0).all() and (battery<=v.config.energy.battery_capacity_j+1e-6).all()
 assert (present.sum(axis=2)*v.config.network.report_bytes<=v.config.network.buffer_bytes).all()
 assert (v.s.delivered.numpy().sum(axis=1)<=v.config.world.num_targets).all()
 assert not v.s.errors.numpy().any()
for count,steps in ((8,1800),(4096,16)):
 c=deepcopy(base);c.runtime.num_envs=count
 v=WarpVectorEnv(c,device='cuda:0');started=time.monotonic()
 try:
  obs,_=v.reset(seed=44); completed=0;next_seed=100044
  for step in range(steps):
   a=v.random_actions(obs['observation'])
   obs,reward,done,truncated,infos=v.step({'continuous':a.continuous,'destination':a.destination})
   if step%25==0 or step==steps-1:invariants(v,obs,reward)
   if bool(done.any()):
    indices=torch.nonzero(done).flatten().cpu().tolist(); seeds=[None]*count
    for i in indices:seeds[i]=next_seed;next_seed+=1
    completed+=len(indices)
    obs,_=v.reset(seed=seeds,options={'reset_mask':done.cpu().numpy()})
  torch.cuda.synchronize()
  report['checks'].append({'kind':'random_scenario_stress','num_envs':count,'vector_steps':steps,
    'joint_transitions':count*steps,'max_steps':c.world.max_steps,'completed_episodes':completed,
    'allocated_environment_bytes':v.allocated_bytes,'device':str(v.device),'seconds':time.monotonic()-started,'status':'pass'})
  save()
 finally:v.close()
# Privileged engineering controller: success proves a feasible trace; failure does not prove impossibility.
report['oracle_records']=[]
for seed in (44,2000000044,2000000045):
 c=deepcopy(base);c.runtime.num_envs=1
 v=WarpVectorEnv(c,device='cuda:0')
 try:
  obs,_=v.reset(seed=seed); started=time.monotonic()
  for step in range(c.world.max_steps):
   s=v.s
   view=SimpleNamespace(config=c,vector=v,positions=s.positions.numpy()[0],velocities=s.velocities.numpy()[0],
       targets=s.targets.numpy()[0],obstacles=s.obstacles.numpy()[0],gcs=s.gcs.numpy()[0])
   a=oracle_action(view,observation=obs['observation'][0].cpu().numpy())
   obs,reward,done,truncated,infos=v.step({'continuous':a.continuous[None],'destination':a.destination[None]})
   if step%50==0:invariants(v,obs,reward)
   if bool(done[0]):break
  info=world_info(v.s,v.p,0)
  report['oracle_records'].append({'seed':seed,'device':str(v.device),'steps':step+1,'info':info,
    'seconds':time.monotonic()-started,'not_a_policy_baseline':True})
  save()
 finally:v.close()
report['status']='pass';save();print(json.dumps(report,indent=2,allow_nan=False))
