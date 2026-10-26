import gc,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from uav_network.configuration import load_config
from uav_network.environment.world import generate_scenario
import warp as wp
from uav_network.environment.gpu.state import allocate_state
from uav_network.environment.gpu.world import reset_geometry
out=Path('outputs/cuda-scenario-audit-20261026/profile-generation-scan.json')
started=time.monotonic(); records=[]
for path in sorted(Path('configs').glob('*.yaml')):
 c=load_config(path); c.validate()
 seeds=sorted({c.runtime.seed,c.runtime.seed+1,c.runtime.seed+2,2000000044,2000000045})
 item={'profile':str(path),'seeds':seeds,'native':[], 'warp_cpu_worlds':2}
 try:
  for seed in seeds:
   s=generate_scenario(c,np.random.default_rng(seed)); item['native'].append({'seed':seed,'ok':True})
  state,params=allocate_state(c,num_envs=2,device='cpu')
  mask=wp.array([1,1],dtype=wp.int32,device='cpu')
  wp.launch(reset_geometry,dim=2,inputs=[state,params,mask],device='cpu')
  errors=state.errors.numpy(); assert not errors.any(), errors.tolist()
  for obstacles,targets in zip(state.obstacles.numpy(),state.targets.numpy()):
   expanded=obstacles[:,2]+c.world.obstacle_clearance_m
   assert np.all(obstacles[:,:2]>expanded[:,None]+1e-9)
   assert np.all(obstacles[:,:2]<c.world.map_size_m-expanded[:,None]-1e-9)
   for i,circle in enumerate(obstacles):
    distances=np.linalg.norm(obstacles[:i,:2]-circle[:2],axis=1)
    assert np.all(distances>expanded[:i]+expanded[i]+2e-9)
   extra=max(0.,c.world.obstacle_clearance_m-c.sensor.footprint_radius_m)
   d=np.linalg.norm(targets[:,None]-obstacles[None,:,:2],axis=2)
   assert np.all(d>obstacles[None,:,2]+extra+1e-9)
  item['ok']=True
  del state,params,mask; gc.collect()
 except Exception as e:
  item.update(ok=False,error_type=type(e).__name__,error=str(e))
 records.append(item)
 out.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,'profiles':records,'all_passed':all(r['ok'] for r in records)},indent=2))
 print(path,item['ok'],flush=True)
assert all(r['ok'] for r in records)
print('PASS',len(records),'profiles',flush=True)
