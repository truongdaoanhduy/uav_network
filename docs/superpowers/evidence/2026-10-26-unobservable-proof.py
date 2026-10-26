"""Conservative C-space/visibility supersets for the primary seed302 counterexample."""
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from shapely.geometry import Point, box, MultiPoint
from shapely.ops import unary_union
from uav_network.configuration import load_config
from uav_network.environment.world import generate_scenario

c=load_config('configs/comparison-20k-hybrid75-1kb.yaml')
legacy=json.loads(Path(__file__).with_name('2026-10-26-unobservable-layout.json').read_text())
scene=SimpleNamespace(positions=np.asarray(legacy['positions']),obstacles=np.asarray(legacy['obstacles']))
target=np.asarray(legacy['target'])
results=[]
for resolution in (16,32,64,128):
 forbidden=unary_union([Point(*o[:2]).buffer(o[2]+c.world.obstacle_clearance_m,quad_segs=resolution) for o in scene.obstacles])
 free=box(0,0,c.world.map_size_m,c.world.map_size_m).difference(forbidden)
 parts=list(free.geoms) if hasattr(free,'geoms') else [free]
 reachable=unary_union([part for part in parts if any(part.covers(Point(*p)) for p in scene.positions)])
 shadows=[]
 for o in scene.obstacles:
  inscribed=Point(*o[:2]).buffer(o[2],quad_segs=resolution)
  near=np.asarray(inscribed.exterior.coords)[:-1]
  far=target+10000*(near-target)
  shadows.append(MultiPoint(np.r_[near,far]).convex_hull)
 visible_reachable=reachable.difference(unary_union(shadows))
 distance=float(Point(*target).distance(visible_reachable))
 results.append({'quad_segs':resolution,'free_components':len(parts),
   'all_launches_in_reachable_superset':all(reachable.covers(Point(*p)) for p in scene.positions),
   'optimistic_min_visible_reachable_camera_distance_m':distance,
   'camera_radius_m':c.sensor.footprint_radius_m,
   'unobservable_even_in_superset':distance>c.sensor.footprint_radius_m+1e-6})
report={'seed':302,'target_id':6,'target':target.tolist(),'positions':scene.positions.tolist(),
 'obstacles':scene.obstacles.tolist(),'results':results,
 'proof':'Inscribed clearance disks remove less forbidden area: resulting reachable set is optimistic. Shadows of inscribed physical disks block fewer rays: visibility is optimistic. Distance to this visible/reachable superset above camera radius proves absence of a valid vantage point.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'seed':302,'target_id':6,'target':report['target'],'results':results},indent=2))
assert all(r['unobservable_even_in_superset'] for r in results)
