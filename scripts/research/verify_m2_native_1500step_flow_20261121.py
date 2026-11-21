"""Deep single-episode M2 flow validation through TTL, power and map updates."""
from pathlib import Path
import sys,time,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.environment.m2_mission import CooperativeM2Env


def run(*,steps=1500,scan_cells=8,seed=44):
    c=Config()
    c.world.motion_dimensions=3;c.world.motion_safety='cbf'
    c.world.map_size_m=1000.;c.world.num_uavs=6
    c.world.num_targets=35;c.world.num_obstacles=25
    c.world.max_steps=steps;c.sensor.model='bayesian';c.sensor.grid_cell_m=25.
    c.network.enabled=True;c.network.peer_range_m=c.network.gcs_range_m=500.
    c.network.payload_bytes=1024;c.energy.enabled=True
    env=CooperativeM2Env(c,max_scan_cells=scan_cells,correlation_strength=0.)
    rng=np.random.default_rng(seed+2345)
    t0=time.perf_counter()
    failed=[];timestamps=[]
    try:
        obs,_=env.reset(seed=seed)
        old_battery=env.battery_j.copy()
        old_gcs=0
        for tick in range(1,steps+1):
            act=Action(rng.uniform(-1,1,size=(6,5)).astype(np.float32),
                       rng.integers(0,7,size=6,dtype=np.int64))
            obs,reward,done,_,info=env.step(act)
            if obs.shape!=(6,255) or not np.isfinite(obs).all():
                raise AssertionError(('invalid_actor_observation',tick,obs.shape))
            if not math.isfinite(reward):raise AssertionError(('nonfinite_reward',tick,reward))
            for node in env.nodes:
                masses=node.target_masses
                if not np.isfinite(masses).all():raise AssertionError(('nonfinite_mass',tick))
                if not np.allclose(masses.sum(axis=1),1.,atol=2e-10):
                    raise AssertionError(('DS_mass_not_normalized',tick))
                if np.min(masses)<-1e-12:raise AssertionError(('negative_DS_mass',tick))
            if (env.battery_j>old_battery+1e-6).any() or (env.battery_j<0).any():
                raise AssertionError(('nonmonotonic_battery',tick))
            old_battery=env.battery_j.copy()
            if info['gcs_events']<old_gcs:raise AssertionError(('GCS_event_count_decreased',tick))
            old_gcs=info['gcs_events']
            if tick%100==0 or done:
                stamps={
                   'step':tick,'elapsed_s':round(time.perf_counter()-t0,2),
                   'gcs_events':int(info['gcs_events']),
                   'gcs_coverage':float(info['gcs_map']['coverage']),
                   'correct_reports':int(info['gcs_verified_correct']),
                   'false_reports':int(info['gcs_verified_false']),
                   'radio_received':int(info['communication']['received_packets']),
                   'gcs_buffer_bytes':int(env.store.bytes_used(6)),
                   'events_total':sum(node.events_seen for node in env.nodes),
                   'battery_min_j':float(env.battery_j.min()),
                   'reward':float(reward),
                }
                print('M2_FLOW',json.dumps(stamps),flush=True)
                timestamps.append(stamps)
            if done:break
        assert info['steps']==steps or not (env.battery_j>0).any()
        assert info['gcs_events']<=sum(node.events_seen for node in env.nodes[:-1])
        result={'result':'PASS','steps':info['steps'],'real_uavnetsim':True,
                'total_wall_s':time.perf_counter()-t0,
                'final':timestamps[-1],
                'timeline':timestamps}
        Path('/tmp/m2_full_1500step_flow_result.json').write_text(json.dumps(result,indent=2))
        print('M2_COMPLETE_1500_FLOW_PASS',flush=True)
        return result
    finally:env.close()

if __name__=='__main__':run()
