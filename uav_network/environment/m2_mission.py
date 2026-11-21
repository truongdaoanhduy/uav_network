"""Native multi-UAV 3D flight with 2D target belief mapping via GCS.

Components:
* Actual 3D motion + camera FoV/occlusion from the UAV simulator
* Explicit correlated sensor, shared across UAVs and time, exactly simulated
* Independent UAV local DS maps + source-provenanced cooperative SPRT
* Real Native UavNetSim packet transport (no shared-map GCS shortcut)
* Target-only 2D belief F/E/U, coverage/freshness and report corrections
* Observation-only CTDE Critic and policy-controlled XYZ/destination/TX
* Oracle occupancy ONLY inside the sensor simulator and evaluator/reward

This CPU reference is NOT the 4096-env Warp production implementation.
"""
from __future__ import annotations
import math
import numpy as np

from .world import generate_scenario
from .xyz_geometry import prepare_scene, camera_clear
from .energy import funded_motion
from .bayesian import sensor_profile, cell_coverage
from .env import Action
from .correlated_mapping import CorrelatedBinarySensor, CellEvidence, Event
from ..communication.m2_protocol import CooperativeWorldMap, MultiLayerMapMetrics
from ..communication.m2_network import M2PacketStore, M2Radio


class CooperativeM2Env:
    """One native 3D mission, with actual packet deliveries and map snapshots."""
    def __init__(self,config,*,correlation_strength=.5,ds_threshold=.95,
                 sprt_alpha=.05,sprt_beta=.05,max_scan_cells=None,
                 map_buffer_bytes=65536):
        config.runtime.m2_enabled=True  # Only this opt-in environment uses 5D policy actions.
        self.config=config.validate()
        if self.config.motion_dim != 3:
            raise ValueError("M2 experiment requires 3D flight")
        if self.config.world.motion_safety=="dynamic_window":
            raise ValueError("Native XYZ dynamic_window is retired; choose CBF or reject")
        if (not 0<=correlation_strength<=1 or
                (max_scan_cells is not None and max_scan_cells<=0)):
            raise ValueError("invalid M2 sensor correlation or scan capacity")
        self.correlation_strength=float(correlation_strength)
        self.ds_threshold=ds_threshold
        self.sprt_alpha=sprt_alpha;self.sprt_beta=sprt_beta
        self.max_scan_cells=None if max_scan_cells is None else int(max_scan_cells)
        self.map_buffer_bytes=int(map_buffer_bytes)
        self.radio=None;self.nodes=None;self.store=None
        self.steps=0;self._done=True;self._sequence=0

    @property
    def time_s(self):return self.steps*self.config.world.dt_s

    @property
    def gcs_map(self):return self.nodes[-1]

    def close(self):
        if self.radio is not None:
            self.radio.close();self.radio=None

    def _cell_xy(self,xy):
        size=self.config.sensor.grid_cell_m
        gx=min(self.grid_n-1,max(0,int(xy[0]/size)))
        gy=min(self.grid_n-1,max(0,int(xy[1]/size)))
        return gy*self.grid_n+gx

    def _scene_with_single_target_per_cell(self,seed,options):
        # Freeze the core legacy generator: the new experiment rejects duplicate
        # Bernoulli cells without changing previous checkpoint scene distributions.
        rng=np.random.default_rng(seed)
        for _ in range(100):
            selected=options
            if options is not None:
                selected=dict(options)
                selected["positions"]=np.asarray(options["positions"])[:,:2]
                raw=np.asarray(options["obstacles"])
                selected["obstacles"]=raw[:,:3] if raw.size else np.empty((0,3))
            scene=generate_scenario(self.config,rng,selected)
            ids=[self._cell_xy(p) for p in scene.targets]
            if len(set(ids))==len(ids):
                return prepare_scene(scene,self.config,seed,options)
            if options is not None:
                raise ValueError("multiple targets in same cell of the Bernoulli grid")
        raise RuntimeError("unable to place non-overlapping target cells in 100 scenes")

    def reset(self,seed=None,options=None):
        self.close()
        c=self.config;w=c.world;n=w.num_uavs
        self.steps=0;self._done=False;self._sequence=0
        self.seed=int(c.runtime.seed if seed is None else seed)
        self.grid_n=math.ceil(w.map_size_m/c.sensor.grid_cell_m)
        self.num_cells=self.grid_n*self.grid_n
        self.scene=self._scene_with_single_target_per_cell(self.seed,options)
        self.positions=self.scene.positions.copy()
        self.velocities=np.zeros_like(self.positions)
        self.battery_j=np.full(n,c.energy.battery_capacity_j)
        self.flight_energy_j=np.zeros(n)
        self._truth_target=np.zeros(self.num_cells,dtype=np.bool_)
        for xy in self.scene.targets:self._truth_target[self._cell_xy(xy)]=True
        self._target_sensor=CorrelatedBinarySensor(self.num_cells,self.seed+4501,self.correlation_strength)
        self.nodes=[CooperativeWorldMap(self.num_cells,self.ds_threshold,self.sprt_alpha,self.sprt_beta)
                    for _ in range(n+1)]
        self.store=M2PacketStore(self.nodes,buffer_bytes=self.map_buffer_bytes,ttl_s=c.network.report_ttl_s)
        self.radio=M2Radio(c,self.store,self.scene,seed=self.seed,battery_j=self.battery_j)
        self._seen_verified=set();self._seen_candidate=set();self._seen_gcs_claims=set()
        self._rewarded_correct_cells=set()
        self._delivery_recall_sum=0.
        self._false_report_exposure_s=0.
        self._prev_score=self._map_score()
        self._sense()
        self._prev_local_uncertainty=self._local_mean_uncertainty()
        return self.local_observations(),self.info()

    def _center(self,cell):
        width=self.config.sensor.grid_cell_m
        return np.array([(cell%self.grid_n+.5)*width,
                         (cell//self.grid_n+.5)*width])

    def _sense(self):
        c=self.config;n=c.world.num_uavs;width=c.sensor.grid_cell_m
        for i,pos in enumerate(self.positions):
            if c.energy.enabled and self.battery_j[i]<=0:continue
            if pos[2]<=0:continue
            pd,pf,r=sensor_profile(c.sensor,pos[2])
            lo_x=max(0,int((pos[0]-r)//width));hi_x=min(self.grid_n-1,int((pos[0]+r)//width))
            lo_y=max(0,int((pos[1]-r)//width));hi_y=min(self.grid_n-1,int((pos[1]+r)//width))
            candidates=[]
            for gy in range(lo_y,hi_y+1):
                for gx in range(lo_x,hi_x+1):
                    cell=gy*self.grid_n+gx
                    q=cell_coverage(gx,gy,pos[:2],r,width,c.world.map_size_m)
                    if q<=0:continue
                    center=self._center(cell)
                    # Target detection is line-of-sight in the 3D flight scene.
                    # Obstacle geometry serves safety and occlusion only; no mapping.
                    target_los=(not c.sensor.obstacle_occlusion or
                                camera_clear(pos,center,self.scene.obstacles,self.scene.obstacle_heights))
                    if not target_los:
                        continue
                    status=self.nodes[i].candidates.get(cell)
                    priority=0 if status and status.status=="pending" else 1
                    # Critically, this selection uses ONLY local map values.
                    age=self.nodes[i].last_observed[cell]
                    candidates.append((priority,age,cell,q,target_los))
            candidates.sort(key=lambda item:(item[0],item[1],item[2]))
            output=[]
            selected=candidates if self.max_scan_cells is None else candidates[:self.max_scan_cells]
            for _priority,_age,cell,q,target_los in selected:
                self._sequence+=1
                if target_los:
                    effective=CellEvidence.from_footprint(pd,pf,q)
                    if effective is not None:
                        yes=self._target_sensor.sample(cell,effective.p1,effective.p0,bool(self._truth_target[cell]))
                        output.append(Event(f"t:{self.seed}:{i}:{self.steps}:{cell}",i,self.steps,cell,
                                            "target",yes,effective.p1,effective.p0,self.correlation_strength))
            if output:
                self.store.publish(i,output,created_s=self.time_s,
                                   payload_bytes=c.network.payload_bytes)

    def local_observations(self):
        c=self.config;n=c.world.num_uavs;width=c.sensor.grid_cell_m
        rows=[]
        radius=2
        for i,pos in enumerate(self.positions):
            # CRITICAL policy ABI: the existing MATD3/MASAC helpers require
            # active at index 9 and action masks in the last (num_uavs+1)
            # observation elements. Keep this historical prefix unchanged.
            base=list(pos[:2]/c.world.map_size_m)
            base.extend(self.velocities[i,:2]/c.world.max_speed_mps)
            base.extend((self.scene.gcs[:2]-pos[:2])/c.world.map_size_m)
            base.extend((self.steps/c.world.max_steps,
                         min(1.,self.store.bytes_used(i)/self.map_buffer_bytes),
                         self.battery_j[i]/c.energy.battery_capacity_j,
                         float(not c.energy.enabled or self.battery_j[i]>0)))
            base.extend((pos[2]/c.world.altitude_max_m,
                         self.velocities[i,2]/c.world.max_speed_mps,
                         (self.scene.gcs[2]-pos[2])/c.world.altitude_max_m))
            cx=int(pos[0]/width);cy=int(pos[1]/width)
            features=[]
            for dy in range(-radius,radius+1):
                for dx in range(-radius,radius+1):
                    gx=cx+dx;gy=cy+dy
                    if 0<=gx<self.grid_n and 0<=gy<self.grid_n:
                        cell=gy*self.grid_n+gx
                        target,status,sprt=self.nodes[i].visible_summary(cell)
                        last_seen=self.nodes[i].last_observed[cell]
                        age_norm=(min(1.,max(0.,self.steps-last_seen)/max(c.world.max_steps,1))
                                  if last_seen>=0 else 1.)
                        features.extend((*target,
                                         float(self.nodes[i].coverage[cell]>0),
                                         np.tanh(sprt/3),age_norm))
                    else:features.extend((0,0,1,0,0,1))
            peers=[j for j in range(n+1) if j!=i]
            # All destinations stay selectable. True link feasibility is known
            # only to the PHY; never leak perfect instantaneous peer positions
            # through an action mask. Actor learns from packet/ACK history.
            masks=[1. for _ in range(n+1)]
            # Only successfully received contacts, never remote true buffer state.
            contacts=[min(1.,max(0.,self.steps-self.nodes[i].link_contacts[j])/max(1,c.world.max_steps))
                      if j in self.nodes[i].link_contacts else 1. for j in peers]
            kinds=("evidence","map_delta","report","correction")
            queued=[sum(p.kind==kind for p in self.store.buffers[i].values())
                    for kind in kinds]
            # No remote buffer counts or ground truth in observation.
            base.extend(min(1.,v/16.) for v in queued)
            rows.append(np.asarray(base+features+contacts+masks,dtype=np.float32))
        return np.asarray(rows,dtype=np.float32)

    def state(self):
        """Observation-only centralized critic; never provides target truth."""
        return self.local_observations().ravel().copy()

    def _map_score(self):
        # Zhang (2024) Eq. (18): minimize Σ_c m_c(U), normalized by grid size.
        # Unlike Brier, information gain uses observed DS mass without truth.
        return float(self.gcs_map.target_masses[:,2].mean())

    def _local_mean_uncertainty(self):
        return float(np.mean([node.target_masses[:,2].mean()
                              for node in self.nodes[:-1]]))

    def _reward(self,flight_cost,radio_cost):
        c=self.config
        now_score=self._map_score()
        map_progress=4.*(self._prev_score-now_score)
        self._prev_score=now_score
        local_score=self._local_mean_uncertainty()
        local_information_gain=self._prev_local_uncertainty-local_score
        self._prev_local_uncertainty=local_score
        candidates={cell for j in range(c.world.num_uavs)
                    for cell in self.nodes[j].candidates}
        new_candidates=candidates-self._seen_candidate
        self._seen_candidate|=candidates
        local_verified={cell for j in range(c.world.num_uavs)
                        for cell,v in self.nodes[j].candidates.items() if v.status=="verified"}
        newly_verified=local_verified-self._seen_verified
        self._seen_verified|=local_verified
        # Reward oracle is ONLY the evaluator; actors cannot read it.
        verify_bonus=sum((1. if self._truth_target[cell] else -1.) for cell in newly_verified)
        notices=[notice for notice,arrived in self.gcs_map.report_history
                 if notice.message_id not in self._seen_gcs_claims]
        self._seen_gcs_claims.update(n.message_id for n in notices)
        delivered=0.
        for notice in notices:
            if notice.status != "verified":continue
            if self._truth_target[notice.cell]:
                if notice.cell not in self._rewarded_correct_cells:
                    delivered+=50.
                    self._rewarded_correct_cells.add(notice.cell)
            else:
                delivered-=10.  # Every false verified transmission costs reward.
        result=(.15*len(new_candidates)+verify_bonus+delivered+map_progress+local_information_gain
                -c.reward.propulsion_energy_per_kj*sum(flight_cost)/1000
                -c.reward.communication_energy*radio_cost-c.reward.step_cost)
        self.last_reward_components={"candidate":.15*len(new_candidates),
                                     "verification":verify_bonus,"report":delivered,
                                     "target_map_progress":map_progress,
                                     "local_information_gain":local_information_gain,
                                     "energy":-c.reward.propulsion_energy_per_kj*sum(flight_cost)/1000
                                              -c.reward.communication_energy*radio_cost,
                                     "step":-c.reward.step_cost}
        return float(result)

    def step(self,action:Action):
        if self._done:raise RuntimeError("reset required after terminal state")
        c=self.config;n=c.world.num_uavs
        action.validate(n,c.continuous_dim)
        details={}
        self.positions,self.velocities,blocked,clipped,flight_cost=funded_motion(
            self.positions,self.velocities,np.asarray(action.continuous)[:,:3],
            self.scene.obstacles,self.battery_j,c,
            diagnostics=details,obstacle_heights=self.scene.obstacle_heights)
        self.flight_energy_j+=flight_cost
        self.steps+=1
        self._sense()
        # The GCS summarizes delivered evidence into packetized map deltas;
        # it probes one UAV in round-robin order, never ideal-broadcasts.
        self.store.prepare_gcs_delta(now=self.time_s)
        gcs_peer=(self.steps % n) if self.store.buffers[n] else None
        radio_cost=self.radio.step(action.destination,self.positions,time_s=self.time_s,
                                   power_actions=np.asarray(action.continuous)[:,-1],
                                   payload_actions=np.asarray(action.continuous)[:,3],
                                   gcs_peer=gcs_peer)
        reward=self._reward(flight_cost,radio_cost)
        self._delivery_recall_sum+=self._current_correct_delivery_count()/max(1,int(self._truth_target.sum()))
        self._false_report_exposure_s+=self._current_false_delivery_count()*c.world.dt_s
        self._done=(self.steps>=c.world.max_steps or
                    (c.energy.enabled and not (self.battery_j>0).any()))
        return self.local_observations(),reward,self._done,False,self.info()

    def _currently_verified_gcs_cells(self):
        # The GCS can have strong local DS evidence without ever receiving a
        # verified REPORT. Delivery metrics must require a real report packet.
        cells={cell for (src,cell),claim in self.gcs_map.claims.items()
               if claim.status=="verified"}
        return {cell for cell in cells if self.gcs_map.consensus_status(cell)=="verified"}

    def _current_correct_delivery_count(self):
        return sum(bool(self._truth_target[cell]) for cell in self._currently_verified_gcs_cells())

    def _current_false_delivery_count(self):
        return sum(not bool(self._truth_target[cell]) for cell in self._currently_verified_gcs_cells())

    def info(self):
        if self.nodes is None:raise RuntimeError("reset the environment first")
        claims=self.gcs_map.claims
        verified=self._currently_verified_gcs_cells()
        correct=sum(bool(self._truth_target[cell]) for cell in verified)
        false=sum(not bool(self._truth_target[cell]) for cell in verified)
        n_reports=correct+false
        flight_total=float(self.flight_energy_j.sum())
        rf_total=float(self.radio.network.energy_j[:-1].sum()) if self.radio else 0.
        return {"steps":self.steps,"num_uavs":self.config.world.num_uavs,
                "correct_gcs_delivery_recall":correct/max(int(self._truth_target.sum()),1),
                "false_report_rate":float(false/n_reports) if n_reports else None,
                "normalized_delivery_auc":self._delivery_recall_sum/max(self.steps,1),
                "false_alert_exposure_seconds":self._false_report_exposure_s,
                "energy_total_j":flight_total+rf_total,
                "energy_per_correct_delivery_j":(flight_total+rf_total)/correct if correct else None,
                "gcs_map":MultiLayerMapMetrics.evaluate(
                    self.gcs_map,self._truth_target,at_tick=self.steps),
                "gcs_verified_correct":correct,"gcs_verified_false":false,
                "gcs_report_history_count":len(self.gcs_map.report_history),
                "local_events":[node.events_seen for node in self.nodes[:-1]],
                "gcs_events":self.gcs_map.events_seen,
                "communication":self.radio.network.metrics() if self.radio else None,
                "propulsion_energy_j":self.flight_energy_j.tolist()}
