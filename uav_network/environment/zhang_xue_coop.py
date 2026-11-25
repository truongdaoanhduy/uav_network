"""Zhang–Xue 1-km cooperative *CPU integration pilot*.

Exercises the real package's MATD3/MASAC actor/action/replay interface with
the v3.4 reference data protocol, per-UAV Dempster–Shafer belief, agent-chosen
application destination and E2E acknowledgements.

Not the production Warp backend and not the vendored UavNetSim SimPy PHY/MAC.
No network performance or convergence claim should rely on this pilot alone.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Tuple
import math

import numpy as np

from uav_network.communication.zhang_xue_ref.ds_belief import BeliefTable
from uav_network.communication.zhang_xue_ref.packets_and_queues import (
    MAX_ENTRIES_PER_PACKET,
    MiniDiscreteEventNet,
    SearchDataPacket,
)
from uav_network.communication.zhang_xue_ref.reward_engine import RewardEngine
from uav_network.communication.zhang_xue_ref.phy_patch import sinr_db_patched
from uav_network.environment.zhang_xue_map_geometry import sample_scene
from uav_network.environment.zhang_xue_battery import (
    ZengPropulsionParameters,
    steady_level_propulsion_power_w,
)


@dataclass(frozen=True)
class ZhangXueConfig:
    grid_width: int = 20
    grid_height: int = 20
    cell_size_meter: float = 50.0
    map_size_meter: float = 1000.0
    uav_count: int = 6
    target_count: int = 10
    fixed_altitude_meter: float = 100.0
    camera_fov_degrees: float = 90.0
    dt_seconds: float = 1.0
    horizon_steps: int = 200
    v_max_mps: float = 50.0
    safe_distance_meter: float = 50.0
    no_fly_zone_count: int = 2
    no_fly_radius_meter: float = 50.0
    nfz_observation_radius_meter: float = 300.0
    blackout_zone_count: int = 2
    blackout_radius_cells: int = 3
    pd: float = 0.85
    pf: float = 0.03
    ds_mf: float = 0.8
    ds_me: float = 0.8
    confirm_threshold: float = 0.95
    tx_power_w: float = 0.5
    packet_ttl_seconds: float = 10.0
    packet_entries_limit: int = 62
    fusion_mode: str = "MAX_PLUS_SUM"
    # Explicit CPU six-radio adaptation; legacy finite-horizon behavior is default.
    termination_regime: str = "finite_horizon"
    usable_battery_e_max_j: float = 180000.0
    avionics_baseline_w: float = 15.0
    # Per-node on-air TX durations are unavailable in these pilot backends.
    rf_tx_pa_baseband_w: float = 0.0
    p0_blade_hover_w: float = 79.86
    pi_induced_hover_w: float = 88.63
    u_tip_mps: float = 120.0
    v0_hover_mps: float = 4.03
    fuselage_drag_d0: float = 0.6
    air_density_rho: float = 1.225
    rotor_solidity_s: float = 0.05
    rotor_disc_area_m2: float = 0.503

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ZhangXueConfig":
        """Load a flat CPU pilot profile; the nested seven-node v5.2 is separate."""
        import yaml

        class UniqueKeyLoader(yaml.SafeLoader):
            pass

        def unique_mapping(loader, node):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if not isinstance(key, str):
                    raise ValueError("CPU pilot YAML keys must be field names")
                if key in result:
                    raise ValueError(f"Duplicate CPU pilot YAML field: {key}")
                result[key] = loader.construct_object(value_node)
            return result

        UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping,
        )
        try:
            with Path(path).open(encoding="utf-8") as stream:
                data = yaml.load(stream, Loader=UniqueKeyLoader)
        except yaml.YAMLError as error:
            raise ValueError("Invalid CPU pilot YAML") from error
        if not isinstance(data, dict):
            raise ValueError("CPU pilot YAML must be a flat field mapping")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown or any(isinstance(v, (dict, list)) for v in data.values()):
            raise ValueError(
                "Expected a flat CPU six-radio pilot profile; nested seven-node "
                f"v5.2 candidates are unsupported. Unknown fields: {sorted(unknown)}"
            )
        return cls(**data)

    def __post_init__(self):
        integer_fields = {
            "grid_width", "grid_height", "uav_count", "target_count", "horizon_steps",
            "no_fly_zone_count", "blackout_zone_count", "blackout_radius_cells",
            "packet_entries_limit",
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("fusion_mode", "termination_regime"):
                if not isinstance(value, str):
                    raise ValueError(f"{field.name} must be a string")
                continue
            if (isinstance(value, (bool, str)) or
                    not isinstance(value, (int, float, np.integer, np.floating)) or
                    not math.isfinite(value)):
                raise ValueError(f"{field.name} must be numeric and finite")
            if field.name in integer_fields and not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field.name} must be an integer")
        if self.termination_regime not in ("finite_horizon", "physical_battery"):
            raise ValueError("termination_regime must be finite_horizon or physical_battery")
        if self.fusion_mode not in ("MAX_PLUS_SUM", "MAX"):
            raise ValueError("Unsupported fusion_mode")
        for name in (
            "cell_size_meter", "map_size_meter", "v_max_mps", "safe_distance_meter",
            "no_fly_radius_meter", "tx_power_w", "usable_battery_e_max_j",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "no_fly_zone_count", "blackout_zone_count", "blackout_radius_cells",
            "avionics_baseline_w", "rf_tx_pa_baseband_w",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.rf_tx_pa_baseband_w != 0:
            raise ValueError(
                "RF battery energy unavailable: pilot backends do not expose actual "
                "per-node TX airtime; rf_tx_pa_baseband_w must be zero"
            )
        propulsion = self.propulsion_parameters()
        if self.termination_regime == "physical_battery":
            if self.v_max_mps >= self.u_tip_mps:
                raise ValueError("physical_battery requires v_max_mps < u_tip_mps")
            max_power = max(steady_level_propulsion_power_w(
                np.array([0.0, self.v_max_mps]), propulsion,
            )) + self.avionics_baseline_w
            if not math.isfinite(max_power * self.dt_seconds):
                raise ValueError("Physical battery step energy must be finite")
        if self.grid_width != 20 or self.grid_height != 20:
            raise ValueError("Pilot state encoder requires exactly 20×20 cells")
        if self.cell_size_meter * self.grid_width != self.map_size_meter:
            raise ValueError("grid width must match map size")
        if self.cell_size_meter * self.grid_height != self.map_size_meter:
            raise ValueError("grid height must match map size")
        if self.nfz_observation_radius_meter <= 0 or not math.isfinite(self.nfz_observation_radius_meter):
            raise ValueError("NFZ observation radius must be positive and finite")
        if self.uav_count != 6:
            raise ValueError("This fixed pilot actor mask requires six UAVs")
        if not (1 <= self.target_count < 400):
            raise ValueError("Invalid target_count")
        if self.horizon_steps < 1 or self.dt_seconds <= 0:
            raise ValueError("Time horizon and dt must be positive")
        if not math.isfinite(self.horizon_steps * self.dt_seconds):
            raise ValueError("Time horizon in seconds must be finite")
        if not (0. < self.camera_fov_degrees < 180.):
            raise ValueError("Camera field of view must lie in (0,180) degrees")
        if not (self.fixed_altitude_meter > 0. and math.isfinite(self.fixed_altitude_meter)):
            raise ValueError("Camera altitude must be finite and positive")
        if not (0 <= self.pf < self.pd <= 1):
            raise ValueError("Require 0 <= Pf < Pd <= 1")
        if not (0 < self.ds_mf < 1 and 0 < self.ds_me < 1):
            raise ValueError("DS masses must lie within (0,1)")
        if not (0 < self.confirm_threshold <= 1):
            raise ValueError("Threshold must be in (0,1]")
        if not (0 < self.packet_entries_limit <= MAX_ENTRIES_PER_PACKET):
            raise ValueError("packet_entries_limit exceeds MTU")
        if self.packet_ttl_seconds <= 0:
            raise ValueError("Packet TTL must be positive")

    def propulsion_parameters(self) -> ZengPropulsionParameters:
        return ZengPropulsionParameters(**{
            field.name: getattr(self, field.name)
            for field in fields(ZengPropulsionParameters)
        })


def _segment_min_distance(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 < 1e-12:
        return math.hypot(ax - cx, ay - cy)
    r = max(0.0, min(1.0, ((cx - ax)*vx + (cy - ay)*vy)/l2))
    return math.hypot(ax + r*vx - cx, ay + r*vy - cy)


class ZhangXueCooperativeEnv:
    """Single-episode, six-UAV CTDE environment with own-source network fusion.

    reset -> (obs[6,D], centralized_state[6*D], action_mask[6,6])
    step(act_cont[6,2], act_disc_idx[6])
      -> (next_obs, next_state, next_mask, reward:float, done:bool, info:dict)

    Discrete action index 0 means NO_SEND; indices 1..5 select a peer in
    ascending UAV-ID order excluding the acting UAV. Routes can use relays.
    All observations and centralized critic state are built from agent-local
    observations, not target truth or another UAV's undisclosed evidence.
    physical_battery is a speed-only steady-flight energy adaptation of this
    six-radio CPU pilot, with no acceleration model or seventh GCS node.
    """

    def __init__(self, config: ZhangXueConfig | None = None, seed: int = 44,
                 network_backend: str = "reference"):
        self.config = config or ZhangXueConfig()
        if network_backend not in ("reference", "classic_simpy_v1"):
            raise ValueError("network_backend must be reference or classic_simpy_v1")
        self.network_backend=network_backend
        self.base_seed = int(seed)
        self.num_agents = self.config.uav_count
        self.cont_dim = 2
        self.disc_dim = self.num_agents
        self.obs_dim = 9 + (self.num_agents-1)*4 + 2*25 + 2*16 + 2 + 8
        self.state_dim = self.num_agents * self.obs_dim
        self.rng = None
        self.t = 0
        self.done = False
        self.positions = np.zeros((self.num_agents, 2), np.float32)
        self.velocity = np.zeros((self.num_agents, 2), np.float32)
        self.battery_remaining_j = np.full(
            self.num_agents, self.config.usable_battery_e_max_j, dtype=np.float64,
        )
        self._propulsion_parameters = self.config.propulsion_parameters()
        self.target_mask = np.zeros((400,), bool)
        self.no_fly_zones: List[Tuple[float, float, float]] = []
        self.blackout_zones: List[Tuple[int, int]] = []
        self.net = None
        self.belief = None
        self.reward_engine = None
        self._acked_counts = np.zeros((self.num_agents,self.num_agents,400,2),np.uint16)
        self._pending: Dict[int, tuple] = {}
        self._seq = 0
        self.last_sent_packets: List[SearchDataPacket] = []
        self.reset(seed=seed)

    def _cell(self, xy) -> int:
        w = self.config.grid_width
        cx = max(0,min(w-1,int(float(xy[0]) // self.config.cell_size_meter)))
        cy = max(0,min(self.config.grid_height-1,int(float(xy[1]) // self.config.cell_size_meter)))
        return cy*w + cx

    def _visible_cells(self, xy):
        """Ground-cell centers inside nadir camera footprint, independent of truth.

        The idealized nadir-camera circular footprint has ground radius
        altitude * tan(FOV / 2). No occlusion is modeled by this 2D pilot;
        no-fly disks restrict motion, not camera rays.
        """
        c=self.config
        radius=c.fixed_altitude_meter*math.tan(math.radians(c.camera_fov_degrees/2))
        x,y=float(xy[0]),float(xy[1])
        lo_x=max(0,int(math.floor((x-radius)/c.cell_size_meter)))
        hi_x=min(c.grid_width-1,int(math.floor((x+radius)/c.cell_size_meter)))
        lo_y=max(0,int(math.floor((y-radius)/c.cell_size_meter)))
        hi_y=min(c.grid_height-1,int(math.floor((y+radius)/c.cell_size_meter)))
        visible=[]
        rr=radius*radius
        for cy in range(lo_y,hi_y+1):
            py=(cy+.5)*c.cell_size_meter
            for cx in range(lo_x,hi_x+1):
                px=(cx+.5)*c.cell_size_meter
                if (px-x)**2+(py-y)**2<=rr+1e-7:
                    visible.append(cy*c.grid_width+cx)
        return visible

    def _generate_scenario(self):
        (positions,targets,no_fly,blackout)=sample_scene(self.config,self.rng)
        self.positions[:]=positions
        self.velocity[:]=0.
        self.target_mask[:]=targets
        self.no_fly_zones=no_fly
        self.blackout_zones=blackout

    def reset(self, seed: int | None = None):
        # A None seed continues the existing episode RNG stream (Gymnasium
        # convention), instead of replaying the same training map forever.
        # Explicit reset(seed=...) still reproduces a scene exactly.
        if seed is not None:
            self.rng=np.random.default_rng(int(seed))
            network_seed=int(seed)
        elif self.rng is None:
            self.rng=np.random.default_rng(self.base_seed)
            network_seed=self.base_seed
        else:
            network_seed=int(self.rng.integers(0, 2**31-1))
        self.t=0
        self.done=False
        self.battery_remaining_j.fill(self.config.usable_battery_e_max_j)
        self._generate_scenario()
        c=self.config
        self.belief=BeliefTable(c.uav_count,400,c.fusion_mode,c.ds_mf,c.ds_me)
        network_positions={
            i:tuple(float(v) for v in self.positions[i])
            for i in range(c.uav_count)
        }
        if self.network_backend=="classic_simpy_v1":
            # This path is actual upstream classic v1 SimPy/CSMA. It is
            # intended for one evaluation episode at a time, never E4096.
            from uav_network.communication.classic_v1_simpy import ClassicV1SimPyNetwork
            self.net=ClassicV1SimPyNetwork(
                network_positions,self.belief,seed=network_seed,
                tx_power_w=c.tx_power_w,
            )
        else:
            self.net=MiniDiscreteEventNet(
                network_positions,self.belief,tx_power_w=c.tx_power_w,
            )
        self.reward_engine=RewardEngine(
            c.uav_count,400,self.target_mask,tau=c.confirm_threshold,
            omega_s=30.,omega_c=15.,omega_target=10.,omega_fp=5.,
            omega_fp_hold=.1,lambda_b=30.,
        )
        self._acked_counts.fill(0)
        self._pending.clear()
        self._seq=0
        self.last_sent_packets.clear()
        self._sync_physics()
        return self._build_outputs()

    def _blackout(self, xy) -> bool:
        gx=int(float(xy[0])//self.config.cell_size_meter)
        gy=int(float(xy[1])//self.config.cell_size_meter)
        return any(
            abs(gx-zx)+abs(gy-zy)<=self.config.blackout_radius_cells
            for zx,zy in self.blackout_zones
        )

    def _sync_physics(self):
        for u in range(self.num_agents):
            self.net.positions[u]=tuple(float(v) for v in self.positions[u])
            self.net.radio_enabled[u]=not self._blackout(self.positions[u])

    def _peer_order(self,u:int):
        return [p for p in range(self.num_agents) if p!=u]

    def _dirty_cell_ids(self,u:int,p:int):
        own_p=self.belief.pos[u,u]
        own_n=self.belief.neg[u,u]
        a=self._acked_counts[u,p]
        return np.flatnonzero((own_p>a[:,0]) | (own_n>a[:,1]))

    def _build_outputs(self):
        obs=np.zeros((self.num_agents,self.obs_dim),np.float32)
        mask=np.zeros((self.num_agents,self.disc_dim),np.uint8)
        mask[:,0]=1
        c=self.config
        for u in range(self.num_agents):
            x,y=self.positions[u]
            f,e,m=self.belief.get_belief(u)
            f=f.reshape(20,20)
            m=m.reshape(20,20)
            own_radio=not self._blackout(self.positions[u])
            q=self.net.queues[u]
            queue_len=len(q.transmitting_queue)+len(q.waiting_list)
            ack_count=sum(len(self.net.acked_sequences.get((u,p),set())) for p in self._peer_order(u))
            dirty_counts={p:len(self._dirty_cell_ids(u,p)) for p in self._peer_order(u)}
            # Idealized *local* NFZ sensor: only hazards within perception
            # range are exposed. No globally omniscient NFZ/blackout map.
            nfz_near=[]
            for fx,fy,fr in self.no_fly_zones:
                dx=float(fx-x)
                dy=float(fy-y)
                dist=math.hypot(dx,dy)
                if dist <= c.nfz_observation_radius_meter:
                    nfz_near.append((dist,dx,dy))
            nfz_near.sort(key=lambda z:z[0])
            nfz_local=[]
            for idx in range(2):
                if idx<len(nfz_near):
                    dist,dx,dy=nfz_near[idx]
                    nfz_local.extend([
                        dx/c.map_size_meter, dy/c.map_size_meter,
                        dist/c.map_size_meter, 1.,
                    ])
                else:
                    # Explicit absent bit eliminates collision-at-center alias;
                    # radius-valued padding follows Zhang 2024 Eq.(26).
                    nfz_local.extend([
                        0., 0., c.nfz_observation_radius_meter/c.map_size_meter, 0.,
                    ])
            base=[
                x/c.map_size_meter,y/c.map_size_meter,
                float(self.velocity[u,0])/c.v_max_mps,float(self.velocity[u,1])/c.v_max_mps,
                (float(self.battery_remaining_j[u])/c.usable_battery_e_max_j
                 if c.termination_regime == "physical_battery"
                 else (c.horizon_steps-self.t)/c.horizon_steps),
                min(len(nfz_near)/max(c.no_fly_zone_count,1),1.),
                float(own_radio),queue_len/q.max_size,
                min(ack_count/10.,1.),
            ]
            peers=[]
            for j,p in enumerate(self._peer_order(u),start=1):
                if self.network_backend=="classic_simpy_v1":
                    # The real classic Greedy HELLO process populates this
                    # receiver-local cache by on-air packets. Do NOT use true
                    # current remote positions as actor observations.
                    tbl=self.net.simulator.drones[u].routing_protocol.neighbor_table
                    entry=tbl.neighbor_table.get(p)
                    seen=bool(
                        own_radio and entry is not None
                        and self.net.simulator.env.now-entry[1]<=tbl.entry_life_time
                    )
                    if seen:
                        dx,dy=np.asarray(entry[0][:2],np.float32)-self.positions[u]
                        dist=math.hypot(float(dx),float(dy))
                    else:
                        dx=dy=dist=0.
                else:
                    # Reference-only ideal one-hop sighting; not a real
                    # HELLO process. Never conflate it with the SimPy path.
                    dx,dy=self.positions[p]-self.positions[u]
                    dist=math.hypot(float(dx),float(dy))
                    seen=bool(
                        own_radio and not self._blackout(self.positions[p])
                        and sinr_db_patched(c.tx_power_w,dist,[],
                                           tx_radio_enabled=True)
                        >=self.net.sinr_threshold_db
                    )
                peers.extend([
                    float(dx)/c.map_size_meter if seen else 0.,
                    float(dy)/c.map_size_meter if seen else 0.,
                    dist/c.map_size_meter if seen else 0.,
                    min(1.,dirty_counts[p]/400.),
                ])
                if own_radio and dirty_counts[p]>0:
                    mask[u,j]=1
            cy,cx=divmod(self._cell(self.positions[u]),20)
            patch=[]
            for dy in (-2,-1,0,1,2):
                for dx in (-2,-1,0,1,2):
                    yy,xx=cy+dy,cx+dx
                    if 0<=yy<20 and 0<=xx<20:
                        patch.extend((float(f[yy,xx]),float(m[yy,xx])))
                    else:
                        patch.extend((0.,1.))
            pooled=[]
            for yy in range(0,20,5):
                for xx in range(0,20,5):
                    pooled.extend((float(f[yy:yy+5,xx:xx+5].mean()),
                                   float(m[yy:yy+5,xx:xx+5].mean())))
            cid=self._cell(self.positions[u])
            own=[float(self.belief.pos[u,u,cid])/c.horizon_steps,
                 float(self.belief.neg[u,u,cid])/c.horizon_steps]
            encoded=np.asarray(base+peers+patch+pooled+own+nfz_local,dtype=np.float32)
            if encoded.shape!=(self.obs_dim,):
                raise AssertionError((encoded.shape,self.obs_dim))
            obs[u]=encoded
        state=obs.reshape(-1).copy()  # centralized CTDE contains local observations only
        return obs,state,mask

    def _process_delivered_ack(self):
        # E2E sequence acknowledgement is exact, never infer that skipped seq IDs
        # were successfully delivered from a later max/high watermark.
        for (sender,peer),acked_ids in self.net.acked_sequences.items():
            for seq in acked_ids:
                pending=self._pending.pop(seq,None)
                if pending is None:
                    continue
                src,dst,entries,_created_at=pending
                if src!=sender or dst!=peer:
                    raise AssertionError("ACK peer mismatch")
                for origin,cell,n_pos,n_neg in entries:
                    if origin!=src:
                        raise AssertionError("Own-source-only payload contract violated")
                    self._acked_counts[src,dst,cell,0]=max(
                        self._acked_counts[src,dst,cell,0],n_pos)
                    self._acked_counts[src,dst,cell,1]=max(
                        self._acked_counts[src,dst,cell,1],n_neg)
        # Unacknowledged application packets cannot be useful forever:
        # data TTL + independently routed application ACK TTL are bounded.
        cutoff=self.net.now - 2*self.config.packet_ttl_seconds
        for seq,(_src,_dst,_entries,created) in tuple(self._pending.items()):
            if created<=cutoff:
                del self._pending[seq]

    def _schedule_actions(self,action_disc,mask):
        c=self.config
        self.last_sent_packets=[]
        for u in range(self.num_agents):
            idx=int(action_disc[u])
            if idx<0 or idx>=c.uav_count:
                raise ValueError("Invalid discrete action index")
            if not mask[u,idx] or idx==0:
                continue
            peer=self._peer_order(u)[idx-1]
            cells=self._dirty_cell_ids(u,peer)[:c.packet_entries_limit]
            entries=tuple((u,int(cell),int(self.belief.pos[u,u,cell]),
                           int(self.belief.neg[u,u,cell])) for cell in cells)
            if not entries:
                continue
            self._seq+=1
            p=SearchDataPacket(
                packet_id=self._seq, episode_id=1, src_id=u,dst_id=peer,
                seq_id=self._seq,creation_time=float(self.t)*c.dt_seconds,
                deadline=c.packet_ttl_seconds,entries=entries,
            )
            if self.net.send_data(float(self.t)*c.dt_seconds,p):
                self._pending[self._seq]=(u,peer,entries,float(self.t)*c.dt_seconds)
                self.last_sent_packets.append(p)

    def _move(self,action_cont):
        c=self.config
        safety=0
        boundary=0
        old=self.positions.copy()
        proposed=old.copy()
        # Alternate ID processing order every step; priority is deterministic
        # but no UAV consistently wins close-motion conflicts.
        ordering = (range(self.num_agents) if self.t % 2 == 0
                    else range(self.num_agents - 1, -1, -1))
        for u in ordering:
            vec=np.asarray(action_cont[u],np.float64)
            speed=np.linalg.norm(vec)
            if speed>1.:
                vec/=speed
            new=old[u]+vec*c.v_max_mps*c.dt_seconds
            if np.any(new<0) or np.any(new>=c.map_size_meter):
                boundary+=1
                continue
            if any(
                _segment_min_distance(float(old[u,0]),float(old[u,1]),
                                      float(new[0]),float(new[1]),cx,cy)<radius
                for cx,cy,radius in self.no_fly_zones
            ):
                safety+=1
                continue
            # Conservative planning against each other UAV's *old* position and
            # moves already accepted. Opposite simultaneous crossings are blocked.
            if any(
                p!=u and (
                    _segment_min_distance(float(old[u,0]),float(old[u,1]),
                                          float(new[0]),float(new[1]),
                                          float(old[p,0]),float(old[p,1]))
                    < c.safe_distance_meter
                    or math.hypot(float(new[0]-proposed[p,0]),
                                  float(new[1]-proposed[p,1]))<c.safe_distance_meter
                ) for p in range(self.num_agents)
            ):
                safety+=1
                continue
            proposed[u]=new.astype(np.float32)
        self.velocity=(proposed-old)/c.dt_seconds
        self.positions=proposed
        self._sync_physics()
        return safety,boundary

    def step(self,action_cont,action_disc_idx):
        if self.done:
            raise RuntimeError("Episode ended; call reset()")
        c=self.config
        a=np.asarray(action_cont,dtype=np.float32)
        d=np.asarray(action_disc_idx)
        if a.shape!=(self.num_agents,2) or d.shape!=(self.num_agents,):
            raise ValueError("Expected continuous [6,2] and discrete [6]")
        if not np.isfinite(a).all() or not np.issubdtype(d.dtype,np.integer):
            raise ValueError("Actions must be finite and discrete indices integers")
        _,_,mask=self._build_outputs()
        # t_k: send only the evidence available before this step's sensing
        self._schedule_actions(d,mask)
        before_mu=np.stack([self.belief.get_belief(u)[2] for u in range(self.num_agents)])
        airtime_prev=self.net.metrics.total_airtime_s
        self.net.run_until((self.t+1)*c.dt_seconds)
        self._process_delivered_ack()
        after_mu=np.stack([self.belief.get_belief(u)[2] for u in range(self.num_agents)])
        shared_delta=np.maximum(0.,before_mu-after_mu).sum(axis=0)
        airtime_used=self.net.metrics.total_airtime_s-airtime_prev

        # t_{k+1}: motion then independent sensing, with no same-step transmission
        safety,boundary=self._move(a)
        if c.termination_regime == "physical_battery":
            accepted_speed = np.linalg.norm(self.velocity.astype(np.float64), axis=1)
            battery_requested_j = (
                steady_level_propulsion_power_w(accepted_speed, self._propulsion_parameters)
                + c.avionics_baseline_w
            ) * c.dt_seconds
            battery_used_j = np.minimum(self.battery_remaining_j, battery_requested_j)
            self.battery_remaining_j = np.maximum(
                0.0, self.battery_remaining_j - battery_used_j,
            )
        own_mu_before=self.belief.get_team_own_belief()[2]
        for u in range(self.num_agents):
            for cell in self._visible_cells(self.positions[u]):
                detected=bool(self.rng.random() < (
                    c.pd if self.target_mask[cell] else c.pf
                ))
                self.belief.record_own_observation(u,cell,detected)
        own_mu_after=self.belief.get_team_own_belief()[2]
        search_delta=np.maximum(0.,own_mu_before-own_mu_after)
        team_mf=self.belief.get_team_own_belief()[0]
        reward=self.reward_engine.compute_unified_step_reward(
            search_delta,shared_delta,team_mf,airtime_used/c.dt_seconds,
            safety_penalty=30.*safety,boundary_penalty=20.*boundary,
        )
        self.t+=1
        mission_success = bool(np.all(self.reward_engine.team_tp_latch[self.target_mask]))
        horizon_reached = self.t >= c.horizon_steps
        battery_exhausted = bool(
            c.termination_regime == "physical_battery"
            and np.any(self.battery_remaining_j <= 0.0)
        )
        # First depleted UAV ends this cooperative sortie. Energy is integrated
        # over accepted full steps; depletion time is resolved to dt, not substeps.
        terminated = bool(mission_success or battery_exhausted or (
            c.termination_regime == "finite_horizon" and horizon_reached
        ))
        truncated = bool(horizon_reached and not terminated)
        self.done = terminated or truncated
        info={
            "step":self.t,
            "terminated":terminated,
            "truncated":truncated,
            "bootstrap_allowed":not terminated,
            "termination_regime":c.termination_regime,
            "battery_exhausted":battery_exhausted,
            "mission_success":mission_success,
            "termination_reason":(
                "battery_exhaustion" if battery_exhausted else
                "team_target_confirmation" if mission_success else
                "finite_horizon" if terminated else
                "safety_horizon" if truncated else None
            ),
            "data_generated":self.net.metrics.data_generated,
            "data_pdr":float(self.net.metrics.data_pdr),
            "ack_delivered":self.net.metrics.e2e_ack_delivered,
            "safety_violations":safety,
            "boundary_violations":boundary,
            "airtime_this_step_s":float(airtime_used),
            "tp_confirmed":int(self.reward_engine.team_tp_latch[self.target_mask].sum()),
            "fp_latched":int(self.reward_engine.team_fp_latch.sum()),
        }
        if c.termination_regime == "physical_battery":
            info.update({
                "battery_model": "zeng_eq12_steady_level_speed_only",
                "battery_remaining_j": self.battery_remaining_j.tolist(),
                "battery_soc": (self.battery_remaining_j/c.usable_battery_e_max_j).tolist(),
                "battery_energy_used_j": battery_used_j.tolist(),
                "battery_requested_energy_j": battery_requested_j.tolist(),
                "battery_accepted_speed_mps": accepted_speed.tolist(),
                "battery_rf_energy_status": "unavailable_no_per_node_tx_airtime",
            })
        obs,state,mask=self._build_outputs()
        return obs,state,mask,float(reward),bool(self.done),info
