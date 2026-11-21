"""Opt-in M2 event packets through the REAL native UavNetSim PHY/MAC.

Only packet_received -> store.transfer -> node.ingest changes a remote map.
UAV forwarding destinations and TX power come from policy actions. The GCS
can downlink only to an in-range UAV; beyond reach peers must relay with their
own policy actions. No ideal/global map fanout is used here.

The legacy native bridge expects report.target_id in diagnostic traces. This
adapter supplies -1 (NOT a true target ID); envelopes carry observed cell IDs.
Legacy SearchEnv and Warp packet kernels are unchanged by this adapter.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import OrderedDict
from typing import Sequence
import itertools
import math
import numpy as np
from .m2_protocol import CooperativeWorldMap, DecisionNotice, packetize_events
from ..environment.correlated_mapping import Event, ObstacleGeometryEvent
from .link_budget import nominal_required_power_w


@dataclass(frozen=True)
class M2Envelope:
    message_id: str
    source_id: int
    created_s: float
    size_bytes: int
    events: tuple[Event|DecisionNotice|ObstacleGeometryEvent,...]
    hops: int = 0
    kind: str = "evidence"
    map_snapshots: tuple = ()

    @property
    def key(self):
        return (self.message_id, self.source_id, self.created_s)

    @property
    def target_id(self):
        # Diagnostics compatibility only. NEVER a simulator true-target ID.
        return -1


class M2PacketStore:
    """Bounded store-and-forward gossip; copy custody, immutable event IDs."""
    def __init__(self,nodes:Sequence[CooperativeWorldMap],buffer_bytes=65536,ttl_s=300):
        if len(nodes)<2 or buffer_bytes<1024 or ttl_s<=0:
            raise ValueError("need UAV nodes, GCS, finite buffer and TTL")
        self.nodes=list(nodes)
        self.n=len(nodes)-1
        self.buffer_bytes=buffer_bytes
        self.ttl_s=ttl_s
        self.buffers=[OrderedDict() for _ in nodes]
        self.received_ids=[set() for _ in nodes]
        self._serial=itertools.count(1)
        # Existing UavNetSim telemetry and PHY adapter hooks (not mission truth).
        self.delivered=np.zeros(0,dtype=np.int32)
        self.delivery_times_s=[]
        self.delivery_hops=[]
        self.delivery_hop_histogram=np.zeros(8,dtype=np.int64)
        self.custody_peer_transfers=0
        self.direct_gcs_reports=0
        self.relayed_gcs_reports=0
        self.event_delivery_count=0
        self.overflow_drops=0
        self.acked_pairs=set()  # only populated after actual accepted reception
        self._delta_prepared=set()

    def bytes_used(self,node:int):
        return sum(packet.size_bytes for packet in self.buffers[node].values())

    def _make_room(self,node:int,size:int):
        if size>self.buffer_bytes:raise BufferError("one packet exceeds node storage")
        while self.bytes_used(node)+size>self.buffer_bytes:
            candidates=list(self.buffers[node].items())
            if not candidates:raise BufferError("cannot reserve packet buffer")
            # Evict oldest evidence before a report/correction; maps retain
            # already-processed evidence. Messages in flight become invalid.
            msg_id,_=min(candidates,key=lambda item:(
                bool(any(isinstance(x,DecisionNotice) for x in item[1].events)),
                item[1].created_s))
            del self.buffers[node][msg_id]
            self.overflow_drops+=1

    def publish(self,source:int,events:Sequence[Event],*,created_s:float,
                payload_bytes:int=1024):
        if not (0<=source<self.n):raise ValueError("only UAV sensors can publish observations")
        batches=packetize_events(events,payload_bytes=payload_bytes)
        published=[]
        for batch in batches:
            identity=f"e:{source}:{next(self._serial)}"
            envelope=M2Envelope(identity,source,float(created_s),batch.size_bytes,batch.events)
            self._make_room(source,envelope.size_bytes)
            self.buffers[source][identity]=envelope
            self.received_ids[source].add(identity)
            for event in envelope.events:
                self.nodes[source].ingest(event,arrived_at=int(created_s))
            published.append(envelope)
        for notice in self.nodes[source].make_decision_notices(source=source):
            self.publish_notice(source,notice)
        return published[0] if len(published)==1 else published

    def publish_notice(self,source:int,notice:DecisionNotice):
        if not 0 <= source < self.n or notice.source != source:
            raise ValueError("only a source UAV can publish its own claim")
        import json
        from dataclasses import asdict
        size=len(json.dumps(asdict(notice),sort_keys=True,separators=(",",":")).encode())+24
        self._make_room(source,size)
        identity=f"d:{source}:{next(self._serial)}"
        kind="correction" if notice.status=="retracted" else "report"
        packet=M2Envelope(identity,source,float(notice.tick),size,(notice,),kind=kind)
        self.buffers[source][identity]=packet
        if notice.status in ("verified", "retracted", "rejected"):
            self.buffers[source].move_to_end(identity,last=False)
        self.received_ids[source].add(identity)
        self.nodes[source].ingest_notice(notice,arrived_at=notice.tick)
        return packet

    def publish_geometry(self,source:int,observations,*,created_s:float):
        from dataclasses import asdict
        import json
        if not 0<=source<self.n:raise ValueError("UAV sensor source required")
        result=[]
        for obs in observations:
            if not isinstance(obs,ObstacleGeometryEvent):
                raise TypeError("ObstacleGeometryEvent required")
            size=len(json.dumps(asdict(obs),separators=(",",":")).encode())+24
            self._make_room(source,size)
            identity=f"g:{source}:{next(self._serial)}"
            packet=M2Envelope(identity,source,float(created_s),size,(obs,),kind="geometry")
            self.buffers[source][identity]=packet
            self.received_ids[source].add(identity)
            self.nodes[source].ingest_geometry(obs,arrived_at=math.ceil(created_s))
            result.append(packet)
        return result

    def select_packet(self,source:int,destination:int,time_s:float,*,payload_choice="any",
                      eligibility=None):
        if payload_choice not in {"any","evidence","geometry","map_delta","report","correction"}:
            raise ValueError("unknown packet kind")
        rank={"correction":0,"report":1,"map_delta":2,"evidence":3,"geometry":4}
        eligible=[p for p in self.buffers[source].values()
                  if (payload_choice=="any" or p.kind==payload_choice)
                  and self.can_forward(source,destination,p,time_s)
                  and (eligibility is None or eligibility(p))]
        return min(eligible,key=lambda p:(rank.get(p.kind,5),p.created_s,p.message_id)) if eligible else None

    def prepare_gcs_delta(self,*,now:float):
        """GCS synthesizes an actual map-state delta with ORIGINAL evidence IDs.

        Recipients recompute conditioned DS from the attached events, never
        double-fuse the snapshot as if it were independent evidence.
        """
        from .m2_protocol import _wire_bytes
        gcs=self.nodes[self.n]
        fresh=[e for e in gcs.events.values() if e.event_id not in self._delta_prepared]
        if not fresh:return None
        fresh.sort(key=lambda e:e.sort_key)
        first=fresh[0]
        # Build a compact cell delta. Other observed cells will be sent later.
        chosen=[]
        for e in fresh:
            if e.cell!=first.cell:continue
            trial=chosen+[e]
            if _wire_bytes(trial)+192>1024:break
            chosen=trial
        if not chosen:chosen=[first]
        snapshot=(int(first.cell),
                  tuple(float(x) for x in gcs.target_masses[first.cell]),
                  tuple(float(x) for x in gcs.obstacle_masses[first.cell]))
        size=_wire_bytes(chosen)+112
        if size>1024:raise ValueError("map delta payload too large")
        self._make_room(self.n,size)
        identity=f"m:{self.n}:{next(self._serial)}"
        packet=M2Envelope(identity,self.n,float(now),size,tuple(chosen),kind="map_delta",
                          map_snapshots=(snapshot,))
        self.buffers[self.n][identity]=packet
        self.received_ids[self.n].add(identity)
        self._delta_prepared.update(e.event_id for e in chosen)
        return packet

    @staticmethod
    def payload_choice_from_action(kind_action:float):
        """Policy continuous output mapped to packet category; zero = any."""
        a=float(kind_action)
        if a < -.6:return "evidence"
        if a < -.2:return "geometry"
        if a <= .2:return "any"
        if a <= .6:return "map_delta"
        if a <= .85:return "report"
        return "correction"

    def contains(self,source:int,report:M2Envelope):
        return self.buffers[source].get(report.message_id)==report

    def can_forward(self,source:int,destination:int,report:M2Envelope,time_s:float):
        return (source!=destination and self.contains(source,report)
                and not (destination==report.source_id and
                         all(getattr(e,"origin",-1)==destination for e in report.events))
                and (source,destination,report.message_id) not in self.acked_pairs
                and 0 <= time_s-report.created_s < self.ttl_s
                and report.size_bytes <= self.buffer_bytes)

    def transfer(self,source:int,destination:int,report:M2Envelope,time_s:float):
        if not self.can_forward(source,destination,report,time_s):
            return False
        # The source RETAINS a copy to gossip to other peers. On the receiving
        # node the radio packet is materialized only after all fragments arrive.
        self.acked_pairs.add((source,destination,report.message_id))
        # A link contact is visible to the RECEIVER only after radio delivery.
        self.nodes[destination].link_contacts[source]=math.ceil(time_s)
        # Receiver delivery acknowledgment proxy: the transmitting UAV learns
        # this contact only after an actual completed radio transfer, never
        # merely because two nodes are geometrically close.
        self.nodes[source].link_contacts[destination]=math.ceil(time_s)
        if report.message_id in self.received_ids[destination]:
            # Another route reached this node first. The packet still had to
            # traverse the PHY/MAC; only its evidence is discarded as duplicate.
            return True
        self._make_room(destination,report.size_bytes)
        self.buffers[destination][report.message_id]=replace(report,hops=report.hops+1)
        self.received_ids[destination].add(report.message_id)
        for event in report.events:
            if isinstance(event,DecisionNotice):
                self.nodes[destination].ingest_notice(event,arrived_at=math.ceil(time_s))
            elif isinstance(event,ObstacleGeometryEvent):
                self.nodes[destination].ingest_geometry(event,arrived_at=math.ceil(time_s))
            else:
                self.nodes[destination].ingest(event,arrived_at=math.ceil(time_s))
        if destination<self.n:
            for notice in self.nodes[destination].make_decision_notices(source=destination):
                self.publish_notice(destination,notice)
        if destination==self.n:
            self.event_delivery_count+=len(report.events)
            self.delivery_times_s.append(time_s-report.created_s)
            self.delivery_hops.append(report.hops+1)
            self.delivery_hop_histogram[min(report.hops,7)]+=1
            if report.hops==0:self.direct_gcs_reports+=1
            else:self.relayed_gcs_reports+=1
        elif source<self.n:
            self.custody_peer_transfers+=1
        return True

    def expire(self,now:float):
        for node in range(len(self.nodes)):
            for message_id,packet in list(self.buffers[node].items()):
                if now-packet.created_s >= self.ttl_s:
                    del self.buffers[node][message_id]


class M2Radio:
    """Native UavNetSim bridge with policy-selected UAV hops and GCS downlink."""
    def __init__(self,config,store:M2PacketStore,scene,seed=44,battery_j=None):
        from .backend import UavNetSimNetwork
        if len(store.nodes)!=config.world.num_uavs+1:
            raise ValueError("store requires exactly one GCS map")
        self.config=config;self.store=store;self.scene=scene
        self.network=UavNetSimNetwork(config,store,scene,seed,battery_j=battery_j)

    def link_available(self,source:int,destination:int,positions):
        n=self.config.world.num_uavs
        if not (0<=source<=n and 0<=destination<=n and source!=destination):
            return False
        src=np.asarray(self.scene.gcs if source==n else positions[source],dtype=float)
        dst=np.asarray(self.scene.gcs if destination==n else positions[destination],dtype=float)
        reach=(self.config.network.gcs_range_m if (source==n or destination==n)
               else self.config.network.peer_range_m)
        if np.linalg.norm(src-dst)>reach or not self.network.node_active(source) or not self.network.node_active(destination):
            return False
        if self.config.network.power_limited_admission:
            required=nominal_required_power_w(src,dst,self.scene.obstacles,self.config.network,
                self.config.sensor.altitude_m,source==n or destination==n,
                heights=getattr(self.scene,"obstacle_heights",None))
            if required>self.config.network.tx_power_max_w:return False
        return True

    def step(self,destinations,positions,*,time_s:float,power_actions,gcs_peer=None,payload_actions=None):
        from ..environment.env import Action
        n=self.config.world.num_uavs
        destinations=np.asarray(destinations,dtype=np.int64)
        power_actions=np.asarray(power_actions,dtype=np.float32)
        if destinations.shape!=(n,) or power_actions.shape!=(n,):
            raise ValueError("one destination and power action per UAV")
        continuous=np.zeros((n,self.config.continuous_dim),dtype=np.float32)
        continuous[:,-1]=power_actions
        if self.config.runtime.m2_enabled and payload_actions is not None:
            payload_actions=np.asarray(payload_actions,dtype=np.float32)
            if payload_actions.shape!=(n,):
                raise ValueError("one packet category per UAV")
            continuous[:,-2]=payload_actions
        self.store.expire(self.network.clock.now/1e6)
        if gcs_peer is not None:
            if not 0<=gcs_peer<n:raise ValueError("invalid downlink peer")
            # GCS cannot send outside range and must not learn a peer's map by
            # direct memory access. It only schedules *packets it previously got*.
            if self.link_available(n,gcs_peer,positions):
                self.network.sync_positions(positions)
                msg=self.store.select_packet(n,gcs_peer,self.network.clock.now/1e6,
                                             payload_choice="map_delta")
                if msg is None:
                    msg=self.store.select_packet(n,gcs_peer,self.network.clock.now/1e6,
                                                 payload_choice="any")
                if msg is not None:
                    self.network._inject(n,gcs_peer,self.config.network.tx_power_max_w,msg)
        reward_energy=self.network.step(Action(continuous,destinations),positions,time_s)
        return reward_energy

    def close(self):
        self.network.close()
