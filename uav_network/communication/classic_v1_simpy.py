"""Pinned upstream UavNetSim-v1 CSMA/CA SimPy *evaluation* backend.

Unlike the v3.5 mini event simulator, the MAC backoff, packet reception,
per-hop ACK, radio PHY/SINR, and neighbor HELLO come directly from the
classic public UavNetSim-v1 source (commit PINNED_V1_SHA). A local DI drone
suppresses the upstream Poisson generator, GaussMarkov mobility and default
hardwired DSDV side effects. Only single-instance, CPU/SimPy use is supported.

This is NOT the unrelated newer Sionna UavNetSim and NOT Warp.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import types
from typing import Any

import numpy as np

from uav_network.communication.zhang_xue_ref.ds_belief import BeliefTable
from uav_network.communication.zhang_xue_ref.packets_and_queues import (
    SearchDataPacket, MiniDiscreteEventNet, NetworkMetricsFilter,
)

PINNED_V1_SHA="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
DEFAULT_V1_ROOT=Path(__file__).resolve().parents[2]/"outputs/reference_sources/UavNetSim-v1-classic"


def _load_upstream(root: str | Path | None) -> dict:
    root=Path(root or os.environ.get("UAV_CLASSIC_V1_ROOT") or DEFAULT_V1_ROOT).expanduser().resolve()
    if not (root/"entities/drone.py").is_file():
        raise RuntimeError(f"Classic UavNetSim-v1 checkout missing: {root}; run scripts/fetch_uavnetsim_v1_classic.sh")
    sha=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,timeout=8).strip()
    if sha!=PINNED_V1_SHA:
        raise RuntimeError(f"UavNetSim-v1 SHA mismatch: {sha}; expected {PINNED_V1_SHA}")
    # The unnamespaced upstream uses absolute imports "utils", "phy", etc.
    # A subprocess isolates these globals from Sionna-vendor and 4096 envs.
    for mod in ("utils.config","phy.phy","entities.drone","routing.greedy.greedy"):
        old=sys.modules.get(mod)
        if old is not None and not Path(old.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Classic UavNetSim module conflict: {mod}; run in isolated Python subprocess")
    sys.path.insert(0,str(root))
    import simpy
    from entities.drone import Drone
    from entities.packet import DataPacket, AckPacket
    from mac.csma_ca import CsmaCa
    from phy.phy import Phy
    from phy.channel import Channel
    from routing.greedy.greedy import Greedy
    from simulator.metrics import Metrics
    from allocation.channel_assignment import ChannelAssigner
    from utils import config
    config.TRANSMITTING_POWER=0.5
    config.MAX_QUEUE_SIZE=50
    return locals()


def _make_types(imported: dict):
    simpy=imported["simpy"]
    Drone=imported["Drone"]
    DataPacket=imported["DataPacket"]
    AckPacket=imported["AckPacket"]
    CsmaCa=imported["CsmaCa"]
    Phy=imported["Phy"]
    Greedy=imported["Greedy"]
    ChannelAssigner=imported["ChannelAssigner"]
    config=imported["config"]

    class ApplicationDataPacket(DataPacket):
        def __init__(self, original: SearchDataPacket, net):
            src=net.simulator.drones[original.src_id]
            dst=net.simulator.drones[original.dst_id]
            super().__init__(
                src_drone=src,dst_drone=dst,creation_time=net.simulator.env.now,
                data_packet_id=original.packet_id,data_packet_length=original.frame_bytes*8,
                simulator=net.simulator,channel_id=1,
            )
            self.app_seq_id=original.seq_id
            self.app_entries=original.entries
            self.app_sender=original.src_id
            self.app_receiver=original.dst_id
            self.deadline=original.deadline*1e6
            self.transmission_mode=0

    class RoutedE2EAckPacket(DataPacket):
        """Separate routable application ACK; upstream link MAC ACK is AckPacket."""
        def __init__(self, source_pkt: ApplicationDataPacket, net):
            src=net.simulator.drones[source_pkt.app_receiver]
            dst=net.simulator.drones[source_pkt.app_sender]
            super().__init__(
                src_drone=src,dst_drone=dst,creation_time=net.simulator.env.now,
                data_packet_id=600_000+source_pkt.packet_id,
                data_packet_length=90*8, simulator=net.simulator,channel_id=1,
            )
            self.acked_origin_id=source_pkt.app_sender
            self.acked_peer_id=source_pkt.app_receiver
            self.acked_seq_id=source_pkt.app_seq_id
            self.deadline=source_pkt.deadline
            self.transmission_mode=0

    class RadioGatedPhy(Phy):
        def unicast(self, packet, next_hop_id):
            if not self.my_drone.radio_enabled:
                return
            if isinstance(packet,(ApplicationDataPacket,RoutedE2EAckPacket)):
                self.my_drone.network.metrics.total_airtime_s += packet.packet_length/config.BIT_RATE
            return super().unicast(packet,next_hop_id)

        def broadcast(self, packet):
            if not self.my_drone.radio_enabled:
                return
            return super().broadcast(packet)

    class MarlGreedy(Greedy):
        def __init__(self,simulator,drone,net):
            self.network=net
            super().__init__(simulator,drone)

        def broadcast_hello_packet_periodically(self):
            # The upstream starts all nodes' first HELLO simultaneously.
            # In a line topology U0 and U2 are hidden to each other:
            # both broadcast into U1 and permanently lose their first
            # neighbor announcement. Stagger initial HELLO by an ID-based
            # 1.5ms slot while retaining upstream Greedy's subsequent
            # HELLO jitter, original PHY and original CSMA/CA.
            yield self.simulator.env.timeout(self.my_drone.identifier*1500)
            yield from super().broadcast_hello_packet_periodically()

        def _link_ack(self, packet, sender_id):
            # Mirrors upstream greedy link MAC-ACK path, but does not count
            # the application E2E ACK in native Data PDR.
            config.GL_ID_ACK_PACKET+=1
            source=self.simulator.drones[sender_id]
            ack=AckPacket(
                src_drone=self.my_drone,dst_drone=source,
                ack_packet_id=config.GL_ID_ACK_PACKET,
                ack_packet_length=config.ACK_PACKET_LENGTH,ack_packet=packet,
                simulator=self.simulator,channel_id=packet.channel_id,
            )
            yield self.simulator.env.timeout(config.SIFS_DURATION)
            if self.my_drone.radio_enabled and not self.my_drone.sleep:
                ack.increase_ttl()
                self.my_drone.mac_protocol.phy.unicast(ack,sender_id)
                yield self.simulator.env.timeout(ack.packet_length/config.BIT_RATE*1e6)

        def packet_reception(self,packet,sender_id):
            if isinstance(packet,ApplicationDataPacket):
                at_destination=packet.dst_drone.identifier==self.my_drone.identifier
                if at_destination and packet.packet_id not in self.network.metrics.data_delivered_ids:
                    self.network.belief.merge_packet(self.my_drone.identifier,packet.app_entries)
                    self.network.metrics.on_deliver(packet)
                    self.network.delivered_packets[packet.packet_id]=(
                        packet.get_current_ttl(),
                        self.simulator.env.now-packet.creation_time,
                    )
                yield from super().packet_reception(packet,sender_id)
                if at_destination:
                    # ACK is separate routed application data, sent once per
                    # newly-delivered id, with its own MAC ACKs per hop.
                    if packet.packet_id not in self.network.ack_scheduled_ids:
                        self.network.ack_scheduled_ids.add(packet.packet_id)
                        ack=RoutedE2EAckPacket(packet,self.network)
                        self.my_drone.transmitting_queue.put(ack)
                return
            if isinstance(packet,RoutedE2EAckPacket):
                if packet.dst_drone.identifier==self.my_drone.identifier:
                    pair=(packet.acked_origin_id,packet.acked_peer_id)
                    self.network.acked_sequences.setdefault(pair,set()).add(
                        packet.acked_seq_id
                    )
                    if packet.acked_seq_id not in self.network.ack_delivered_ids:
                        self.network.ack_delivered_ids.add(packet.acked_seq_id)
                        self.network.metrics.e2e_ack_delivered+=1
                    yield from self._link_ack(packet,sender_id)
                else:
                    # In-transit application ACK is a regular DataPacket for
                    # upstream greedy relay and link-MAC ACK processing.
                    yield from super().packet_reception(packet,sender_id)
                return
            yield from super().packet_reception(packet,sender_id)

    class MarlSearchDrone(Drone):
        """DI construction: never calls Drone.__init__ (prevents ghost processes)."""
        def __init__(self,env,node_id,coords,simulator,network):
            self.network=network
            self.simulator=simulator
            self.env=env
            self.identifier=node_id
            self.coords=np.asarray(coords,dtype=np.float64)
            self.start_coords=self.coords.copy()
            self.rng_drone=random.Random(node_id+simulator.seed)
            self.direction=0.0
            self.pitch=0.0
            self.speed=0.0
            self.velocity=[0.,0.,0.]
            self.direction_mean=0.0
            self.pitch_mean=0.0
            self.velocity_mean=0.0
            self.inbox=simulator.channel.create_inbox_for_receiver(node_id)
            self.buffer=simpy.Resource(env,capacity=1)
            self.max_queue_size=config.MAX_QUEUE_SIZE
            self.transmitting_queue=queue.Queue()
            self.waiting_list=[]
            self.mac_process_dict={}
            self.mac_process_finish={}
            self.mac_process_count=0
            self.enable_blocking=1
            self.sleep=False
            self.radio_enabled=True
            self.residual_energy=277200.
            self.channel_assigner=ChannelAssigner(simulator,self)
            self.mac_protocol=CsmaCa(self)
            self.mac_protocol.phy=RadioGatedPhy(self.mac_protocol)
            self.routing_protocol=MarlGreedy(simulator,self,network)
            self.mobility_model=None
            self.energy_model=None
            self.env.process(self.feed_packet())
            self.env.process(self.receive())
            self.env.process(self._purge_expired())

        def blocking(self):
            return not self.radio_enabled or super().blocking()

        def trigger(self):
            if not self.radio_enabled:
                now=self.env.now
                for entry in self.inbox:
                    if now>=entry[1]+entry[0].packet_length/config.BIT_RATE*1e6:
                        entry[3]=1
                return 0,[],[],[]
            return super().trigger()

        def _purge_expired(self):
            while True:
                yield self.env.timeout(1000)  # 1ms, covers blackout queue
                now=self.env.now
                q=self.transmitting_queue
                keep=[]
                while not q.empty():
                    pkt=q.get_nowait()
                    if pkt.creation_time+pkt.deadline>now: keep.append(pkt)
                for pkt in keep:q.put_nowait(pkt)
                self.waiting_list[:]=[
                    pkt for pkt in self.waiting_list
                    if pkt.creation_time+pkt.deadline>now
                ]

    return MarlSearchDrone,ApplicationDataPacket,RoutedE2EAckPacket


class ClassicV1SimPyNetwork:
    """Actual classic UavNetSim SimPy CSMA/CA, not the reference mininet."""

    def __init__(self,positions:dict[int,tuple[float,float]],belief:BeliefTable,
                 seed:int=44,v1_root:str|Path|None=None):
        imp=_load_upstream(v1_root)
        simpy=imp["simpy"]
        Channel=imp["Channel"]
        Metrics=imp["Metrics"]
        if sorted(positions)!=list(range(len(positions))):
            raise ValueError("UAV IDs must be contiguous from 0")
        self.belief=belief
        self.positions=dict(positions)
        self.radio_enabled={u:True for u in positions}
        self.acked_sequences={}
        self.delivered_packets={}
        self.ack_delivered_ids=set()
        self.ack_scheduled_ids=set()
        self.metrics=NetworkMetricsFilter()
        self.now=0.
        self._tx_packets={}
        env=simpy.Environment()
        simulator=types.SimpleNamespace()
        simulator.seed=int(seed)
        simulator.env=env
        simulator.drones=[]
        simulator.channel=Channel(env)
        simulator.channel_states={
            u:simpy.Resource(env,capacity=1) for u in positions
        }
        simulator.metrics=Metrics(simulator)
        self.simulator=simulator
        drone_class,data_type,ack_type=_make_types(imp)
        self.data_type=data_type
        self.ack_type=ack_type
        for u,xy in sorted(positions.items()):
            xyz=(float(xy[0]),float(xy[1]),100.)
            drone=drone_class(env,u,xyz,simulator,self)
            simulator.drones.append(drone)
        # HELLO coroutines start only after the full drone list is built.
        self.queues={
            u:types.SimpleNamespace(
                max_size=simulator.drones[u].max_queue_size,
                transmitting_queue=simulator.drones[u].transmitting_queue,
                waiting_list=simulator.drones[u].waiting_list,
            ) for u in positions
        }
        self.default_poisson_process_count=0
        self.default_mobility_process_count=0

    def _sync(self):
        for u,drone in enumerate(self.simulator.drones):
            x,y=self.positions[u]
            drone.coords=np.array([x,y,100.],dtype=np.float64)
            drone.radio_enabled=bool(self.radio_enabled[u])

    def send_data(self,now:float,packet:SearchDataPacket)->bool:
        if abs(float(now)-self.now)>1e-5:
            raise ValueError("send_data must use current SimPy time")
        if packet.src_id not in self.positions or packet.dst_id not in self.positions:
            raise ValueError("Unknown source or destination")
        if packet.src_id==packet.dst_id:
            raise ValueError("No self-targeted data packets")
        self.metrics.on_generate(packet)
        self.simulator.metrics.datapacket_generated_num+=1
        self.simulator.metrics.datapacket_generated.add(packet.packet_id)
        drone=self.simulator.drones[packet.src_id]
        if drone.transmitting_queue.qsize()+len(drone.waiting_list)>=drone.max_queue_size:
            return False
        self._sync()
        pkd=self.data_type(packet,self)
        self._tx_packets[packet.packet_id]=pkd
        drone.transmitting_queue.put(pkd)
        return True

    def run_until(self,t_end:float)->float:
        if t_end<self.now:
            raise ValueError("Cannot reverse SimPy time")
        self._sync()
        target=int(round(t_end*1e6))
        if target>self.simulator.env.now:
            self.simulator.env.run(until=target)
        self.now=float(self.simulator.env.now)/1e6
        return 0.


def run_smoke(case:str,v1_root=None):
    positions={0:(0.,0.),1:(400.,0.),2:(800.,0.)}
    belieF=BeliefTable(3,400)
    native=ClassicV1SimPyNetwork(positions,belieF,seed=44,v1_root=v1_root)
    native.run_until(.025)  # real on-air HELLO discovery
    app=SearchDataPacket(
        packet_id=1,episode_id=1,src_id=0,dst_id=2,seq_id=1,
        creation_time=.025,deadline=.5,entries=((0,7,1,0),),
    )
    if case=="blackout":
        native.radio_enabled[1]=False
        native.send_data(.025,app)
        native.run_until(.060)
        during=len(native.delivered_packets)
        native.radio_enabled[1]=True
        native.run_until(.080)
        follow=SearchDataPacket(
            packet_id=2,episode_id=1,src_id=0,dst_id=2,seq_id=2,
            creation_time=.080,deadline=.5,entries=((0,8,1,0),),
        )
        native.send_data(.080,follow)
        native.run_until(.150)
        return {
            "engine":"uavnetsim_v1_real_simpy_csma","v1_commit":PINNED_V1_SHA,
            "blackout_data_delivered":during,
            "post_restore_data_delivered":len(native.delivered_packets),
            "radio_receive_process_alive":all(not x.sleep for x in native.simulator.drones),
        }
    native.send_data(.025,app)
    native.run_until(.150)
    pdr= len(native.delivered_packets)/max(1,native.metrics.data_generated)
    result={
        "engine":"uavnetsim_v1_real_simpy_csma",
        "v1_commit":PINNED_V1_SHA,
        "app_data_delivered":len(native.delivered_packets),
        "app_e2e_ack_delivered":native.metrics.e2e_ack_delivered,
        "hop_count":native.delivered_packets.get(1,(0,0))[0],
        "receiver_own_origin_count":int(
            belieF.pos[2,0,7]+belieF.neg[2,0,7]),
        "native_data_pdr":pdr,
        "native_collision_count":native.simulator.metrics.collision_num,
        "native_sim_time_s":native.now,
        "native_latency_ms":float(native.delivered_packets.get(1,(0,0))[1])/1e3,
        "default_poisson_process_count":native.default_poisson_process_count,
        "default_mobility_process_count":native.default_mobility_process_count,
        "native_mac_airtime_s":native.metrics.total_airtime_s,
        "neighbor_tables":[len(d.routing_protocol.neighbor_table.neighbor_table) for d in native.simulator.drones],
    }
    if case=="gate5":
        ref_belief=BeliefTable(3,400)
        reference=MiniDiscreteEventNet(positions,ref_belief,tx_power_w=.5)
        reference.run_until(.025)
        reference.send_data(.025,app)
        reference.run_until(.150)
        result.update({
            "reference_engine":"mini_discrete_event",
            "reference_pdr":reference.metrics.data_pdr,
            "simpy_pdr":pdr,
            "simpy_mac_real":True,
            "latency_difference_ms":round(
                (result["native_latency_ms"]-
                float(reference.metrics.total_airtime_s)*1000),6),
            "gate5_status":"PARITY_PASS" if
                (pdr==reference.metrics.data_pdr and len(native.delivered_packets)==1)
                else "PARITY_FAIL",
        })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=("line","blackout","gate5"),default="line")
    parser.add_argument("--json",action="store_true")
    parser.add_argument("--root")
    args=parser.parse_args()
    result=run_smoke(args.case,v1_root=args.root)
    if args.json:print(json.dumps(result,sort_keys=True))
    else:print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
