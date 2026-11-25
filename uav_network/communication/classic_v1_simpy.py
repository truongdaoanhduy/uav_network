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
                    # ApplicationDataPacket subclasses upstream DataPacket,
                    # not local SearchDataPacket: NetworkMetricsFilter.on_deliver
                    # would silently ignore it, making genuine deliveries
                    # report PDR=0. Record the DATA id explicitly once.
                    self.network.metrics.data_delivered_ids.add(packet.packet_id)
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




class _QueueView:
    """Read-only compatibility view of a native UavNetSim drone packet queue."""
    def __init__(self, drone):
        self.drone=drone

    @property
    def max_size(self):
        return self.drone.max_queue_size

    @property
    def transmitting_queue(self):
        return tuple(self.drone.transmitting_queue.queue)

    @property
    def waiting_list(self):
        return tuple(self.drone.waiting_list)


class ClassicV1SimPyNetwork:
    """Actual classic UavNetSim SimPy CSMA/CA, not the reference mininet."""

    def __init__(self,positions:dict[int,tuple[float,float]],belief:BeliefTable,
                 seed:int=44,v1_root:str|Path|None=None,*,tx_power_w:float=0.5):
        # The pinned PHY reads a process-global power constant. Arbitrary
        # per-network overrides would mutate the radio of an existing native
        # network, so reject unsupported settings before loading upstream.
        if not math.isfinite(tx_power_w) or tx_power_w != 0.5:
            raise ValueError(
                "classic_simpy_v1 requires tx_power_w=0.5 W; arbitrary "
                "per-network power is unsupported by the process-global PHY"
            )
        imp=_load_upstream(v1_root)
        simpy=imp["simpy"]
        Channel=imp["Channel"]
        Metrics=imp["Metrics"]
        if sorted(positions)!=list(range(len(positions))):
            raise ValueError("UAV IDs must be contiguous from 0")
        self.belief=belief
        self.sinr_threshold_db=float(imp["config"].SNR_THRESHOLD)
        self.tx_power_w=float(imp["config"].TRANSMITTING_POWER)
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
        self.queues={u:_QueueView(simulator.drones[u]) for u in positions}
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


def run_smoke(case:str,v1_root=None,*,trace_start_seed=44,trace_num_seeds=6):
    if case=="warp_backoff":
        # Execute the REAL source CsmaCa.mac_send Python generator up to
        # its first timed wait; instrument randint bounds and timeout.
        # This verifies the actual (CW, DIFS+slots*slot_duration) path,
        # without simulating subsequent busy-channel freezes or ACK.
        imp=_load_upstream(v1_root)
        from uav_network.communication.gpu.classic_v1_backoff_warp import (
            compute_backoff_windows,
        )
        simpy=imp["simpy"]
        config=imp["config"]
        rng=np.random.default_rng(20261125)
        worlds,agents=128,6
        attempts=rng.integers(1,6,size=(worlds,agents),dtype=np.int32)
        backoff_draws=np.zeros((worlds,agents),dtype=np.int32)
        native_cw=np.zeros_like(backoff_draws)
        native_backoff_us=np.zeros_like(backoff_draws)
        native_wait_us=np.zeros_like(backoff_draws)
        for e in range(worlds):
            for u in range(agents):
                # Draw from same independently seeded Python.Random family
                # used by original CsmaCa constructor; then pass it through
                # a spy so that native generator remains the source of truth.
                seed=44+e
                attempt=int(attempts[e,u])
                cw_formula=(config.CW_MIN+1)*(2**(attempt-1))-1
                draw=random.Random(u+seed+5).randint(0,cw_formula-1)
                backoff_draws[e,u]=draw
                env=simpy.Environment()
                simulator=types.SimpleNamespace(
                    seed=seed,env=env,
                    channel_states={u:simpy.Resource(env,capacity=1)},
                    drones=[],
                )
                drone=types.SimpleNamespace(
                    identifier=u,env=env,simulator=simulator,
                    mac_process_finish={},
                )
                simulator.drones=[drone]
                mac=imp["CsmaCa"](drone)
                drone.mac_protocol=mac
                bounds=[]
                timeouts=[]
                class SpyRandom:
                    def randint(self,lo,hi):
                        bounds.append((lo,hi))
                        if not lo<=draw<=hi:
                            raise AssertionError("Random draw out of native bounds")
                        return draw
                mac.rng_mac=SpyRandom()
                timeout_original=env.timeout
                def capture(duration):
                    timeouts.append(duration)
                    return timeout_original(duration)
                env.timeout=capture
                pkd=types.SimpleNamespace(
                    packet_id=101,
                    number_retransmission_attempt={u:attempt},
                )
                gen=mac.mac_send(pkd)
                next(gen)       # invokes original randint and wait_idle
                gen.send(None)  # invokes original DIFS+backoff timeout
                if len(bounds)!=1 or len(timeouts)!=1:
                    raise AssertionError("Unexpected upstream CsmaCa.mac_send behavior")
                lo,hi=bounds[0]
                native_cw[e,u]=hi+1
                native_wait_us[e,u]=int(timeouts[0])
                native_backoff_us[e,u]=int(timeouts[0]-config.DIFS_DURATION)
        warped=compute_backoff_windows(
            attempts,backoff_draws,
            cw_min=config.CW_MIN,
            slot_duration_us=config.SLOT_DURATION,
            difs_duration_us=config.DIFS_DURATION,
            max_attempts=config.MAX_RETRANSMISSION_ATTEMPT,
            device="cpu",
        )
        mismatch=sum(int(np.count_nonzero(warped[name]!=native))
                     for name,native in (
                         ("contention_window",native_cw),
                         ("backoff_us",native_backoff_us),
                         ("wait_difs_plus_backoff_us",native_wait_us),
                     ))
        return {
            "source_sha":PINNED_V1_SHA,
            "mac_source":"mac.csma_ca.CsmaCa.mac_send",
            "engine":"warp_fp32_int32_cpu_backoff_vs_native_mac_generator",
            "worlds":worlds,"agents_per_world":agents,
            "backoff_mismatch_count":int(mismatch),
            "cw_min":config.CW_MIN,
            "slot_duration_us":config.SLOT_DURATION,
            "difs_duration_us":config.DIFS_DURATION,
            "max_retries":config.MAX_RETRANSMISSION_ATTEMPT,
            "max_native_wait_us":int(native_wait_us.max()),
            "full_mac_backoff_freeze_and_ack_parity":"NOT_TESTED",
            "limit":"RNG draw supplied; dynamic CCA/busy channel freeze/retry not executed",
        }
    if case=="warp_cca":
        # Compare with ACTUAL classic v1 CCA function, not an independently
        # reconstructed rule. The original reads occupied_resource.users and
        # ignores channel-number for carrier-state testing.
        imported=_load_upstream(v1_root)
        from utils.util_function import check_channel_availability
        from uav_network.communication.gpu.classic_v1_cca_warp import compute_cca_free
        rng=np.random.default_rng(20261125)
        worlds,nodes=128,6
        xyz=rng.uniform(0,1000,size=(worlds,nodes,3)).astype(np.float32)
        xyz[:,:,2]=100.
        active=rng.integers(0,2,size=(worlds,nodes),dtype=np.int32)
        warped=compute_cca_free(
            xyz,active,sensing_range_m=imported["config"].SENSING_RANGE,
            device="cpu",
        )
        source=np.empty((worlds,nodes),dtype=np.int32)
        for e in range(worlds):
            drones=[
                types.SimpleNamespace(identifier=i,coords=xyz[e,i])
                for i in range(nodes)
            ]
            states={
                i:types.SimpleNamespace(users=[object()] if active[e,i] else [])
                for i in range(nodes)
            }
            for u in range(nodes):
                source[e,u]=int(
                    check_channel_availability(states,drones[u],drones)
                )
        mismatches=np.count_nonzero(source!=warped)
        return {
            "engine":"warp_cpu_fp32_snapshot_carrier_sense",
            "source_sha":PINNED_V1_SHA,
            "original_source_function":"utils.util_function.check_channel_availability",
            "worlds":worlds,"nodes_per_world":nodes,
            "num_receiver_checks":worlds*nodes,
            "cca_mismatch_count":int(mismatches),
            "sensing_range_m":float(imported["config"].SENSING_RANGE),
            "blocked_checks":int(np.count_nonzero(source==0)),
            "free_checks":int(np.count_nonzero(source==1)),
            "full_warp_mac_parity_status":"NOT_TESTED",
            "limitations":"snapshot CCA only; no backoff, random scheduling, ACK or retransmissions",
        }
    if case=="collision_trace":
        # Capture the real upstream LOGGER (rather than recomputing MAC
        # event times from approximate PHY equations). Native classes and
        # CSMA/CA generators are unchanged. Filter by app packet IDs 1/2.
        import logging
        import re

        class PacketTraceHandler(logging.Handler):
            TX=re.compile(r"At time: ([0-9.]+).*?UAV: ([0-9]+) can send packet \(pkd id: ([0-9]+)\)")
            TIMEOUT=re.compile(r"At time: ([0-9.]+).*?ACK timeout of packet: ([0-9]+)")
            DROP=re.compile(r"At time: ([0-9.]+).*?Packet: ([0-9]+) is dropped!")
            ACK=re.compile(r"At time: ([0-9.]+).*?UAV: ([0-9]+) receives the ACK for data packet: ([0-9]+)")
            COLLISION=re.compile(r"At time: ([0-9.]+).*?Packets collision: Main node is: ([0-9]+)")
            def __init__(self):
                super().__init__(logging.INFO)
                self.events=[]
            def emit(self,record):
                msg=record.getMessage()
                for name,pattern in (
                    ("data_tx",self.TX),
                    ("data_ack_timeout",self.TIMEOUT),
                    ("data_drop",self.DROP),
                    ("data_ack_rx",self.ACK),
                    ("data_collision",self.COLLISION),
                ):
                    match=pattern.search(msg)
                    if match is None:
                        continue
                    values=match.groups()
                    at=float(values[0])
                    if at < 25000.:
                        return
                    if name in ("data_tx","data_ack_rx"):
                        u,pk=int(values[1]),int(values[2])
                    elif name=="data_collision":
                        u,pk=int(values[1]),0
                    else:
                        pk=int(values[1])
                        u={1:0,2:2}.get(pk,-1)
                    if name=="data_collision":
                        if u not in (0,2):
                            return
                    elif pk not in (1,2):
                        return
                    self.events.append(dict(type=name,time_us=at,uav_id=u,packet_id=pk))
                    return

        if trace_num_seeds<1 or trace_num_seeds>128 or trace_start_seed<0:
            raise ValueError("Invalid trace seed range")
        seeds=list(range(trace_start_seed,trace_start_seed+trace_num_seeds))
        events_by_seed=[]
        pdr=[]
        native_attempts=[]
        native_delivered_ids_by_seed=[]
        mac_draws_by_seed=[]
        source_metric_ids_by_seed=[]
        collision_counts=[]
        for seed in seeds:
            real=ClassicV1SimPyNetwork(
                {0:(0.,0.),1:(400.,0.),2:(800.,0.)},
                BeliefTable(3,400),seed=seed,v1_root=v1_root,
            )
            real.run_until(.025)
            # Wrap each upstream per-drone RNG after native HELLO startup.
            # Preserve *exact* random.Random state, draws and generator
            # semantics; only add a reproducible trace of numbers used.
            mac_draws=[]
            for drone in real.simulator.drones:
                original_rng=drone.mac_protocol.rng_mac
                uid=drone.identifier

                class TracedRng:
                    def __init__(self,upstream,uav_id):
                        self.upstream=upstream
                        self.uav_id=uav_id
                    def randint(self,lo,hi):
                        selected=self.upstream.randint(lo,hi)
                        mac_draws.append({
                            "time_us":float(real.simulator.env.now),
                            "uav_id":self.uav_id,"lo":lo,"hi":hi,
                            "slot":selected,
                        })
                        return selected

                drone.mac_protocol.rng_mac=TracedRng(original_rng,uid)
            handler=PacketTraceHandler()
            root_logger=logging.getLogger()
            root_logger.addHandler(handler)
            try:
                for index,src in enumerate((0,2),start=1):
                    pkt=SearchDataPacket(
                        packet_id=index,episode_id=1,src_id=src,dst_id=1,
                        seq_id=index,creation_time=.025,deadline=.5,
                        entries=tuple((src,c,1,0) for c in range(62)),
                    )
                    real.send_data(.025,pkt)
                real.run_until(.10)
            finally:
                root_logger.removeHandler(handler)
            events_by_seed.append(handler.events)
            mac_draws_by_seed.append(mac_draws)
            pdr.append(float(real.metrics.data_pdr))
            native_delivered_ids_by_seed.append(sorted(real.delivered_packets))
            source_metric_ids_by_seed.append(sorted(real.metrics.data_delivered_ids))
            collision_counts.append(int(real.simulator.metrics.collision_num))
            native_attempts.append(sum(
                sum(pkt.number_retransmission_attempt.values())
                for pkt in real._tx_packets.values()
            ))
        return {
            "source_sha":PINNED_V1_SHA,
            "scope":"actual_upstream_mac_events",
            "engine":"uavnetsim_v1_real_simpy_csma",
            "seeds":seeds,
            "native_pdr_by_seed":pdr,
            "native_delivered_ids_by_seed":native_delivered_ids_by_seed,
            "metric_delivered_ids_by_seed":source_metric_ids_by_seed,
            "native_attempts_by_seed":native_attempts,
            "native_collisions_by_seed":collision_counts,
            "events_by_seed":events_by_seed,
            "mac_draws_by_seed":mac_draws_by_seed,
            "notes":"Only 1/2 application DATA packets; logged native TX, ACK timeout, ACK reception, drops and same-source collision events after t=25ms. No HELLO/ACK reverse trace or RNG capture.",
        }
    if case=="collision":
        # Hidden terminal: UAV0 and UAV2 are 800m apart so neither senses
        # the other's CCA carrier (v1 range 750m), but both can reach UAV1.
        # Two 62-entry data payloads are offered simultaneously on a shared
        # channel. The reference mini-simulator does not model collisions.
        positions={0:(0.,0.),1:(400.,0.),2:(800.,0.)}
        native_pdr=[]
        reference_pdr=[]
        collision_counts=[]
        attempts=[]
        for seed in range(44,50):
            real=ClassicV1SimPyNetwork(
                positions,BeliefTable(3,400),seed=seed,v1_root=v1_root,
            )
            ref=MiniDiscreteEventNet(
                positions,BeliefTable(3,400),tx_power_w=.5,
            )
            real.run_until(.025)
            ref.run_until(.025)
            for index,src in enumerate((0,2),start=1):
                pkt=SearchDataPacket(
                    packet_id=index,episode_id=1,src_id=src,dst_id=1,
                    seq_id=index,creation_time=.025,deadline=.5,
                    entries=tuple((src,c,1,0) for c in range(62)),
                )
                real.send_data(.025,pkt)
                ref.send_data(.025,pkt)
            real.run_until(.10)
            ref.run_until(.10)
            native_pdr.append(float(real.metrics.data_pdr))
            reference_pdr.append(float(ref.metrics.data_pdr))
            collision_counts.append(int(real.simulator.metrics.collision_num))
            attempts.append(sum(
                sum(pkt.number_retransmission_attempt.values())
                for pkt in real._tx_packets.values()
            ))
        return {
            "source_sha":PINNED_V1_SHA,
            "engine":"uavnetsim_v1_real_simpy_csma",
            "num_seeds":len(native_pdr),
            "seeds":list(range(44,50)),
            "native_pdr_by_seed":native_pdr,
            "reference_pdr_by_seed":reference_pdr,
            "native_collision_by_seed":collision_counts,
            "native_total_collisions":sum(collision_counts),
            "native_tx_attempts_by_seed":attempts,
            "full_warp_mac_parity_status":"NOT_TESTED",
            "contention_case":"two hidden terminals simultaneous 62-entry packets to one relay",
        }
    if case=="warp_phy":
        imp=_load_upstream(v1_root)
        from phy.large_scale_fading import general_path_loss
        from uav_network.communication.gpu.classic_v1_friis_warp import (
            compute_classic_v1_friis_snr,
        )
        xyz=np.array([
            (0.,0.,100.),(400.,0.,100.),(800.,0.,100.),
            (250.,350.,100.),(700.,200.,100.),(100.,600.,100.),
        ],dtype=np.float32)
        c=imp["config"]
        native=np.zeros((len(xyz),len(xyz)),dtype=np.float64)
        for i in range(len(xyz)):
            for j in range(len(xyz)):
                rcv=types.SimpleNamespace(coords=xyz[i])
                tx=types.SimpleNamespace(coords=xyz[j])
                gain=general_path_loss(rcv,tx)
                native[i,j]=10*math.log10(c.TRANSMITTING_POWER*gain/c.NOISE_POWER)
        warped=compute_classic_v1_friis_snr(
            xyz,tx_power_w=c.TRANSMITTING_POWER,
            carrier_frequency_hz=c.CARRIER_FREQUENCY,
            noise_w=c.NOISE_POWER,device="cpu",
        )
        difference=np.abs(warped.astype(np.float64)-native)
        return {
            "engine":"warp_cpu_fp32_link_kernel",
            "source_sha":PINNED_V1_SHA,
            "device":"cpu","num_nodes":len(xyz),
            "max_abs_snr_error_db":float(difference.max()),
            "mean_abs_snr_error_db":float(difference.mean()),
            "connectivity_mismatch_count":int(
                np.count_nonzero(
                    (warped>=c.SNR_THRESHOLD)!=(native>=c.SNR_THRESHOLD)
                )
            ),
            "example_link_400m_snr_db":float(warped[0,1]),
            "example_link_800m_snr_db":float(warped[0,2]),
            "full_warp_mac_parity_status":"NOT_TESTED",
            "comparison_scope":"no interferers, line-of-sight Friis only",
        }
    if case=="observations":
        from uav_network.environment.zhang_xue_coop import (
            ZhangXueCooperativeEnv,ZhangXueConfig,
        )
        env=ZhangXueCooperativeEnv(
            ZhangXueConfig(horizon_steps=3),seed=44,
            network_backend="classic_simpy_v1",
        )
        initial=env._build_outputs()[0][0,9:12].tolist()
        env.net.run_until(.025)
        after=env._build_outputs()[0][0,9:12].tolist()
        table=env.net.simulator.drones[0].routing_protocol.neighbor_table.neighbor_table
        return {
            "engine":"uavnetsim_v1_real_simpy_csma",
            "local_peer_offset_before_hello":initial,
            "local_peer_offset_after_hello":after,
            "received_hello_from_peer":bool(1 in table),
        }
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
        "native_data_pdr":(
            len(native.simulator.metrics.datapacket_arrived)/
            max(native.simulator.metrics.datapacket_generated_num,1)
        ),
        "simpy_native_data_generated":native.simulator.metrics.datapacket_generated_num,
        "simpy_native_data_arrived":len(native.simulator.metrics.datapacket_arrived),
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
        ref_at=reference.data_delivery_times_s.get(app.packet_id)
        ref_latency_ms=(
            float((ref_at-app.creation_time)*1000)
            if ref_at is not None else float("nan")
        )
        simpy_latency_ms=float(result["native_latency_ms"])
        delta_ms=(
            simpy_latency_ms-ref_latency_ms
            if math.isfinite(ref_latency_ms) else float("nan")
        )
        ref_hops=int(reference.data_delivered_hops.get(app.packet_id,0))
        same_delivery=(len(native.delivered_packets)==len(reference.metrics.data_delivered_ids))
        same_hops=(int(result["hop_count"])==ref_hops)
        same_pdr=(abs(pdr-reference.metrics.data_pdr)<1e-12)
        latency_close=(math.isfinite(delta_ms) and abs(delta_ms)<=0.5)
        result.update({
            "reference_engine":"mini_discrete_event",
            "reference_pdr":float(reference.metrics.data_pdr),
            "simpy_pdr":float(pdr),
            "simpy_hops":int(result["hop_count"]),
            "reference_hops":ref_hops,
            "simpy_mac_real":True,
            "simpy_data_latency_ms":simpy_latency_ms,
            "reference_data_latency_ms":ref_latency_ms,
            "latency_difference_ms":round(delta_ms,6),
            "latency_absolute_tolerance_ms":0.5,
            "full_warp_parity_status":"NOT_TESTED",
            "gate5_status":(
                "PARTIAL_SIMPY_REFERENCE_PASS"
                if same_delivery and same_hops and same_pdr and latency_close
                else "PARTIAL_SIMPY_REFERENCE_FAIL"
            ),
            "gate5_scope":"single fixed 3-node 2-hop, CPU reference vs real v1 SimPy; NOT Warp",
        })
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=("line","blackout","gate5","observations","warp_phy","collision","collision_trace","warp_cca","warp_backoff"),default="line")
    parser.add_argument("--json",action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--trace-start-seed",type=int,default=44)
    parser.add_argument("--trace-num-seeds",type=int,default=6)
    args=parser.parse_args()
    result=run_smoke(
        args.case,v1_root=args.root,
        trace_start_seed=args.trace_start_seed,
        trace_num_seeds=args.trace_num_seeds,
    )
    if args.json:print(json.dumps(result,sort_keys=True))
    else:print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
