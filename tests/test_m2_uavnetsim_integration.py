"""Real pinned Native UavNetSim transport, not simulated direct map copies."""
from types import SimpleNamespace
import numpy as np
import pytest
from uav_network.configuration import Config
from uav_network.environment.correlated_mapping import Event
from uav_network.communication.m2_protocol import CooperativeWorldMap
from uav_network.communication.m2_network import M2PacketStore, M2Radio


def setup_network():
    c=Config();c.world.num_uavs=3;c.world.num_targets=1;c.world.num_obstacles=0
    c.network.peer_range_m=c.network.gcs_range_m=180.0
    c.network.report_ttl_s=40.;c.network.payload_bytes=1024
    c.energy.enabled=False
    pos=np.array([[500.,300.],[500.,150.],[600.,140.]])
    scene=SimpleNamespace(positions=pos.copy(),gcs=np.array([500.,0.]),
                          targets=np.array([[600.,600.]]),obstacles=np.empty((0,3)),
                          obstacle_heights=np.empty((0,)))
    nodes=[CooperativeWorldMap(16) for _ in range(4)]
    store=M2PacketStore(nodes, buffer_bytes=16384)
    radio=M2Radio(c,store,scene,seed=44)
    return pos,store,radio


def test_m2_packets_relay_to_gcs_and_downlink_real_radio():
    pos,store,radio=setup_network()
    try:
        event=Event("0:1:6",0,1,6,"target",True,.9,.1,.5)
        store.publish(0,[event],created_s=0)
        assert store.nodes[0].target_masses[6][0] > .0
        assert store.nodes[3].target_masses[6] == pytest.approx((0,0,1))
        assert not radio.link_available(3,0,pos)
        radio.step(np.array([1,0,0]),pos,time_s=1,power_actions=np.ones(3))
        assert store.nodes[1].target_masses[6][0] > 0
        assert store.nodes[3].target_masses[6] == pytest.approx((0,0,1))
        radio.step(np.array([0,3,0]),pos,time_s=2,power_actions=np.ones(3))
        assert store.nodes[3].target_masses[6][0] > 0
        assert store.nodes[2].target_masses[6] == pytest.approx((0,0,1))
        radio.step(np.array([0,0,0]),pos,time_s=3,power_actions=np.ones(3),gcs_peer=2)
        assert store.nodes[2].target_masses[6][0] > 0
        assert store.nodes[2].events_seen == 1
        assert store.nodes[3].events_seen == 1
        assert radio.network.metrics()["received_packets"] >= 3
    finally:
        radio.close()


def test_no_direct_gcs_map_copy_when_out_of_range():
    pos,store,radio=setup_network()
    try:
        event=Event("0:1:6",0,1,6,"target",True,.9,.1,.5)
        store.publish(0,[event],created_s=0)
        assert not radio.link_available(0,3,pos)
        assert radio.step(np.array([3,0,0]),pos,time_s=1,power_actions=np.ones(3)) >= 0
        assert store.nodes[3].events_seen == 0
    finally:
        radio.close()


def test_duplicate_fanout_does_not_reobserve_same_event():
    maps=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(maps,buffer_bytes=16384)
    evt=Event("u0:t1:c2",0,1,2,"target",True,.8,.2,.5)
    pkt=store.publish(0,[evt],created_s=0)
    assert store.transfer(0,1,pkt,1)
    assert maps[1].events_seen == 1
    assert not store.transfer(0,1,pkt,2)
    assert maps[1].events_seen == 1


def test_gcs_receives_verified_and_correction_notices_as_real_packets():
    from uav_network.communication.m2_protocol import DecisionNotice
    pos,store,radio=setup_network()
    try:
        store.publish_notice(1,DecisionNotice("u1:c3:rev1",1,3,1,"verified",0))
        radio.step(np.array([0,3,0]),pos,time_s=1,power_actions=np.ones(3))
        assert store.nodes[3].claims[(1,3)].status=="verified"
        store.publish_notice(1,DecisionNotice("u1:c3:rev2",1,3,2,"retracted",1))
        radio.step(np.array([0,3,0]),pos,time_s=2,power_actions=np.ones(3))
        assert store.nodes[3].claims[(1,3)].status=="retracted"
        assert len(store.nodes[3].report_history)==2
    finally:
        radio.close()


def test_map_store_finite_buffer_can_keep_recent_packets_without_oracle_map_copy():
    nodes=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=1024)
    for tick in range(1,35):
        event=Event(f"u0:{tick}:c0",0,tick,0,"target",bool(tick%2),.8,.2,.5)
        store.publish(0,[event],created_s=tick)
    assert store.bytes_used(0)<=1024
    assert nodes[0].events_seen==34
    assert nodes[-1].events_seen==0
    assert store.overflow_drops>0


def test_packet_hop_count_increases_on_each_actual_reception():
    nodes=[CooperativeWorldMap(4) for _ in range(4)]
    store=M2PacketStore(nodes,buffer_bytes=4096)
    event=Event("u0:1:0",0,1,0,"target",True,.8,.2,.3)
    source=store.publish(0,[event],created_s=0)
    assert store.transfer(0,1,source,1)
    forwarded=store.buffers[1][source.message_id]
    assert forwarded.hops==1
    assert store.transfer(1,3,forwarded,2)
    assert store.delivery_hops==[2]


def test_shared_latent_creates_nonzero_cross_uav_noise_correlation():
    from uav_network.environment.correlated_mapping import CorrelatedBinarySensor
    first=[];second=[]
    for seed in range(1800):
        sensor=CorrelatedBinarySensor(1,seed,.8)
        first.append(int(sensor.sample(0,.8,.2,True)))
        second.append(int(sensor.sample(0,.8,.2,True)))
    assert np.cov(first,second,ddof=0)[0,1]>.008


def test_gcs_downlink_inflight_does_not_look_like_a_uav_buffer():
    from uav_network.communication.backend import Transfer
    from uav_network.communication.m2_network import M2Envelope
    pos,store,radio=setup_network()
    try:
        envelope=M2Envelope("gcs-probe",3,0,128,tuple())
        transfer=Transfer(source=3,destination=2,report=envelope,pending={0:(17,10.)})
        radio.network.sessions[(3,"probe",2)]=transfer
        counts=radio.network.inflight_counts()
        assert counts.shape==(3,)
        assert counts.sum()==0
    finally:
        radio.close()


def test_high_priority_verified_report_is_forwarded_before_stale_map_data():
    from uav_network.communication.m2_protocol import DecisionNotice
    nodes=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=4096)
    for tick in range(5):
        store.publish(0,[Event(f"u0:{tick}:1",0,tick,1,"target",True,.8,.2,.5)],created_s=tick)
    previous=store.nodes[0].claims.get((0,1))
    revision=(previous.revision+1 if previous else 1)
    report=store.publish_notice(0,DecisionNotice("r:0:1",0,1,revision,"verified",5))
    assert next(iter(store.buffers[0]))==report.message_id


def test_communication_layer_records_only_actual_receive_not_unseen_links():
    pos,store,radio=setup_network()
    try:
        assert store.nodes[3].link_contacts=={}
        event=Event("u0:2:1",0,2,1,"target",True,.8,.2,.5)
        store.publish(0,[event],created_s=0)
        # Reading GCS/receiver state before the packet cannot reveal the event.
        assert store.nodes[3].link_contacts=={}
        radio.step(np.array([1,0,0]),pos,time_s=1,power_actions=np.ones(3))
        assert store.nodes[1].link_contacts[0] >= 0
        assert store.nodes[3].link_contacts=={}
        radio.step(np.array([0,3,0]),pos,time_s=2,power_actions=np.ones(3))
        assert store.nodes[3].link_contacts[1] >= 0
    finally:radio.close()
