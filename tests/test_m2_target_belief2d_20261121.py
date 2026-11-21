"""Ground-only target belief mapping contract for the 2024 Zhang-inspired M2 mission."""
import numpy as np
import pytest
from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.environment.m2_mission import CooperativeM2Env
from uav_network.communication.m2_protocol import CooperativeWorldMap, MultiLayerMapMetrics
from uav_network.communication.m2_network import M2PacketStore
from uav_network.environment.correlated_mapping import Event


def cfg():
    c=Config(); c.world.motion_dimensions=3; c.world.motion_safety='cbf'; c.sensor.model='bayesian'
    c.world.map_size_m=1000.; c.sensor.grid_cell_m=25.
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=1
    c.world.max_steps=6;c.energy.enabled=False
    return c


def scene():
    return {'positions':np.array([[400.,450.,100.],[500.,450.,100.]]),
            'targets':np.array([[405.,490.]]),
            'obstacles':np.array([[600.,550.,18.,70.]])}


def event(i,t,cell,positive=True):
    return Event(f'e:{i}:{t}:{cell}',i,t,cell,'target',positive,.9,.1,.4)


def test_target_belief_2d_only_has_three_masses_and_no_obstacle_reconstruction():
    e=CooperativeM2Env(cfg(),max_scan_cells=3)
    try:
        obs,info=e.reset(seed=44,options=scene())
        assert e.grid_n==40 and e.num_cells==1600
        assert e.nodes[0].target_masses.shape==(1600,3)
        assert np.all(e.nodes[0].target_masses.sum(axis=1)==pytest.approx(1.))
        assert not hasattr(e.nodes[0],'obstacle_masses')
        assert not hasattr(e.nodes[0],'obstacle_tracks')
        assert not hasattr(e.nodes[0],'geometry_events')
        assert all(x.layer=='target' for node in e.nodes for x in node.events.values())
        assert not any('obstacle' in k or '3d' in k or 'voxel' in k for k in info['gcs_map'])
        assert not any('obstacle' in k or '3d' in k or 'voxel' in k for k in info)
        assert e.scene.obstacle_heights.shape==(1,)  # real 3D collision geometry remains
    finally:e.close()


def test_target_belief_metrics_only_target_and_uncertainty():
    m=CooperativeWorldMap(4)
    x=MultiLayerMapMetrics.evaluate(m,[True,False,False,False],at_tick=0)
    assert 'target_brier' in x and 'target_auprc' in x
    assert 'coverage' in x and 'total_uncertainty' in x
    assert x['total_uncertainty']==pytest.approx(4.)
    assert not any('obstacle' in key or 'voxel' in key for key in x)
    with pytest.raises(ValueError,match='target'):
        m.ingest(Event('bad',0,1,0,'obstacle',True,.8,.2,.5),arrived_at=1)


def test_gcs_fused_target_belief_delta_is_sent_back_through_multi_hop_without_oracle():
    nodes=[CooperativeWorldMap(4) for _ in range(4)]
    store=M2PacketStore(nodes,buffer_bytes=8192)
    assert np.all(nodes[-1].target_masses[:,2]==1)
    pkt=store.publish(0,[event(0,1,1,True)],created_s=1.)
    assert store.transfer(0,3,pkt,2.)
    assert nodes[-1].target_masses[1,0]>0
    assert nodes[1].target_masses[1,0]==0
    delta=store.prepare_gcs_delta(now=3.)
    assert delta is not None and delta.kind=='map_delta'
    assert len(delta.map_snapshots)==1
    cell,mass=delta.map_snapshots[0]
    assert cell==1
    np.testing.assert_allclose(mass,nodes[-1].target_masses[cell])
    assert store.transfer(3,2,delta,4.)
    assert nodes[1].target_masses[1,0]==0
    assert store.transfer(2,1,store.buffers[2][delta.message_id],5.)
    np.testing.assert_allclose(nodes[1].target_masses,nodes[-1].target_masses)
    assert nodes[1].events_seen==1  # event IDs remain identical on every relay
    assert not store.transfer(2,1,store.buffers[2][delta.message_id],6.)


def test_m2_action_packet_classes_exclude_geometry_but_keep_bidirectional_map():
    assert M2PacketStore.payload_choice_from_action(-1.)=='evidence'
    assert M2PacketStore.payload_choice_from_action(-.4)=='evidence'
    assert M2PacketStore.payload_choice_from_action(.4)=='map_delta'
    assert M2PacketStore.payload_choice_from_action(.7)=='report'
    assert M2PacketStore.payload_choice_from_action(.99)=='correction'
    assert 'geometry' not in M2PacketStore.allowed_packet_kinds


def test_2d_belief_reward_contains_no_3d_mapping_component():
    e=CooperativeM2Env(cfg(),max_scan_cells=2)
    try:
        e.reset(seed=44,options=scene())
        a=Action(np.zeros((2,e.config.continuous_dim),np.float32),np.array([0,0],np.int64))
        _,_,_,_,info=e.step(a)
        assert 'obstacle_3d_progress' not in e.last_reward_components
        assert 'target_map_progress' in e.last_reward_components
        assert 'correct_gcs_delivery_recall' in info
        assert 'normalized_delivery_auc' in info
        assert 'gcs_map' in info
        assert info['gcs_events']==0
    finally:e.close()


def test_gcs_belief_map_reaches_out_of_range_uav_through_real_native_radio():
    from types import SimpleNamespace
    from uav_network.communication.m2_network import M2Radio
    c=cfg();c.world.num_uavs=3;c.world.num_obstacles=0
    c.network.peer_range_m=c.network.gcs_range_m=180.
    c.network.report_ttl_s=40.
    p=np.array([[500.,300.,30.],[500.,150.,30.],[600.,140.,30.]])
    scene=SimpleNamespace(positions=p.copy(),gcs=np.array([500.,0.,0.]),
                          targets=np.array([[850.,850.]]),obstacles=np.empty((0,3)),
                          obstacle_heights=np.empty(0))
    nodes=[CooperativeWorldMap(8) for _ in range(4)]
    store=M2PacketStore(nodes,buffer_bytes=16384)
    radio=M2Radio(c,store,scene,seed=44)
    try:
        store.publish(2,[event(2,0,3)],created_s=0)
        assert not radio.link_available(3,0,p)
        radio.step(np.array([0,0,3]),p,time_s=1,power_actions=np.ones(3))
        assert nodes[3].target_masses[3,0]>0
        assert nodes[0].events_seen==0
        delta=store.prepare_gcs_delta(now=1)
        assert delta and delta.kind=='map_delta'
        radio.step(np.array([0,0,0]),p,time_s=2,power_actions=np.ones(3),gcs_peer=1)
        assert nodes[1].target_masses[3,0]>0
        assert nodes[0].events_seen==0
        radio.step(np.array([0,1,0]),p,time_s=3,power_actions=np.ones(3))
        assert nodes[0].target_masses[3,0]>0
        np.testing.assert_allclose(nodes[0].target_masses,nodes[3].target_masses)
    finally:radio.close()


def test_gcs_2d_map_delta_batches_distinct_cells_with_actual_wire_size_bounded():
    nodes=[CooperativeWorldMap(20) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=16384)
    for cell in range(12):
        p=store.publish(0,[event(0,cell+1,cell)],created_s=cell+1)
        assert store.transfer(0,2,p,cell+1)
    delta=store.prepare_gcs_delta(now=13)
    assert delta.kind=='map_delta'
    assert 1<len(delta.map_snapshots)
    assert delta.size_bytes<=1024
    assert {c for c,_ in delta.map_snapshots}=={e.cell for e in delta.events}


def test_gcs_retries_expired_delta_instead_of_permanently_losing_sync():
    nodes=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192,ttl_s=6)
    p=store.publish(0,[event(0,1,2)],created_s=1)
    assert store.transfer(0,2,p,2)
    first=store.prepare_gcs_delta(now=2)
    assert first is not None
    store.expire(now=9)
    retry=store.prepare_gcs_delta(now=9)
    assert retry is not None
    assert retry.message_id!=first.message_id
    assert retry.events[0].event_id==first.events[0].event_id


def test_paper_cognitive_reward_uses_2d_ds_uncertainty_not_truth_brier():
    e=CooperativeM2Env(cfg(),max_scan_cells=1)
    try:
        e.reset(seed=44,options=scene())
        old=e.gcs_map.target_masses[:,2].sum()
        assert e._map_score()==pytest.approx(old/e.num_cells)
        e.gcs_map.ingest(event(0,1,4,True),arrived_at=1)
        e._reward(np.zeros(2),0)
        assert e.last_reward_components['target_map_progress']>0
        assert 'obstacle_3d_progress' not in e.last_reward_components
    finally:e.close()


def test_local_ds_information_gain_is_available_before_gcs_delivery():
    e=CooperativeM2Env(cfg(),max_scan_cells=3)
    try:
        e.reset(seed=44,options=scene())
        silent=Action(np.zeros((2,e.config.continuous_dim),np.float32),np.zeros(2,np.int64))
        e.step(silent)
        assert e.gcs_map.events_seen==0
        assert e.last_reward_components['local_information_gain']>0
    finally:e.close()


def test_gcs_2d_target_report_conflict_is_disputed_even_if_fused_map_is_confident():
    from uav_network.communication.m2_protocol import DecisionNotice
    gcs=CooperativeWorldMap(4)
    for tick in range(1,12):
        gcs.ingest(event(tick%2,tick,1,True),arrived_at=tick)
    assert gcs.candidates[1].status=='verified'
    gcs.ingest_notice(DecisionNotice('yes',0,1,1,'verified',12),arrived_at=12)
    gcs.ingest_notice(DecisionNotice('no',1,1,1,'rejected',13),arrived_at=13)
    assert gcs.consensus_status(1)=='disputed'
    assert gcs.ingest_notice(DecisionNotice('corrected',1,1,2,'retracted',14),arrived_at=14)
    assert gcs.consensus_status(1)=='verified'


def test_gcs_downlink_prioritizes_verified_report_before_bulk_2d_map_delta():
    from types import SimpleNamespace
    from uav_network.communication.m2_protocol import DecisionNotice
    from uav_network.communication.m2_network import M2Radio
    c=cfg();c.world.num_obstacles=0
    c.network.gcs_range_m=300.;c.network.peer_range_m=300.
    p=np.array([[200.,100.,30.],[300.,100.,30.]])
    scene=SimpleNamespace(positions=p.copy(),gcs=np.array([200.,0.,0.]),
                          targets=np.array([[800.,800.]]),obstacles=np.empty((0,3)),
                          obstacle_heights=np.empty(0))
    nodes=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192)
    radio=M2Radio(c,store,scene,seed=44)
    try:
        e=store.publish(0,[event(0,0,1)],created_s=0)
        d=store.publish_notice(0,DecisionNotice('claim',0,1,1,'verified',0))
        assert store.transfer(0,2,e,.1)
        assert store.transfer(0,2,d,.2)
        delta=store.prepare_gcs_delta(now=.3)
        assert delta is not None
        radio.step(np.zeros(2,dtype=int),p,time_s=1,power_actions=np.ones(2),gcs_peer=1)
        assert nodes[1].claims[(0,1)].status=='verified'
        assert nodes[1].events_seen==0
        radio.step(np.zeros(2,dtype=int),p,time_s=2,power_actions=np.ones(2),gcs_peer=1)
        assert nodes[1].events_seen==1
    finally:radio.close()


def test_gcs_avoids_downlinking_a_report_to_its_original_author():
    from uav_network.communication.m2_protocol import DecisionNotice
    nodes=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192)
    p=store.publish_notice(0,DecisionNotice('u0:c1',0,1,1,'verified',1))
    assert store.transfer(0,2,p,2)
    received=store.buffers[2][p.message_id]
    assert not store.can_forward(2,0,received,3)
