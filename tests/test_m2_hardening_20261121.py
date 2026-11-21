"""2D target belief M2 numerical, network and reporting safeguards."""
import math
import numpy as np
import pytest
from uav_network.communication.m2_protocol import CooperativeWorldMap, DecisionNotice
from uav_network.environment.correlated_mapping import Event
from uav_network.communication.m2_network import M2PacketStore
from uav_network.configuration import Config
from uav_network.environment.m2_mission import CooperativeM2Env
from uav_network.environment.env import Action


def ev(t,yes,*,p1=.8,p0=.2,rho=.55):
    return Event(f'event-{t}',t%2,t,0,'target',yes,p1,p0,rho)


def test_ds_stays_normalized_under_long_conflicting_correlated_evidence():
    m=CooperativeWorldMap(1)
    for t in range(1,201):
        m.ingest(ev(t,t<=100),arrived_at=t)
        mass=m.target_masses[0]
        assert np.isfinite(mass).all()
        assert np.all(mass>=-1e-12)
        np.testing.assert_allclose(mass.sum(),1,rtol=0,atol=1e-11)


def test_sign_of_conditional_llr_controls_ds_evidence_even_for_positive_observation():
    m=CooperativeWorldMap(1)
    m.ingest(Event('old',0,1,0,'target',False,.8,.2,.99),arrived_at=1)
    m.ingest(Event('new',1,2,0,'target',True,.05179188369065535,.017675667042795107,.99),arrived_at=2)
    assert m.target_masses[0][1]>m.target_masses[0][0]


def test_gcs_has_a_single_nonoracle_status_for_conflicting_uav_reports():
    gcs=CooperativeWorldMap(4)
    gcs.ingest_notice(DecisionNotice('verify',0,2,1,'verified',1),arrived_at=2)
    gcs.ingest_notice(DecisionNotice('reject',1,2,1,'rejected',1),arrived_at=3)
    assert gcs.consensus_status(2)=='disputed'
    assert gcs.consensus_status(1)=='unknown'
    # A later correction from the original source resolves the dispute.
    gcs.ingest_notice(DecisionNotice('retract',0,2,2,'retracted',2),arrived_at=4)
    assert gcs.consensus_status(2)=='rejected'


def test_policy_can_select_packet_type_without_overwriting_radio_power():
    cfg=Config();cfg.runtime.m2_enabled=True;cfg.world.motion_dimensions=3
    assert cfg.continuous_dim==5
    maps=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(maps,buffer_bytes=8192)
    store.publish(0,[Event('source0:1',0,1,0,'target',True,.8,.2,.55)],created_s=1)
    from uav_network.communication.m2_protocol import DecisionNotice
    store.publish_notice(0,DecisionNotice('verified',0,0,1,'verified',1))
    choice=store.select_packet(0,1,1,payload_choice='report')
    assert choice.kind=='report'
    choice2=store.select_packet(0,1,1,payload_choice='evidence')
    assert choice2.kind=='evidence'


def test_gcs_generates_provenance_preserving_map_delta():
    maps=[CooperativeWorldMap(4) for _ in range(3)]
    store=M2PacketStore(maps,buffer_bytes=8192)
    origin=store.publish(0,[Event('source0:1',0,1,0,'target',True,.8,.2,.55)],created_s=1)
    assert store.transfer(0,2,origin,2)
    delta=store.prepare_gcs_delta(now=2)
    assert delta is not None and delta.kind=='map_delta'
    assert delta.source_id==2
    assert delta.events[0].event_id==origin.events[0].event_id
    assert len(delta.map_snapshots)>0
    assert store.transfer(2,1,delta,3)
    np.testing.assert_allclose(maps[1].target_masses,maps[2].target_masses)


def test_relay_avoids_sending_identical_message_back_to_known_source():
    maps=[CooperativeWorldMap(2) for _ in range(3)]
    store=M2PacketStore(maps,buffer_bytes=8192)
    p=store.publish(0,[Event('source0:1',0,1,0,'target',True,.8,.2,.55)],created_s=1)
    assert store.transfer(0,2,p,2)
    assert not store.can_forward(2,0,store.buffers[2][p.message_id],3)




def test_radio_and_flight_share_same_battery():
    c=Config();c.world.motion_dimensions=3;c.sensor.model='bayesian';c.world.motion_safety='cbf'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.world.max_steps=4
    options={'positions':np.array([[500.,100.,100.],[600.,100.,100.]]),
             'targets':np.array([[800.,800.]]),'obstacles':np.empty((0,4))}
    env=CooperativeM2Env(c,max_scan_cells=1)
    try:
        env.reset(seed=44,options=options)
        assert env.radio.network.battery_j is env.battery_j
        before=env.battery_j.copy()
        action=Action(np.zeros((2,env.config.continuous_dim),np.float32),np.array([2,2],np.int64))
        action.continuous[:,-1]=1
        env.step(action)
        rf=env.radio.network.energy_j[:-1]
        np.testing.assert_allclose(before-env.battery_j,env.flight_energy_j+rf,atol=1e-5)
    finally:env.close()


def test_full_camera_fov_scans_more_cells_at_150_than_50m():
    c=Config();c.world.motion_dimensions=3;c.sensor.model='bayesian';c.world.motion_safety='cbf'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.world.max_steps=4;c.energy.enabled=False
    positions=np.array([[500.,500.,50.],[700.,500.,50.]])
    options={'positions':positions,'targets':np.array([[900.,900.]]),'obstacles':np.empty((0,4))}
    e=CooperativeM2Env(c,max_scan_cells=None)
    try:
        e.reset(seed=44,options=options)
        low=e.nodes[0].coverage.sum()
        positions[:,2]=150.
        e.reset(seed=44,options=options)
        high=e.nodes[0].coverage.sum()
        assert high>low*4,(low,high)
    finally:e.close()


def test_radio_policy_payload_branch_controls_actual_uavnetsim_packet_type():
    from types import SimpleNamespace
    from uav_network.communication.m2_network import M2Radio
    cfg=Config();cfg.runtime.m2_enabled=True;cfg.world.motion_dimensions=3
    cfg.world.num_uavs=2;cfg.world.num_targets=1;cfg.world.num_obstacles=0
    cfg.network.gcs_range_m=cfg.network.peer_range_m=500
    cfg.energy.enabled=False
    pos=np.array([[200.,100.,100.],[300.,100.,100.]])
    scene=SimpleNamespace(positions=pos.copy(),gcs=np.array([200.,0.,0.]),
                          targets=np.array([[900.,900.]]),obstacles=np.empty((0,3)),
                          obstacle_heights=np.empty(0))
    maps=[CooperativeWorldMap(2) for _ in range(3)]
    store=M2PacketStore(maps,buffer_bytes=8192)
    radio=M2Radio(cfg,store,scene,seed=44)
    try:
        store.publish(0,[Event('evidence',0,0,0,'target',True,.8,.2,.5)],created_s=0)
        store.publish_notice(0,DecisionNotice('claim',0,0,1,'verified',0))
        radio.step(np.array([2,0]),pos,time_s=1,power_actions=np.ones(2),payload_actions=np.array([.7,0.]))
        assert maps[-1].claims[(0,0)].status=='verified'
        assert maps[-1].events_seen==0  # report selected BEFORE evidence, learned category
        radio.step(np.array([2,0]),pos,time_s=2,power_actions=np.ones(2),payload_actions=np.array([-.8,0.]))
        assert maps[-1].events_seen==1
    finally:radio.close()






def test_map_metrics_separate_candidate_recall_from_verified_gcs_recall():
    from uav_network.communication.m2_protocol import MultiLayerMapMetrics
    m=CooperativeWorldMap(4)
    m.target_masses[0]=[.98,0.,.02]
    t=np.array([True,False,False,False]);o=np.zeros(4,bool)
    x=MultiLayerMapMetrics.evaluate(m,t,at_tick=5)
    assert x['target_recall']==1.0  # backward-compatible candidate threshold metric
    assert x['target_auprc']>0
    assert x['target_brier']>=0
    assert x['total_uncertainty']>=0
    assert x['target_calibration_ece']>=0
    assert m.consensus_status(0)=='unknown'  # candidate threshold != verified




def test_m2_actor_observation_contains_no_instantaneous_link_oracle():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.world.max_steps=4;c.energy.enabled=False
    c.sensor.model='bayesian'
    options={'positions':np.array([[500.,100.,100.],[550.,100.,100.]]),
             'targets':np.array([[900.,900.]]),'obstacles':np.empty((0,4))}
    e=CooperativeM2Env(c,max_scan_cells=1)
    try:
        e.reset(seed=44,options=options)
        initial=e.local_observations()[0,-3:]
        assert list(initial)==[1.,1.,1.]
        e.positions[1,0]=950.  # remote UAV moves; no radio packet received
        after=e.local_observations()[0,-3:]
        assert list(after)==[1.,1.,1.]
    finally:e.close()






def test_gcs_candidate_alone_never_counts_as_a_delivered_verified_report():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.energy.enabled=False
    options={'positions':np.array([[500.,100.,100.],[600.,100.,100.]]),
             'targets':np.array([[800.,800.]]),'obstacles':np.empty((0,4))}
    env=CooperativeM2Env(c,max_scan_cells=1)
    try:
        env.reset(seed=44,options=options)
        cell=int(np.flatnonzero(env._truth_target)[0])
        for t in range(1,11):
            env.gcs_map.ingest(Event(f'gcs-ev-{t}',t%2,t,cell,'target',True,.9,.1,.2),arrived_at=t)
        assert env.gcs_map.candidates[cell].status=='verified'
        assert len(env.gcs_map.report_history)==0
        assert env._current_correct_delivery_count()==0
    finally:env.close()


def test_sender_receives_ack_contact_only_after_successful_radio_delivery():
    nodes=[CooperativeWorldMap(2) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=4096)
    p=store.publish(0,[Event('ack-e1',0,1,0,'target',True,.8,.2,.4)],created_s=1)
    assert nodes[0].link_contacts=={}
    assert store.transfer(0,1,p,2)
    assert nodes[0].link_contacts[1]==2
    assert nodes[1].link_contacts[0]==2
    assert not store.transfer(0,1,p,3)
    assert nodes[0].link_contacts[1]==2




def test_m2_opt_in_cannot_silently_enter_legacy_or_warp_training_backend():
    from uav_network.environment.vector import make_vector
    c=Config();c.world.motion_dimensions=3;c.sensor.model='bayesian';c.runtime.m2_enabled=True
    c.world.motion_safety='cbf';c.runtime.environment_backend='warp_packet'
    with pytest.raises(NotImplementedError,match='M2'):
        make_vector(c)
