"""Regression tests for multi-hop custody, versioned claims and correct GCS reward."""
import numpy as np
import pytest

from uav_network.communication.m2_protocol import CooperativeWorldMap, DecisionNotice
from uav_network.communication.m2_network import M2PacketStore
from uav_network.environment.correlated_mapping import Event
from uav_network.environment.m2_mission import CooperativeM2Env
from uav_network.configuration import Config


def ev(i,t,cell,positive=True):
    return Event(f'audit:{i}:{t}:{cell}:{int(positive)}',i,t,cell,'target',positive,.9,.1,0.)


def test_claim_revisions_never_reset_after_candidate_retracted_and_rediscovered():
    node=CooperativeWorldMap(1, ds_threshold=.95)
    node.ingest(ev(0,10,0),arrived_at=10)
    node.ingest(ev(0,11,0),arrived_at=11)
    first=node.make_decision_notices(source=0)
    assert first[-1].revision==1
    for t in (1,2,3,4): node.ingest(ev(1,t,0,False),arrived_at=12)
    retract=node.make_decision_notices(source=0)
    assert retract[-1].status=='retracted'
    assert retract[-1].revision>first[-1].revision
    for t in range(12,30):
        node.ingest(ev(0,t,0),arrived_at=t+1)
        if node.candidates.get(0):break
    renewed=node.make_decision_notices(source=0)
    assert renewed[-1].status=='pending'
    assert renewed[-1].revision>retract[-1].revision
    gcs=CooperativeWorldMap(1)
    for notice in first+retract+renewed: assert gcs.ingest_notice(notice,arrived_at=50)
    assert gcs.claims[(0,0)].revision==renewed[-1].revision


def test_reacquired_packet_restores_forwarding_custody_after_eviction_without_recounting():
    nodes=[CooperativeWorldMap(12) for _ in range(4)]
    store=M2PacketStore(nodes,buffer_bytes=1024,ttl_s=100)
    first=store.publish(0,[ev(0,1,0)],created_s=1)
    assert store.transfer(0,1,first,2)
    assert store.transfer(0,2,first,2)
    assert nodes[1].events_seen==1
    for cell in range(1,11):
        store.publish(1,[ev(1,cell+2,cell)],created_s=cell+2)
    assert first.message_id not in store.buffers[1], 'eviction setup failed'
    assert store.transfer(2,1,store.buffers[2][first.message_id],25)
    assert first.message_id in store.buffers[1], 'duplicate receipt must restore forwarding custody'
    assert nodes[1].events_seen==11, 'recovered event must not be fused twice'
    assert store.can_forward(1,3,store.buffers[1][first.message_id],26)


def test_gcs_report_reward_waits_for_consensus_and_is_not_paid_twice():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0
    c.world.max_steps=4;c.energy.enabled=False
    opts={'positions':np.array([[200.,200.,100.],[300.,200.,100.]]),
          'targets':np.array([[500.,500.]]),'obstacles':np.empty((0,4))}
    env=CooperativeM2Env(c,max_scan_cells=1)
    try:
        env.reset(seed=45,options=opts)
        cell=int(np.flatnonzero(env._truth_target)[0])
        gcs=env.gcs_map
        gcs.ingest_notice(DecisionNotice('claim1',0,cell,1,'verified',1),arrived_at=1)
        gcs.ingest_notice(DecisionNotice('claim2',1,cell,1,'rejected',1),arrived_at=1)
        assert gcs.consensus_status(cell)=='disputed'
        env._reward(np.zeros(2),0)
        assert env.last_reward_components['report']==0
        gcs.ingest_notice(DecisionNotice('retract2',1,cell,2,'retracted',2),arrived_at=2)
        env._reward(np.zeros(2),0)
        assert env.last_reward_components['report']==50
        env._reward(np.zeros(2),0)
        assert env.last_reward_components['report']==0
    finally:env.close()


def test_gcs_map_scores_do_not_use_truth_in_actor_or_critic():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.world.max_steps=3
    opts={'positions':np.array([[200.,200.,100.],[300.,200.,100.]]),
          'targets':np.array([[800.,800.]]),'obstacles':np.empty((0,4))}
    env=CooperativeM2Env(c,max_scan_cells=2)
    try:
        obs,_=env.reset(seed=44,options=opts)
        prev=obs.copy()
        env._truth_target[:]=~env._truth_target
        np.testing.assert_array_equal(env.local_observations(),prev)
        np.testing.assert_array_equal(env.state(),prev.ravel())
    finally:env.close()


def test_partial_fov_tangency_likelihoods_are_stable_even_for_tiny_nonzero_overlap():
    from uav_network.environment.correlated_mapping import CellEvidence, SharedBiasLikelihood
    from math import log1p, expm1
    q=1e-16
    e=CellEvidence.from_footprint(.90,.1,q)
    assert e is not None
    assert e.p0>0 and e.p1>e.p0
    assert e.p0 == pytest.approx(-expm1(q*log1p(-.1)),rel=1e-9)
    observed=Event('tiny-overlap',0,1,0,'target',False,e.p1,e.p0,.5)
    assert np.isfinite(SharedBiasLikelihood().observe(observed))


def test_all_new_observations_do_not_replay_all_past_cell_history_quadratically(monkeypatch):
    from uav_network.environment.correlated_mapping import SharedBiasLikelihood
    counter={'observations':0}
    original=SharedBiasLikelihood.observe
    def counted(self,event):
        counter['observations']+=1
        return original(self,event)
    monkeypatch.setattr(SharedBiasLikelihood,'observe',counted)
    node=CooperativeWorldMap(1)
    for tick in range(1,201):
        node.ingest(ev(tick%3,tick,0,bool(tick%7)),arrived_at=tick)
    assert counter['observations']<=240, 'in-order sensing must not incur O(n^2) replay'


def test_m2_environment_does_not_mutate_caller_config_or_legacy_action_abi():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    assert c.runtime.m2_enabled is False
    assert c.continuous_dim==4
    env=CooperativeM2Env(c,max_scan_cells=2)
    try:
        assert env.config.runtime.m2_enabled is True
        assert env.config.continuous_dim==5
        assert c.runtime.m2_enabled is False
        assert c.continuous_dim==4
    finally:env.close()


@pytest.mark.parametrize('seed',[1,7,42,100])
def test_random_delayed_multihop_event_orders_are_numerically_equivalent(seed):
    rng=np.random.default_rng(seed)
    events=[Event(f'{seed}/{tick}',tick%3,tick,int(rng.integers(0,7)),
                  'target',bool(rng.integers(0,2)),.88,.12,.72)
            for tick in range(1,101)]
    ordered=CooperativeWorldMap(7)
    scrambled=CooperativeWorldMap(7)
    for event in events: ordered.ingest(event,arrived_at=event.tick)
    for event in rng.permutation(events).tolist():scrambled.ingest(event,arrived_at=1000)
    np.testing.assert_allclose(ordered.target_masses,scrambled.target_masses,atol=5e-13)
    assert ordered.events_seen==scrambled.events_seen==100
    assert {c:s.status for c,s in ordered.candidates.items()}=={c:s.status for c,s in scrambled.candidates.items()}
    assert np.all(np.abs(scrambled.target_masses.sum(axis=1)-1)<1e-12)


def test_actor_local_patch_distinguishes_candidate_pending_verified_and_rejected():
    from uav_network.communication.m2_protocol import CandidateState
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=0;c.world.max_steps=3
    opts={'positions':np.array([[300.,300.,100.],[400.,300.,100.]]),
          'targets':np.array([[800.,800.]]),'obstacles':np.empty((0,4))}
    env=CooperativeM2Env(c,max_scan_cells=1)
    try:
        env.reset(seed=44,options=opts)
        cell=env._cell_xy(env.positions[0,:2])
        base=env.local_observations()[0].copy()
        node=env.nodes[0]
        for status in ('pending','verified','rejected'):
            node.candidates[cell]=CandidateState(0,'candidate',(0,0,'candidate'),status=status)
            if status=='pending':pending=env.local_observations()[0].copy()
            if status=='verified':verified=env.local_observations()[0].copy()
            if status=='rejected':rejected=env.local_observations()[0].copy()
        assert not np.array_equal(base,pending)
        assert not np.array_equal(pending,verified)
        assert not np.array_equal(verified,rejected)
    finally:env.close()


def test_declared_radio_packet_bytes_cover_serialized_evidence_notice_and_gcs_map():
    import json
    from dataclasses import asdict
    nodes=[CooperativeWorldMap(5) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192)
    p=store.publish(0,[ev(0,1,1)],created_s=1)
    notice=store.publish_notice(0,DecisionNotice('notice',0,1,1,'verified',2))
    assert store.transfer(0,2,p,2)
    delta=store.prepare_gcs_delta(now=3)
    def serialized_size(packet):
        header={'message_id':packet.message_id,'source_id':packet.source_id,
                'created_s':packet.created_s,
                'events':[asdict(e) for e in packet.events],
                'hops':packet.hops,'kind':packet.kind,
                'map_snapshots':packet.map_snapshots}
        return len(json.dumps(header,separators=(',',':')).encode())
    for pkt in (p,notice,delta):
        assert pkt.size_bytes>=serialized_size(pkt), (pkt.kind,pkt.size_bytes,serialized_size(pkt))
        assert pkt.size_bytes<=1024


def test_large_evidence_event_ids_are_packetized_by_actual_wire_budget():
    import json
    from dataclasses import asdict
    nodes=[CooperativeWorldMap(10) for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192)
    events=[Event(f'uuid-long-{i}-'+('X'*180),0,i+1,i,'target',True,.8,.2,.4)
            for i in range(6)]
    packets=store.publish(0,events,created_s=1,payload_bytes=1024)
    if not isinstance(packets,list):packets=[packets]
    assert len(packets)>1
    for pkt in packets:
        body={'message_id':pkt.message_id,'source_id':pkt.source_id,'created_s':pkt.created_s,
              'events':[asdict(e) for e in pkt.events],
              'hops':pkt.hops,'kind':pkt.kind,'map_snapshots':pkt.map_snapshots}
        size=len(json.dumps(body,separators=(',',':')).encode())
        assert size<=pkt.size_bytes<=1024
    assert sum(len(pkt.events) for pkt in packets)==6


def test_gcs_does_not_resend_already_acknowledged_target_evidence_in_new_delta():
    nodes=[CooperativeWorldMap(5) for _ in range(4)]
    store=M2PacketStore(nodes,buffer_bytes=8192,ttl_s=6)
    original=store.publish(0,[ev(0,1,2)],created_s=1)
    assert store.transfer(0,3,original,2)
    first=store.prepare_gcs_delta(now=2)
    assert first is not None
    assert store.transfer(3,1,first,3)
    assert nodes[1].events_seen==1
    store.expire(now=9)
    repeated=store.prepare_gcs_delta(now=9)
    assert repeated is not None and repeated.message_id!=first.message_id
    assert not store.can_forward(3,1,repeated,9), 'acknowledged 2D evidence must not be sent to same UAV again'
    assert store.can_forward(3,2,repeated,9), 'a different UAV still needs the GCS delta'


def test_tall_physical_obstacle_blocks_target_sensor_without_building_obstacle_map():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety='cbf';c.sensor.model='bayesian'
    c.world.num_uavs=2;c.world.num_targets=1;c.world.num_obstacles=1;c.world.max_steps=2
    c.sensor.grid_cell_m=25.;c.energy.enabled=False
    scene={'positions':np.array([[450.,540.,100.],[800.,800.,100.]]),
           'targets':np.array([[535.,540.]]),
           'obstacles':np.array([[490.,540.,12.,120.]])}
    seen=[]
    for obstacle_occlusion in (True,False):
        c.sensor.obstacle_occlusion=obstacle_occlusion
        env=CooperativeM2Env(c,max_scan_cells=None)
        try:
            env.reset(seed=44,options=scene)
            cell=env._cell_xy(scene['targets'][0])
            seen.append(cell in [event.cell for event in env.nodes[0].events.values()])
            assert not hasattr(env.nodes[0],'obstacle_masses')
        finally:env.close()
    assert seen==[False,True]
