"""2024 Zhang Eq. (9)-(18) source-faithful DS decision mode for M2.

Network transport remains a declared M2 extension of the paper.
"""
import numpy as np
import pytest

from uav_network.communication.m2_network import M2PacketStore
from uav_network.communication.m2_protocol import CooperativeWorldMap
from uav_network.configuration import Config
from uav_network.environment.correlated_mapping import Event
from uav_network.environment.m2_mission import CooperativeM2Env


def observation(source, tick, positive, cell=0):
    return Event(f"paper:{source}:{tick}:{cell}:{int(positive)}",
                 source, tick, cell, "target", bool(positive), .9, .1, 0.)


def paper_node():
    return CooperativeWorldMap(4, ds_threshold=.95,
                               confirmation_mode="ds_threshold",
                               evidence_mode="zhang_max_counts",
                               paper_mf=.8, paper_me=.8)


def test_zhang_two_positive_readings_immediately_confirm_without_sprt():
    node=paper_node()
    node.ingest(observation(0, 1, True), arrived_at=1)
    assert node.target_masses[0] == pytest.approx((.8, 0., .2))
    assert node.candidates.get(0) is None
    assert node.make_decision_notices(source=0) == []
    node.ingest(observation(0, 2, True), arrived_at=2)
    assert node.target_masses[0] == pytest.approx((.96, 0., .04))
    assert node.candidates[0].status == "verified"
    assert node.candidates[0].sprt_n == 0
    assert [n.status for n in node.make_decision_notices(source=0)] == ["verified"]


def test_zhang_max_across_uavs_not_sum_and_mixed_evidence():
    node=paper_node()
    node.ingest(observation(0, 1, True), arrived_at=1)
    node.ingest(observation(1, 2, True), arrived_at=2)
    assert node.target_masses[0,0] == pytest.approx(.8)  # MAX(1,1), not SUM(1,1)
    assert node.candidates.get(0) is None
    node.ingest(observation(1, 3, False), arrived_at=3)
    assert node.target_masses[0] == pytest.approx((4./9., 4./9., 1./9.))
    assert node.candidates.get(0) is None


def test_zhang_threshold_cross_down_emits_versioned_network_correction():
    node=paper_node()
    for t in (1,2):node.ingest(observation(0,t,True),arrived_at=t)
    verified=node.make_decision_notices(source=0)[0]
    node.ingest(observation(0,3,False),arrived_at=3)
    assert node.target_masses[0,0] < .95
    assert node.candidates.get(0) is None
    retracted=node.make_decision_notices(source=0)[0]
    assert retracted.status == "retracted"
    assert retracted.revision > verified.revision
    node.ingest(observation(0,4,True),arrived_at=4)
    renewed=node.make_decision_notices(source=0)
    assert renewed[-1].status == "verified"
    assert renewed[-1].revision > retracted.revision


def test_zhang_out_of_order_dedup_provenance_and_gcs_packet_gate():
    events=[observation(0,1,True), observation(0,2,True),observation(1,3,False)]
    ordered=paper_node();reversed_order=paper_node()
    for e in events:ordered.ingest(e,arrived_at=e.tick)
    for e in events[::-1]:reversed_order.ingest(e,arrived_at=8)
    np.testing.assert_allclose(ordered.target_masses,reversed_order.target_masses)
    assert reversed_order.events_seen == 3
    assert not reversed_order.ingest(events[0],arrived_at=9)

    uav=paper_node();gcs=paper_node()
    store=M2PacketStore([uav,gcs],buffer_bytes=8192)
    store.publish(0,events[:1],created_s=1)
    packets=store.publish(0,events[1:2],created_s=2)
    verified_packet=next(p for p in store.buffers[0].values() if p.kind=="report")
    assert uav.candidates[0].status=="verified"
    assert gcs.events_seen==0 and not gcs.claims
    # Transporting evidence alone does not award verified GCS delivery.
    for packet in ([packets] if not isinstance(packets,list) else packets):
        assert store.transfer(0,1,packet,3)
    assert gcs.events_seen==1 and not gcs.claims
    assert store.transfer(0,1,verified_packet,4)
    assert gcs.claims[(0,0)].status=="verified"


def test_m2_mission_defaults_to_paper_ds_and_independent_sensor():
    cfg=Config()
    cfg.world.motion_dimensions=3
    cfg.world.motion_safety="reject"
    cfg.sensor.model="bayesian"
    cfg.world.num_uavs=2
    cfg.world.num_targets=1
    cfg.world.num_obstacles=0
    cfg.energy.enabled=False
    env=CooperativeM2Env(cfg,max_scan_cells=1)
    try:
        obs,info=env.reset(seed=44)
        assert env.correlation_strength == 0.
        assert all(n.confirmation_mode=="ds_threshold" for n in env.nodes)
        assert all(n.evidence_mode=="zhang_max_counts" for n in env.nodes)
        assert obs.shape==(2,247)  # 6 UAVs use 255; contacts/mask depend on N
        assert "correct_gcs_delivery_recall" in info
    finally:
        env.close()


def test_relay_generated_ds_confirmation_notice_has_fresh_packet_ttl():
    """A late observation must not make the newly minted REPORT already expired."""
    nodes=[paper_node() for _ in range(3)]
    store=M2PacketStore(nodes,buffer_bytes=8192,ttl_s=300.)
    p1=store.publish(0,[observation(0,1,True)],created_s=1.)
    p2=store.publish(0,[observation(0,2,True)],created_s=2.)
    assert store.transfer(0,1,p1,250.)
    assert store.transfer(0,1,p2,250.)
    packets=[p for p in store.buffers[1].values()
             if p.kind=='report' and p.source_id==1]
    assert len(packets)==1
    report=packets[0]
    assert report.created_s==250.  # creation, not source sensing tick=2
    assert store.transfer(1,2,report,320.)
    assert nodes[2].claims[(1,0)].status=='verified'


def test_old_sprt_is_available_only_as_explicit_comparison_mode():
    old=CooperativeWorldMap(4,confirmation_mode="sprt",
                            evidence_mode="correlated_llr")
    assert old.confirmation_mode=="sprt"
    for modes in (("sprt","zhang_max_counts"),("ds_threshold","correlated_llr")):
        with pytest.raises(ValueError,match="unsupported"):
            CooperativeWorldMap(4,confirmation_mode=modes[0],evidence_mode=modes[1])
