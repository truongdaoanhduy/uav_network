"""Behavioral specifications for opt-in correlated cooperative world mapping."""
import math
import numpy as np
import pytest

from uav_network.environment.correlated_mapping import (
    CorrelatedBinarySensor, CellEvidence, Event, SharedBiasLikelihood,
)
from uav_network.communication.m2_protocol import (
    CooperativeWorldMap, MultiLayerMapMetrics, map_evidence_weight,
    packetize_events,
)


def obs(uid, tick, cell, result, *, p1=.9, p0=.1, layer="target", rho=.5):
    return Event(event_id=f"{uid}:{tick}:{cell}:{layer}", origin=uid, tick=tick,
                 cell=cell, layer=layer, positive=result,
                 p1=p1, p0=p0, correlation_strength=rho)


def test_sampler_correlated_with_calibrated_occupancy_marginals():
    fractions = []
    for occupied in (False, True):
        successes = 0
        for seed in range(2500):
            sampler = CorrelatedBinarySensor(num_cells=1, seed=seed, correlation_strength=.8)
            successes += int(sampler.sample(0, .8, .2, occupied))
        fractions.append(successes / 2500)
    assert fractions[0] == pytest.approx(.2, abs=.045)
    assert fractions[1] == pytest.approx(.8, abs=.045)


def test_shared_latent_correlates_uav_observations_and_likelihood_is_conditional():
    engine = SharedBiasLikelihood()
    first = engine.observe(obs(0,1,0,True,p1=.8,p0=.2,rho=.7))
    second = engine.observe(obs(1,2,0,True,p1=.8,p0=.2,rho=.7))
    assert first == pytest.approx(math.log(4.0))
    assert 0 < second < first
    assert engine.observation_count == 2


def test_likelihood_rejects_events_with_invalid_sensor_parameters():
    with pytest.raises(ValueError):
        SharedBiasLikelihood().observe(obs(0,1,0,True,p1=.5,p0=.5))
    with pytest.raises(ValueError):
        SharedBiasLikelihood().observe(obs(0,1,0,True,p1=.8,p0=.2,rho=1.4))


def test_map_ds_and_sprt_replay_out_of_order_is_deterministic():
    events = [obs(0,1,3,True,rho=.4),obs(1,2,3,True,rho=.4),
              obs(0,3,3,True,rho=.4),obs(1,4,3,True,rho=.4)]
    forward = CooperativeWorldMap(num_cells=8, ds_threshold=.95)
    reverse = CooperativeWorldMap(num_cells=8, ds_threshold=.95)
    for x in events: forward.ingest(x, arrived_at=x.tick)
    for x in events[::-1]: reverse.ingest(x, arrived_at=12)
    np.testing.assert_allclose(forward.target_masses[3],reverse.target_masses[3],atol=1e-12)
    assert forward.events_seen == reverse.events_seen == 4
    assert not reverse.ingest(events[0],arrived_at=20)
    assert reverse.events_seen == 4


def test_candidate_separated_from_verification_evidence():
    world = CooperativeWorldMap(num_cells=4,ds_threshold=.95,alpha=.05,beta=.05)
    for tick in range(1,10):
        world.ingest(obs(0,tick,1,True,rho=.3),arrived_at=tick)
        if world.candidates.get(1): break
    assert world.candidates[1].cutoff_tick == tick
    assert world.candidates[1].sprt_n == 0  # trigger observations must not be verification
    for next_tick in range(tick+1,tick+8):
        world.ingest(obs(1,next_tick,1,True,rho=.3),arrived_at=next_tick)
        if world.candidates[1].status == "verified": break
    assert world.candidates[1].status == "verified"
    assert world.candidates[1].sprt_n > 0


def test_gcs_world_map_never_updates_without_delivery():
    uav = CooperativeWorldMap(num_cells=4)
    gcs = CooperativeWorldMap(num_cells=4)
    observation=obs(0,2,1,True)
    uav.ingest(observation,arrived_at=2)
    assert uav.target_masses[1][0] > 0
    assert gcs.target_masses[1] == pytest.approx((0,0,1))
    gcs.ingest(observation,arrived_at=5)
    assert gcs.target_masses[1] == pytest.approx(uav.target_masses[1])
    assert gcs.last_arrival[1] == 5


def test_target_only_map_rejects_obstacle_events():
    world=CooperativeWorldMap(num_cells=4)
    with pytest.raises(ValueError,match="target"):
        obs(0,1,1,True,layer="obstacle")
    assert world.target_masses[1] == pytest.approx((0,0,1))
    assert world.coverage[1] == 0


def test_m2_map_metrics_do_not_leak_truth_into_actor_map():
    gcs=CooperativeWorldMap(num_cells=4)
    gcs.ingest(obs(0,1,2,True),arrived_at=4)
    truth_target=np.array([False,False,True,False])
    measured=MultiLayerMapMetrics.evaluate(gcs,truth_target,at_tick=6)
    assert measured["target_brier"] >= 0
    assert "obstacle_brier" not in measured
    assert measured["coverage"] == pytest.approx(.25)
    assert not hasattr(gcs,"truth_target")


def test_batching_fits_packet_payload_and_preserves_event_ids():
    events=[obs(i%3,i+1,i%8, bool(i%2)) for i in range(22)]
    batches=packetize_events(events,payload_bytes=220)
    assert len(batches)>1
    assert all(b.size_bytes<=220 for b in batches)
    assert [event.event_id for batch in batches for event in batch.events] == [e.event_id for e in events]


def test_map_evidence_weight_has_no_information_when_p1_equals_p0():
    assert map_evidence_weight(.2,.2,True) == pytest.approx(0)
    assert map_evidence_weight(.9,.1,True) == pytest.approx(1-.1/.9)
    assert map_evidence_weight(.9,.1,False) == pytest.approx(1-.1/.9)


def test_revisioned_ground_station_correction_never_uses_true_target_id():
    from uav_network.communication.m2_protocol import DecisionNotice
    gcs=CooperativeWorldMap(num_cells=4)
    verified=DecisionNotice("u0:c1:v1",source=0,cell=1,revision=1,status="verified",tick=10)
    retracted=DecisionNotice("u0:c1:v2",source=0,cell=1,revision=2,status="retracted",tick=12)
    assert gcs.ingest_notice(verified,arrived_at=15)
    assert gcs.claims[(0,1)].status == "verified"
    assert gcs.ingest_notice(retracted,arrived_at=18)
    assert gcs.claims[(0,1)].status == "retracted"
    assert not gcs.ingest_notice(verified,arrived_at=19)
    assert gcs.claims[(0,1)].status == "retracted"
    assert not hasattr(verified,"true_target_id")
    assert len(gcs.report_history) == 2


def test_local_candidate_and_verified_emit_revision_notices():
    from uav_network.communication.m2_protocol import DecisionNotice
    node=CooperativeWorldMap(num_cells=4,ds_threshold=.95,alpha=.05,beta=.05)
    notices=[]
    for tick in range(1,20):
        node.ingest(obs(0,tick,1,True,rho=.3),arrived_at=tick)
        notices.extend(node.make_decision_notices(source=0))
        if node.candidates.get(1) and node.candidates[1].status=="verified":break
    assert [n.status for n in notices][0] == "pending"
    assert [n.status for n in notices][-1] == "verified"
    assert all(isinstance(n,DecisionNotice) for n in notices)
    assert notices[0].revision < notices[-1].revision


def test_real_event_correction_retracts_previous_cooperative_verification():
    """A stream of negative evidence can reverse a prior verified claim.

    This is an empirical behavior check for Q4B, NOT a Wald error guarantee.
    """
    from uav_network.communication.m2_protocol import DecisionNotice
    uav=CooperativeWorldMap(4,ds_threshold=.95,alpha=.05,beta=.05)
    gcs=CooperativeWorldMap(4,ds_threshold=.95,alpha=.05,beta=.05)
    statuses=[]
    for tick in range(1,17):
        event=obs(0,tick,1,tick<=8,p1=.9,p0=.1,rho=.3)
        uav.ingest(event,arrived_at=tick)
        for notice in uav.make_decision_notices(source=0):
            statuses.append(notice.status)
            gcs.ingest_notice(notice,arrived_at=tick+1)
    assert "verified" in statuses
    assert statuses[-1] in {"rejected","retracted"}
    assert gcs.claims[(0,1)].status==statuses[-1]
    revisions=[entry[0].revision for entry in gcs.report_history]
    assert revisions == sorted(revisions)
    assert len(revisions)==len(set(revisions))
