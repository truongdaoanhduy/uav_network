"""Test-first source-provenance receiver belief on actual Warp memory."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_origin_evidence_warp import (
    ZhangXueWarpOriginBelief,
)
from uav_network.communication.zhang_xue_ref.ds_belief import (
    BeliefTable, ds_log_space_safe,
)


def test_receiver_cannot_learn_remote_data_before_packet_arrives():
    store=ZhangXueWarpOriginBelief(num_envs=2,device="cpu")
    own_p=np.zeros((2,6,400),np.int32)
    own_n=np.zeros_like(own_p)
    own_p[0,0,7]=2
    own_n[0,0,7]=3
    own_p[1,0,13]=4
    store.set_own_evidence(own_p,own_n)
    r=store.read()
    assert r["pos"][0,0,0,7]==2
    assert r["neg"][0,0,0,7]==3
    assert r["pos"][0,1,0,7]==0
    assert r["neg"][0,1,0,7]==0
    assert r["pos"][1,1,0,13]==0
    assert r["mass_target"][0,1,7]==0
    assert r["unknown"][0,1,7]==1


def test_received_source_snapshot_max_plus_sum_fuses_identical_to_belief_table():
    envs=2
    store=ZhangXueWarpOriginBelief(num_envs=envs,device="cpu")
    own_p=np.zeros((envs,6,400),np.int32)
    own_n=np.zeros_like(own_p)
    own_p[0,1,7]=2
    own_n[0,1,7]=1
    own_p[0,0,7]=3
    own_n[0,0,7]=5
    own_p[1,0,9]=4
    store.set_own_evidence(own_p,own_n)

    rx=np.full((envs,6),-1,np.int32)
    rx[0,0]=1
    rx[1,0]=1
    origin=np.full((envs,6,62),-1,np.int32)
    cid=np.full((envs,6,62),-1,np.int32)
    pos=np.zeros((envs,6,62),np.int32)
    neg=np.zeros_like(pos)
    length=np.zeros((envs,6),np.int32)
    rx[:,0]=1
    origin[:,0,0]=0
    cid[0,0,0]=7
    pos[0,0,0]=3
    neg[0,0,0]=5
    cid[1,0,0]=9
    pos[1,0,0]=4
    length[:,0]=1
    delivered=np.zeros((envs,6),np.int32)
    delivered[0,0]=1

    events=store.merge_delivered(rx,origin,cid,pos,neg,length,delivered)
    assert events==1
    r=store.read()
    assert r["pos"][0,1,0,7]==3
    assert r["neg"][0,1,0,7]==5
    assert r["pos"][1,1,0,9]==0
    # Native BeliefTable reference per-receiver and per-origin Max-plus-Sum.
    reference=BeliefTable(6,400)
    for u in range(6):
        reference.pos[u,u]=own_p[0,u]
        reference.neg[u,u]=own_n[0,u]
    reference.merge_packet(1,((0,7,3,5),))
    mf,me,mu=reference.get_belief(1,[7])
    np.testing.assert_allclose(r["mass_target"][0,1,7],mf[0],atol=5e-6)
    np.testing.assert_allclose(r["mass_empty"][0,1,7],me[0],atol=5e-6)
    np.testing.assert_allclose(r["unknown"][0,1,7],mu[0],atol=5e-6)
    assert np.isclose(r["mass_target"][0,1,7]+r["mass_empty"][0,1,7]+r["unknown"][0,1,7],1)


def test_retransmission_and_relay_copy_of_same_origin_are_idempotent():
    store=ZhangXueWarpOriginBelief(1)
    a=np.zeros((1,6,400),np.int32)
    store.set_own_evidence(a,a)
    rx=np.full((1,6),-1,np.int32)
    rx[0,0]=2
    rx[0,1]=2
    origin=np.full((1,6,62),-1,np.int32)
    cid=np.full((1,6,62),-1,np.int32)
    p=np.zeros((1,6,62),np.int32)
    n=np.zeros_like(p)
    cnt=np.zeros((1,6),np.int32)
    good=np.zeros((1,6),np.int32)
    for source in (0,1):
        origin[0,source,0]=0
        cid[0,source,0]=17
        p[0,source,0]=8
        n[0,source,0]=5
        cnt[0,source]=1
        good[0,source]=1
    store.merge_delivered(rx,origin,cid,p,n,cnt,good)
    once=store.read()
    store.merge_delivered(rx,origin,cid,p,n,cnt,good)
    twice=store.read()
    assert once["pos"][0,2,0,17]==8
    assert twice["pos"][0,2,0,17]==8
    assert twice["neg"][0,2,0,17]==5
    assert np.array_equal(once["pos"],twice["pos"])
    assert np.array_equal(once["unknown"],twice["unknown"])


def test_invalid_payload_rejected_before_gpu_and_reset_clears_all_teams():
    store=ZhangXueWarpOriginBelief(1)
    own=np.zeros((1,6,400),np.int32)
    own[0,0,3]=2
    store.set_own_evidence(own,np.zeros_like(own))
    rx=np.full((1,6),-1,np.int32)
    rx[0,0]=1
    origins=np.zeros((1,6,62),np.int32)
    cells=np.zeros_like(origins)
    positives=np.zeros_like(origins)
    negatives=np.zeros_like(origins)
    counts=np.zeros((1,6),np.int32)
    success=np.zeros((1,6),np.int32)
    counts[0,0]=1
    success[0,0]=1
    cells[0,0,0]=400
    with pytest.raises(ValueError,match="cell"):
        store.merge_delivered(rx,origins,cells,positives,negatives,counts,success)
    cells[0,0,0]=3
    positives[0,0,0]=65536
    with pytest.raises(ValueError,match="count"):
        store.merge_delivered(rx,origins,cells,positives,negatives,counts,success)
    store.reset()
    r=store.read()
    assert not r["pos"].any()
    assert not r["neg"].any()
    assert np.all(r["unknown"]==1)


def test_4096_receiver_world_isolation_and_high_count_saturation():
    e=4096
    store=ZhangXueWarpOriginBelief(e,device="cpu")
    own_p=np.zeros((e,6,400),np.int32)
    own_n=np.zeros_like(own_p)
    own_p[::2,0,33]=65535
    store.set_own_evidence(own_p,own_n)
    rx=np.full((e,6),-1,np.int32)
    rx[:,0]=1
    origin=np.full((e,6,62),-1,np.int32)
    cid=np.full((e,6,62),-1,np.int32)
    p=np.zeros((e,6,62),np.int32)
    n=np.zeros_like(p)
    cnt=np.zeros((e,6),np.int32)
    good=np.zeros((e,6),np.int32)
    cnt[:,0]=1
    good[::2,0]=1
    origin[:,0,0]=0
    cid[:,0,0]=33
    p[:,0,0]=65535
    store.merge_delivered(rx,origin,cid,p,n,cnt,good)
    out=store.read()
    assert np.all(out["pos"][::2,1,0,33]==65535)
    assert np.all(out["pos"][1::2,1,0,33]==0)
    assert np.isfinite(out["mass_target"]).all()
    assert np.isfinite(out["unknown"]).all()
