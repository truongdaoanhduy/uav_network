"""Per-packet origin evidence snapshot must persist while queued, on Warp."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import ZhangXueWarpPacketQueue
from uav_network.communication.gpu.zhang_xue_origin_evidence_warp import ZhangXueWarpOriginBelief
from uav_network.communication.gpu.zhang_xue_payload_store_warp import ZhangXueWarpPayloadStore


def _frame(e=2,ids=101):
    pid=np.full((e,6),-1,np.int64)
    dest=np.full((e,6),-1,np.int32)
    size=np.zeros((e,6),np.int32)
    typ=np.zeros((e,6),np.int32)
    created=np.zeros((e,6),np.int64)
    ttl=np.ones((e,6),np.int64)
    pid[:,0]=ids
    dest[:,0]=1
    size[:,0]=78
    ttl[:,0]=10_000_000
    return pid,dest,size,typ,created,ttl


def _entries(e=2):
    origin=np.full((e,6,62),-1,np.int32)
    cells=np.full((e,6,62),-1,np.int32)
    pos=np.zeros((e,6,62),np.int32)
    neg=np.zeros_like(pos)
    count=np.zeros((e,6),np.int32)
    count[:,0]=1
    origin[:,0,0]=0
    cells[:,0,0]=7
    pos[:,0,0]=4
    neg[:,0,0]=1
    return origin,cells,pos,neg,count


def test_packet_payload_captured_once_and_delivered_only_after_blackout():
    q=ZhangXueWarpPacketQueue(2,capacity=2)
    store=ZhangXueWarpPayloadStore(q)
    ids,dst,sz,typ,cr,ttl=_frame()
    assert np.all(q.enqueue(ids,dst,sz,typ,cr,ttl)[:,0]==1)
    assert np.all(store.attach(ids,*_entries())[:,0]==1)
    off=np.ones((2,6),np.int32)
    off[:,0]=0
    assert np.all(q.dispatch(0,radio_enabled=off)["packet_id"]==-1)
    # Current source sensor changes after packet creation must not
    # alter immutable on-wire payload snapshots.
    updated=list(_entries())
    updated[2][:,0,0]=99
    assert np.all(store.attach(ids,*updated)[:,0]==-2)
    off[:,0]=1
    chosen=q.dispatch(3_000_000,radio_enabled=off)
    captured=store.read_dispatched(chosen["packet_id"])
    assert np.all(captured["positive"][:,0,0]==4)
    assert np.all(captured["negative"][:,0,0]==1)
    belief=ZhangXueWarpOriginBelief(2)
    own=np.zeros((2,6,400),np.int32)
    own[:,0,7]=99
    belief.set_own_evidence(own,np.zeros_like(own))
    yes=np.where(chosen["packet_id"]>=0,1,0).astype(np.int32)
    assert store.merge_receipts(chosen["packet_id"],chosen["dest"],yes,belief)==2
    out=belief.read()
    assert np.all(out["pos"][:,1,0,7]==4)
    assert np.all(out["pos"][:,0,0,7]==99)
    assert np.all(out["pos"][:,2,0,7]==0)
    # Replay can physically arrive twice but never sum evidence twice.
    assert store.merge_receipts(chosen["packet_id"],chosen["dest"],yes,belief)==2
    assert np.all(belief.read()["pos"][:,1,0,7]==4)


def test_payload_lifecycle_cannot_leak_when_queue_slot_is_reused():
    q=ZhangXueWarpPacketQueue(1,capacity=1)
    payload=ZhangXueWarpPayloadStore(q)
    p=_frame(e=1,ids=100)
    assert q.enqueue(*p)[0,0]==1
    assert payload.attach(p[0],*_entries(e=1))[0,0]==1
    d=q.dispatch(0)
    done=np.where(d["packet_id"]>=0,1,0).astype(np.int32)
    q.resolve(d["packet_id"],done)
    new=_frame(e=1,ids=200)
    assert q.enqueue(*new)[0,0]==1
    d2=q.dispatch(1)
    old=payload.read_dispatched(d2["packet_id"])
    assert old["entry_count"][0,0]==0
    assert np.all(old["cell"][0,0]==-1)
    with pytest.raises(ValueError,match="payload"):
        payload.merge_receipts(d2["packet_id"],d2["dest"],done,
                               ZhangXueWarpOriginBelief(1))
    assert payload.attach(new[0],*_entries(e=1))[0,0]==1
    new_payload=payload.read_dispatched(d2["packet_id"])
    assert new_payload["entry_count"][0,0]==1
    assert new_payload["cell"][0,0,0]==7


def test_queued_snapshot_cannot_outlive_packet_ttl_and_reset():
    q=ZhangXueWarpPacketQueue(1,capacity=1)
    payload=ZhangXueWarpPayloadStore(q)
    p=_frame(e=1,ids=100)
    q.enqueue(*p)
    payload.attach(p[0],*_entries(e=1))
    q.dispatch(10_000_000)
    q.reset()
    payload.reset()
    d=q.dispatch(11_000_000)
    assert d["packet_id"][0,0]==-1
    assert payload.read_dispatched(d["packet_id"])["entry_count"][0,0]==0


def test_4096_payload_slots_isolation_under_parallel_delivery():
    e=4096
    q=ZhangXueWarpPacketQueue(e,capacity=2)
    store=ZhangXueWarpPayloadStore(q)
    frame=list(_frame(e=e))
    frame[0][:,0]=np.arange(1,e+1,dtype=np.int64)
    q.enqueue(*frame)
    entries=list(_entries(e=e))
    entries[2][:,0,0]=np.arange(1,e+1,dtype=np.int32)%30000+1
    store.attach(frame[0],*entries)
    radio=np.ones((e,6),np.int32)
    radio[1::2,0]=0
    chosen=q.dispatch(0,radio_enabled=radio)
    delivered=np.where(chosen["packet_id"]>=0,1,0).astype(np.int32)
    belief=ZhangXueWarpOriginBelief(e)
    result=store.merge_receipts(chosen["packet_id"],chosen["dest"],
                                delivered,belief)
    assert result==e//2
    out=belief.read()["pos"][:,1,0,7]
    assert np.array_equal(out[::2],entries[2][::2,0,0])
    assert np.all(out[1::2]==0)
