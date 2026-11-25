"""App E2E ACK (not PHY or MAC ACK) advances origin-peer evidence watermark."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_e2e_watermarks_warp import (
    ZhangXueWarpAckWatermarks,
)


def _peer(e=1,target=1):
    peers=np.full((e,6),-1,np.int32)
    peers[:,0]=target
    return peers


def _counts(e=1):
    p=np.zeros((e,6,400),np.int32)
    n=np.zeros_like(p)
    p[:,0,7]=4
    n[:,0,7]=3
    return p,n


def _ids(e=1,seq=101):
    ids=np.full((e,6),-1,np.int64)
    ids[:,0]=seq
    return ids


def _success(e=1,src=0,yes=True):
    success=np.zeros((e,6),np.int32)
    success[:,src]=int(yes)
    return success


def test_only_returned_app_ack_advances_watermark_and_dirty_is_peer_local():
    w=ZhangXueWarpAckWatermarks(1)
    p,n=_counts()
    own_peer=_peer()
    d=w.dirty(p,n,own_peer)
    assert d["count"][0,0]==1
    assert d["cell"][0,0,0]==7
    assert d["positive"][0,0,0]==4
    assert d["negative"][0,0,0]==3
    ids=_ids()
    assert w.register(ids,own_peer,d)[0,0]==1
    assert w.dirty(p,n,own_peer)["count"][0,0]==0  # hold pending
    # Link MAC-ACK does not advance application-level watermark.
    assert w.read()["acked_positive"][0,0,1,7]==0
    # App E2E-ACK absent: still pending; frame drop does not pretend synced.
    assert w.acknowledge(ids,own_peer,_success(yes=False))[0,0]==0
    assert w.read()["acked_positive"][0,0,1,7]==0
    assert w.acknowledge(ids,own_peer,_success())[0,0]==1
    assert w.read()["acked_positive"][0,0,1,7]==4
    assert w.read()["acked_negative"][0,0,1,7]==3
    assert w.dirty(p,n,own_peer)["count"][0,0]==0
    other_peer=_peer(target=2)
    assert w.dirty(p,n,other_peer)["count"][0,0]==1


def test_late_ack_for_old_sequence_cannot_release_new_or_reward_twice():
    w=ZhangXueWarpAckWatermarks(1)
    p,n=_counts()
    peer=_peer()
    d=w.dirty(p,n,peer)
    assert w.register(_ids(seq=101),peer,d)[0,0]==1
    assert w.acknowledge(_ids(seq=999),peer,_success())[0,0]==-1
    assert w.read()["acked_positive"][0,0,1,7]==0
    assert w.acknowledge(_ids(seq=101),peer,_success())[0,0]==1
    assert w.acknowledge(_ids(seq=101),peer,_success())[0,0]==-1
    assert w.read()["successful_e2e_acks"][0,0]==1
    p[0,0,7]=6
    d2=w.dirty(p,n,peer)
    assert d2["positive"][0,0,0]==6
    assert w.register(_ids(seq=102),peer,d2)[0,0]==1
    assert w.acknowledge(_ids(seq=101),peer,_success())[0,0]==-1
    assert w.dirty(p,n,peer)["count"][0,0]==0
    assert w.acknowledge(_ids(seq=102),peer,_success())[0,0]==1
    assert w.read()["acked_positive"][0,0,1,7]==6


def test_new_sensor_counts_after_enqueue_not_lost_when_old_ack_returns():
    w=ZhangXueWarpAckWatermarks(1)
    p,n=_counts()
    peer=_peer()
    d=w.dirty(p,n,peer)
    assert w.register(_ids(seq=10),peer,d)[0,0]==1
    # New evidence is locally discovered while packet seq=10 is in flight.
    p[0,0,7]=9
    n[0,0,13]=2
    assert w.acknowledge(_ids(seq=10),peer,_success())[0,0]==1
    d2=w.dirty(p,n,peer)
    assert d2["count"][0,0]==2
    assert d2["cell"][0,0,0]==7
    assert d2["positive"][0,0,0]==9
    assert d2["cell"][0,0,1]==13
    assert d2["negative"][0,0,1]==2
    assert w.read()["acked_positive"][0,0,1,7]==4


def test_dirty_selection_chunks_62_cells_and_retries_after_failed_app_ack():
    w=ZhangXueWarpAckWatermarks(1)
    p,n=np.zeros((1,6,400),np.int32),np.zeros((1,6,400),np.int32)
    p[0,0,:130]=1
    peer=_peer()
    d=w.dirty(p,n,peer)
    assert d["count"][0,0]==62
    assert np.array_equal(d["cell"][0,0,:62],np.arange(62))
    assert w.register(_ids(seq=1),peer,d)[0,0]==1
    assert w.acknowledge(_ids(seq=1),peer,_success())[0,0]==1
    d2=w.dirty(p,n,peer)
    assert d2["count"][0,0]==62
    assert np.array_equal(d2["cell"][0,0,:62],np.arange(62,124))
    assert w.register(_ids(seq=2),peer,d2)[0,0]==1
    assert w.abandon(_ids(seq=2),peer)[0,0]==1
    again=w.dirty(p,n,peer)
    assert np.array_equal(again["cell"][0,0,:62],np.arange(62,124))
    assert w.read()["acked_positive"][0,0,1,124]==0


def test_4096_worlds_and_simultaneous_all_senders_are_independent():
    e=4096
    w=ZhangXueWarpAckWatermarks(e)
    p=np.zeros((e,6,400),np.int32)
    n=np.zeros_like(p)
    p[:,:,7]=np.arange(1,e+1,dtype=np.int32)[:,None]
    peers=np.broadcast_to(((np.arange(6)+1)%6)[None,:],(e,6)).copy()
    ids=(np.arange(e,dtype=np.int64)[:,None]*100+np.arange(6)[None,:]+1000)
    d=w.dirty(p,n,peers)
    assert np.all(d["count"]==1)
    assert np.all(w.register(ids,peers,d)==1)
    good=np.zeros((e,6),np.int32)
    good[::2,:]=1
    status=w.acknowledge(ids,peers,good)
    assert np.all(status[::2,:]==1)
    assert np.all(status[1::2,:]==0)
    snap=w.read()
    assert np.all(snap["acked_positive"][::2,0,1,7]==p[::2,0,7])
    assert np.all(snap["acked_positive"][1::2,0,1,7]==0)
    assert np.all(w.dirty(p,n,peers)["count"][::2,:]==0)
    assert np.all(w.dirty(p,n,peers)["count"][1::2,:]==0)
    assert np.all(w.read()["successful_e2e_acks"][::2,:]==1)


def test_reset_does_not_leak_watermark_between_episodes():
    w=ZhangXueWarpAckWatermarks(1)
    p,n=_counts()
    peer=_peer()
    d=w.dirty(p,n,peer)
    w.register(_ids(seq=1),peer,d)
    w.acknowledge(_ids(seq=1),peer,_success())
    w.reset()
    assert w.dirty(p,n,peer)["count"][0,0]==1
    assert not w.read()["acked_positive"].any()
    assert not w.read()["successful_e2e_acks"].any()
