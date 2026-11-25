"""Red-green Warp queue contract: 6 nodes, persistent bounded per-UAV packets."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import (
    ZhangXueWarpPacketQueue,
)


def _packet(e=2,packet_id=101,src=0,dest=1,kind=0,bytes_=566,created=0,ttl=10_000_000):
    ident=np.full((e,6),-1,np.int64)
    ids_dest=np.full((e,6),-1,np.int32)
    size=np.zeros((e,6),np.int32)
    kind_arr=np.zeros((e,6),np.int32)
    created_arr=np.zeros((e,6),np.int64)
    ttl_arr=np.full((e,6),1,np.int64)
    ident[:,src]=packet_id
    ids_dest[:,src]=dest
    size[:,src]=bytes_
    kind_arr[:,src]=kind
    created_arr[:,src]=created
    ttl_arr[:,src]=ttl
    return ident,ids_dest,size,kind_arr,created_arr,ttl_arr


def _enqueue(q,p):
    return q.enqueue(*p)


def test_queue_blackout_persists_across_multiple_ticks_until_recovery():
    q=ZhangXueWarpPacketQueue(num_envs=2,capacity=3,device="cpu")
    got=_enqueue(q,_packet())
    np.testing.assert_array_equal(got[:,0],[1,1])
    blackout=np.ones((2,6),np.int32)
    blackout[:,0]=0
    for step in (0,1_000_000,2_000_000,7_000_000):
        result=q.dispatch(step,radio_enabled=blackout)
        assert np.all(result["packet_id"]==-1)
        np.testing.assert_array_equal(q.snapshot()["queue_length"][:,0],[1,1])
    blackout[:,0]=1
    result=q.dispatch(8_000_000,radio_enabled=blackout)
    np.testing.assert_array_equal(result["packet_id"][:,0],[101,101])
    np.testing.assert_array_equal(result["dest"][:,0],[1,1])
    np.testing.assert_array_equal(result["frame_bytes"][:,0],[566,566])
    assert np.all(q.snapshot()["inflight"][:,0]==1)
    # ACK reception finishes link-service, freeing the only packet.
    result=q.resolve(result["packet_id"],np.where(
        result["packet_id"]>=0,1,0
    ).astype(np.int32))
    np.testing.assert_array_equal(result[:,0],[1,1])
    assert not q.snapshot()["queue_length"].any()
    assert int(q.snapshot()["mac_acked"].sum())==2


def test_ttl_expiry_is_exact_at_deadline_even_after_blackout():
    q=ZhangXueWarpPacketQueue(num_envs=1,capacity=3)
    _enqueue(q,_packet(e=1,ttl=10_000_000))
    off=np.ones((1,6),np.int32)
    off[0,0]=0
    assert q.dispatch(9_999_999,radio_enabled=off)["packet_id"][0,0]==-1
    assert q.snapshot()["queue_length"][0,0]==1
    on=np.ones((1,6),np.int32)
    assert q.dispatch(10_000_000,radio_enabled=on)["packet_id"][0,0]==-1
    assert q.snapshot()["queue_length"][0,0]==0
    assert q.snapshot()["expired"][0,0]==1
    q.dispatch(10_000_001,radio_enabled=on)
    assert q.snapshot()["expired"][0,0]==1


def test_duplicate_suppressed_full_queue_rejected_fifo_survives_slot_reuse():
    q=ZhangXueWarpPacketQueue(num_envs=1,capacity=2)
    a=_packet(e=1,packet_id=101)
    b=_packet(e=1,packet_id=102)
    c=_packet(e=1,packet_id=103)
    assert _enqueue(q,a)[0,0]==1
    assert _enqueue(q,a)[0,0]==-1
    assert _enqueue(q,b)[0,0]==1
    assert _enqueue(q,c)[0,0]==-2
    assert q.snapshot()["duplicate"][0,0]==1
    assert q.snapshot()["dropped_full"][0,0]==1
    first=q.dispatch(0)
    assert first["packet_id"][0,0]==101
    assert q.resolve(first["packet_id"],np.where(
        first["packet_id"]>=0,1,0).astype(np.int32))[0,0]==1
    assert _enqueue(q,c)[0,0]==1
    second=q.dispatch(1)
    assert second["packet_id"][0,0]==102
    q.resolve(second["packet_id"],np.where(
        second["packet_id"]>=0,1,0).astype(np.int32))
    third=q.dispatch(2)
    assert third["packet_id"][0,0]==103


def test_stale_mac_completion_cannot_consume_new_packet():
    q=ZhangXueWarpPacketQueue(num_envs=1,capacity=1)
    _enqueue(q,_packet(e=1,packet_id=100))
    first=q.dispatch(0)
    q.resolve(first["packet_id"],np.where(
        first["packet_id"]>=0,2,0).astype(np.int32))
    _enqueue(q,_packet(e=1,packet_id=101))
    second=q.dispatch(10)
    status=q.resolve(first["packet_id"],np.where(
        first["packet_id"]>=0,1,0).astype(np.int32))
    assert status[0,0]==-1
    assert q.snapshot()["inflight"][0,0]==1
    assert q.snapshot()["invalid_completion"][0,0]==1
    assert second["packet_id"][0,0]==101


def test_batch_4096_world_isolation_and_no_send_returns_no_ghost_packet():
    e=4096
    q=ZhangXueWarpPacketQueue(num_envs=e,capacity=2)
    ids,dest,sz,kind,created,ttl=_packet(e=e,packet_id=1)
    ids[:e//2,0]=np.arange(1,e//2+1,dtype=np.int64)
    _enqueue(q,(ids,dest,sz,kind,created,ttl))
    radio=np.ones((e,6),np.int32)
    radio[::2,0]=0
    first=q.dispatch(0,radio_enabled=radio)
    assert int(np.count_nonzero(first["packet_id"]>=0))==e//2
    assert np.all(first["packet_id"][:,1:]==-1)
    assert np.all(q.snapshot()["queue_length"][:,0]==1)
    radio[:,0]=1
    second=q.dispatch(1,radio_enabled=radio)
    assert np.all(second["packet_id"][::2,0]>=0)
    assert np.all(second["packet_id"][1::2,0]==-1)


def test_queue_input_rejects_self_send_negative_ttl_and_noninteger_clock():
    q=ZhangXueWarpPacketQueue(num_envs=1)
    packet=list(_packet(e=1))
    packet[1][0,0]=0
    with pytest.raises(ValueError,match="destination"):
        _enqueue(q,packet)
    packet=list(_packet(e=1))
    packet[5][0,0]=0
    with pytest.raises(ValueError,match="ttl"):
        _enqueue(q,packet)
    with pytest.raises(ValueError,match="integer"):
        q.dispatch(1.5)


def test_queue_packet_frame_types_enforce_real_application_wire_size():
    q=ZhangXueWarpPacketQueue(num_envs=1,capacity=2)
    good_ack=_packet(e=1,packet_id=600001,src=1,dest=0,kind=1,bytes_=90)
    assert _enqueue(q,good_ack)[0,1]==1
    bad_ack=_packet(e=1,packet_id=600002,src=1,dest=0,kind=1,bytes_=566)
    with pytest.raises(ValueError,match="90"):
        _enqueue(q,bad_ack)
    too_short=_packet(e=1,packet_id=2,kind=0,bytes_=20)
    with pytest.raises(ValueError,match="70"):
        _enqueue(q,too_short)


def test_queue_dispatch_includes_original_absolute_creation_deadline():
    q=ZhangXueWarpPacketQueue(num_envs=1,capacity=2)
    packet=_packet(
        e=1,packet_id=100,created=2_000_000_001,ttl=10_000_000,
    )
    assert _enqueue(q,packet)[0,0]==1
    d=q.dispatch(2_000_000_007)
    assert d["packet_id"][0,0]==100
    assert d["created_us"][0,0]==2_000_000_001
    assert d["ttl_us"][0,0]==10_000_000


def test_queue_reset_erases_previous_episode_and_old_packet_completions():
    q=ZhangXueWarpPacketQueue(num_envs=2,capacity=2)
    _enqueue(q,_packet(e=2,packet_id=87))
    a=q.dispatch(0)
    assert a["packet_id"][0,0]==87
    q.reset()
    snap=q.snapshot()
    assert np.all(snap["queue_length"]==0)
    assert np.all(snap["inflight"]==0)
    assert all(not np.any(v) for k,v in snap.items()
               if k not in ("queue_length","inflight"))
    # The old completion is not allowed to resurrect the cleared packet.
    outcomes=np.where(a["packet_id"]>=0,1,0).astype(np.int32)
    status=q.resolve(a["packet_id"],outcomes)
    assert np.all(status[:,0]==-1)
    assert not q.snapshot()["queue_length"].any()
    assert not q.snapshot()["mac_acked"].any()


def test_queue_future_creation_and_per_environment_int64_time_independence():
    q=ZhangXueWarpPacketQueue(num_envs=2,capacity=2)
    data=list(_packet(e=2,packet_id=1,created=3_000_000_000,ttl=100))
    _enqueue(q,data)
    a=q.dispatch(np.array([2_999_999_999,3_000_000_099],np.int64))
    assert a["packet_id"][0,0]==-1
    assert a["packet_id"][1,0]==1
    # World0 reaches exact TTL boundary before it is ever leased.
    b=q.dispatch(np.array([3_000_000_100,3_000_000_100],np.int64))
    assert b["packet_id"][0,0]==-1
    assert b["packet_id"][1,0]==-1
    np.testing.assert_array_equal(q.snapshot()["expired"][:,0],[1,1])
