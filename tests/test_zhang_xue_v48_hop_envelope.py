"""Device-resident end-to-end route envelope carried across relay queues."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_persistent_queue_warp import (
    ZhangXueWarpPacketQueue,
)
from uav_network.communication.gpu.zhang_xue_forward_envelope_warp import (
    ZhangXueWarpForwardingEnvelope,
)


def send_packet(q,source,next_hop,pid=101,kind=0,created_us=0,ttl=10_000_000):
    e=q.num_envs
    ids=np.full((e,6),-1,np.int64)
    dst=np.full((e,6),-1,np.int32)
    size=np.zeros((e,6),np.int32)
    kinds=np.zeros((e,6),np.int32)
    times=np.zeros((e,6),np.int64)
    ttls=np.ones((e,6),np.int64)
    ids[:,source]=pid
    dst[:,source]=next_hop
    size[:,source]=78 if kind==0 else 90
    kinds[:,source]=kind
    times[:,source]=created_us
    ttls[:,source]=ttl
    return ids,dst,size,kinds,times,ttls


def attach(e,source,pid=101,origin=0,final=5,hop=0,visited=1):
    ids=np.full((e,6),-1,np.int64)
    src=np.full((e,6),-1,np.int32)
    last=np.full((e,6),-1,np.int32)
    count=np.zeros((e,6),np.int32)
    seen=np.zeros((e,6),np.int32)
    ids[:,source]=pid
    src[:,source]=origin
    last[:,source]=final
    count[:,source]=hop
    seen[:,source]=visited
    return ids,src,last,count,seen


def test_envelope_is_immutable_during_blackout_and_late_slot_reuse():
    q=ZhangXueWarpPacketQueue(2,capacity=1)
    e=ZhangXueWarpForwardingEnvelope(q)
    packet=send_packet(q,0,1,pid=701,created_us=200)
    assert np.all(q.enqueue(*packet)[:,0]==1)
    assert np.all(e.attach(*attach(2,0,pid=701))[:,0]==1)
    off=np.ones((2,6),np.int32)
    off[:,0]=0
    assert np.all(q.dispatch(300,radio_enabled=off)["packet_id"]==-1)
    old=attach(2,0,pid=701,origin=0,final=3,visited=1)
    assert np.all(e.attach(*old)[:,0]==-2)
    off[:,0]=1
    sent=q.dispatch(1_000,radio_enabled=off)
    metadata=e.read_dispatched(sent["packet_id"])
    assert np.all(metadata["original_source"][:,0]==0)
    assert np.all(metadata["ultimate_destination"][:,0]==5)
    assert np.all(metadata["hop_count"][:,0]==0)
    assert np.all(metadata["visited_mask"][:,0]==1)

    done=np.where(sent["packet_id"]>=0,1,0).astype(np.int32)
    q.resolve(sent["packet_id"],done)
    packet2=send_packet(q,0,1,pid=702,created_us=1_000)
    assert np.all(q.enqueue(*packet2)[:,0]==1)
    sent2=q.dispatch(1_001)
    stale=e.read_dispatched(sent2["packet_id"])
    assert np.all(stale["ultimate_destination"][:,0]==-1)
    assert np.all(e.attach(*attach(2,0,pid=702))[:,0]==1)
    recovered=e.read_dispatched(sent2["packet_id"])
    assert np.all(recovered["ultimate_destination"][:,0]==5)


def test_5_hop_route_updates_visited_bits_and_protects_against_cycles():
    q=ZhangXueWarpPacketQueue(1,capacity=2)
    envelope=ZhangXueWarpForwardingEnvelope(q)
    p=send_packet(q,0,1,pid=101)
    q.enqueue(*p)
    envelope.attach(*attach(1,0,pid=101))
    sent=q.dispatch(0)
    received=np.zeros((1,6),np.int32)
    received[0,0]=1
    action=envelope.classify_receipt(sent["packet_id"],sent["dest"],received)
    assert action["relay"][0,0]==1
    assert action["final"][0,0]==0
    assert action["hop_count"][0,0]==1
    assert action["visited_mask"][0,0]==0b000011
    assert action["ultimate_destination"][0,0]==5
    # Physical DATA retry that reaches a node already on its route
    # is rejected, even if other routing components made a bad choice.
    bad_received=np.ones((1,6),np.int32)
    dest_already_visited=sent["dest"].copy()
    dest_already_visited[0,0]=0
    bad=envelope.classify_receipt(sent["packet_id"],dest_already_visited,bad_received)
    assert bad["loop_rejected"][0,0]==1

    # Simulate immutable per-hop forwarding fields; each relay needs
    # only source ID, final destination, visited bitmask and hop count.
    for current,next_node in ((1,2),(2,3),(3,4),(4,5)):
        old=action
        q.resolve(sent["packet_id"],received)
        sent_p=send_packet(q,current,next_node,pid=101)
        assert q.enqueue(*sent_p)[0,current]==1
        ids,orig,dest,hop,mask=attach(
            1,current,pid=101,origin=int(old["original_source"][0,0]),
            final=int(old["ultimate_destination"][0,0]),
            hop=int(old["hop_count"][0,0]),
            visited=int(old["visited_mask"][0,0]),
        )
        assert envelope.attach(ids,orig,dest,hop,mask)[0,current]==1
        sent=q.dispatch(current)
        received=np.zeros((1,6),np.int32)
        received[0,current]=1
        action_full=envelope.classify_receipt(sent["packet_id"],sent["dest"],received)
        action={k:v[:,current:current+1] for k,v in action_full.items()}
        if next_node==5:
            assert action["final"][0,0]==1
            assert action["relay"][0,0]==0
        else:
            assert action["final"][0,0]==0
            assert action["relay"][0,0]==1
        assert int(action["hop_count"][0,0])==current+1
    assert int(action["visited_mask"][0,0])==0b111111


def test_forward_ttl_and_reset_do_not_leak_old_envelope():
    q=ZhangXueWarpPacketQueue(1,capacity=1)
    env=ZhangXueWarpForwardingEnvelope(q)
    p=send_packet(q,0,1,ttl=10_000_000)
    q.enqueue(*p)
    env.attach(*attach(1,0))
    assert q.dispatch(10_000_000)["packet_id"][0,0]==-1
    q.reset()
    env.reset()
    next_packet=send_packet(q,0,1,pid=202)
    q.enqueue(*next_packet)
    dispatched=q.dispatch(0)
    out=env.read_dispatched(dispatched["packet_id"])
    assert out["ultimate_destination"][0,0]==-1


def test_4096_worlds_and_invalid_route_meta_do_not_cross_contaminate():
    e=4096
    q=ZhangXueWarpPacketQueue(e,capacity=2)
    envelope=ZhangXueWarpForwardingEnvelope(q)
    p=list(send_packet(q,0,1))
    p[0][:,0]=np.arange(100,e+100,dtype=np.int64)
    q.enqueue(*p)
    route=list(attach(e,0))
    route[0][:,0]=p[0][:,0]
    route[2][::2,0]=5
    route[2][1::2,0]=4
    envelope.attach(*route)
    sent=q.dispatch(0)
    radio=np.zeros((e,6),np.int32)
    radio[::2,0]=1
    got=envelope.classify_receipt(sent["packet_id"],sent["dest"],radio)
    assert np.all(got["relay"][::2,0]==1)
    assert np.all(got["relay"][1::2,0]==0)
    assert np.all(got["ultimate_destination"][::2,0]==5)
    # Envelope remains private if its physical packet was NOT received.
    assert np.all(got["ultimate_destination"][1::2,0]==-1)
    assert np.all(got["visited_mask"][1::2,0]==0)
    bad=list(attach(e,0))
    bad[2][0,0]=6
    with pytest.raises(ValueError,match="destination"):
        envelope.attach(*bad)
    bad=list(attach(e,0))
    bad[4][0,0]=0
    with pytest.raises(ValueError,match="visited"):
        envelope.attach(*bad)


def test_envelope_rejects_forged_hop_and_visited_mask_pair():
    q=ZhangXueWarpPacketQueue(1,capacity=2)
    env=ZhangXueWarpForwardingEnvelope(q)
    q.enqueue(*send_packet(q,0,1,pid=101))
    forged=list(attach(1,0,pid=101,visited=0b000111,hop=0))
    with pytest.raises(ValueError,match="visited"):
        env.attach(*forged)
    actual=list(attach(1,0,pid=101,visited=1,hop=0))
    assert env.attach(*actual)[0,0]==1
