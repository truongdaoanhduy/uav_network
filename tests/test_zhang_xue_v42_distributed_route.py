"""No-oracle routing: a UAV sees only HELLO and DV packets it receives."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import ZhangXueWarpDistanceVector


def _packet(e=1,senders=((0,1),),success=True):
    dst=np.full((e,6),-1,np.int32)
    ok=np.zeros((e,6),np.int32)
    for src,rx in senders:
        dst[:,src]=rx
        ok[:,src]=int(success)
    return dst,ok


def test_absent_hello_leaks_zero_global_route_knowledge():
    r=ZhangXueWarpDistanceVector(num_envs=1)
    out=r.read_routes()
    assert np.all(np.diag(out["next_hop"][0])==np.arange(6))
    assert out["next_hop"][0,0,1]==-1
    assert out["next_hop"][0,0,2]==-1
    assert not out["bidirectional_links"].any()


def test_bidirectional_hello_required_and_advertisement_is_explicit_packet():
    r=ZhangXueWarpDistanceVector(1,hello_ttl_us=5_000_000)
    r.receive_hello(*_packet(senders=((0,1),(1,2))),now_us=0)
    assert r.read_routes()["next_hop"][0,0,1]==-1  # no reverse HELLO yet
    r.receive_hello(*_packet(senders=((1,0),(2,1))),now_us=1)
    d=r.read_routes()
    assert d["next_hop"][0,0,1]==1
    assert d["next_hop"][0,1,2]==2
    assert d["next_hop"][0,0,2]==-1  # no oracle peek at 1's table
    r.receive_advertisement(*_packet(senders=((1,0),)),now_us=2)
    assert r.read_routes()["next_hop"][0,0,2]==1
    assert r.read_routes()["hop_count"][0,0,2]==2


def test_failed_hello_or_advert_packet_cannot_update_routing():
    r=ZhangXueWarpDistanceVector(1)
    r.receive_hello(*_packet(senders=((0,1),(1,0)),success=False),now_us=10)
    assert r.read_routes()["next_hop"][0,0,1]==-1
    r.receive_hello(*_packet(senders=((0,1),(1,0)),success=True),now_us=20)
    assert r.read_routes()["next_hop"][0,0,1]==1
    r.receive_advertisement(*_packet(senders=((1,0),),success=False),now_us=30)
    assert r.read_routes()["next_hop"][0,0,2]==-1


def test_link_expires_and_routing_no_longer_uses_aged_hello():
    r=ZhangXueWarpDistanceVector(1,hello_ttl_us=100)
    r.receive_hello(*_packet(senders=((0,1),(1,0))),now_us=10)
    assert r.read_routes()["next_hop"][0,0,1]==1
    r.tick(111)
    assert r.read_routes()["next_hop"][0,0,1]==-1
    r.receive_hello(*_packet(senders=((0,1),(1,0))),now_us=112)
    assert r.read_routes()["next_hop"][0,0,1]==1


def test_4096_hellos_are_independent_and_do_not_cross_worlds():
    e=4096
    r=ZhangXueWarpDistanceVector(e)
    rx,ok=_packet(e,senders=((0,1),(1,0)))
    ok[1::2,:]=0
    r.receive_hello(rx,ok,now_us=0)
    got=r.read_routes()
    assert np.all(got["next_hop"][::2,0,1]==1)
    assert np.all(got["next_hop"][1::2,0,1]==-1)
    assert np.all(got["next_hop"][:,0,2]==-1)
