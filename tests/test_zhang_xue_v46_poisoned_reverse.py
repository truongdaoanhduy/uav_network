"""Count-to-infinity prevention: split horizon with poisoned reverse.

The direct route from UAV1 to UAV2 expires while UAV0 still has a
time-valid route from UAV1's last advertisement. UAV0 must not tell
UAV1 that it has a viable route to UAV2 through UAV1 itself.
"""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import ZhangXueWarpDistanceVector


def _packet(envs,source,receiver,mask=None):
    rx=np.full((envs,6),-1,np.int32)
    received=np.zeros((envs,6),np.int32)
    rx[:,source]=receiver
    received[:,source]=1 if mask is None else mask
    return rx,received


def _hello(r,source,receiver,at):
    r.receive_hello(*_packet(r.num_envs,source,receiver),now_us=at)


def _adv(r,source,receiver,at):
    r.receive_advertisement(*_packet(r.num_envs,source,receiver),now_us=at)


def test_poisoned_reverse_prevents_relay_1_route_to_2_via_source_0():
    r=ZhangXueWarpDistanceVector(1,hello_ttl_us=100)
    _hello(r,0,1,0)
    _hello(r,1,0,1)
    _hello(r,1,2,2)
    _hello(r,2,1,3)
    _hello(r,0,1,80)
    _hello(r,1,0,81)
    _adv(r,1,0,90)
    assert r.read_routes()["next_hop"][0,0,2]==1
    assert r.read_routes()["next_hop"][0,1,2]==2

    r.tick(105)
    # 1-2 direct link expired but 0-1 and the 1->0 DV packet remain valid.
    assert r.read_routes()["next_hop"][0,1,2]==-1
    assert r.read_routes()["next_hop"][0,0,2]==1
    _adv(r,0,1,106)
    assert r.read_routes()["next_hop"][0,1,2]==-1, (
        "Loop detected: UAV1→UAV0→UAV1 instead of unreachable destination2"
    )
    r.tick(108)
    assert r.read_routes()["next_hop"][0,1,2]==-1


def test_poisoned_reverse_preserves_valid_direct_and_third_party_routes():
    r=ZhangXueWarpDistanceVector(1,hello_ttl_us=100)
    for src,dst,at in ((0,1,0),(1,0,1),(1,2,2),(2,1,3),
                       (1,3,4),(3,1,5)):
        _hello(r,src,dst,at)
    _adv(r,1,0,6)
    assert r.read_routes()["next_hop"][0,0,2]==1
    assert r.read_routes()["next_hop"][0,0,3]==1
    _adv(r,0,1,7)
    # Poisoned reverse only masks destinations routed *via receiver*.
    assert r.read_routes()["next_hop"][0,1,2]==2
    assert r.read_routes()["next_hop"][0,1,3]==3
    assert r.read_routes()["next_hop"][0,1,0]==0


def test_poisoned_reverse_is_independent_over_4096_worlds():
    e=4096
    r=ZhangXueWarpDistanceVector(e,hello_ttl_us=100)
    _hello(r,0,1,0)
    _hello(r,1,0,1)
    _hello(r,1,2,2)
    _hello(r,2,1,3)
    _hello(r,0,1,80)
    _hello(r,1,0,81)
    _adv(r,1,0,90)
    r.tick(105)
    assert np.all(r.read_routes()["next_hop"][:,0,2]==1)
    _adv(r,0,1,106)
    assert np.all(r.read_routes()["next_hop"][:,1,2]==-1)
