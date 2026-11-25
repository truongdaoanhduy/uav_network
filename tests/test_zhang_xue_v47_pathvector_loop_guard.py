"""Three-node count-to-infinity regression beyond poisoned reverse.

Triangle 0-1-2 all remain connected after the 0-3 original destination
link expires. The stale path 0->3, 1->0->3, 2->1->0->3 must not
generate 0->2->1->0 because 2's own advertised path already includes0.
"""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_distributed_routing_warp import (
    ZhangXueWarpDistanceVector,
)


def packet(e,source,dest):
    rx=np.full((e,6),-1,np.int32)
    success=np.zeros((e,6),np.int32)
    rx[:,source]=dest
    success[:,source]=1
    return rx,success


def hello(r,a,b,at):
    r.receive_hello(*packet(r.num_envs,a,b),now_us=at)


def advert(r,a,b,at):
    r.receive_advertisement(*packet(r.num_envs,a,b),now_us=at)


def setup_triangle(e):
    r=ZhangXueWarpDistanceVector(e,hello_ttl_us=100)
    for a,b,at in ((0,1,0),(1,0,1),(1,2,2),(2,1,3),
                   (2,0,4),(0,2,5),(0,3,6),(3,0,7)):
        hello(r,a,b,at)
    advert(r,0,1,8) # UAV1 learned 1->0->3
    advert(r,1,2,9) # UAV2 learned 2->1->0->3
    assert np.all(r.read_routes()["next_hop"][:,1,3]==0)
    assert np.all(r.read_routes()["next_hop"][:,2,3]==1)
    for a,b,at in ((0,1,95),(1,0,96),(1,2,97),
                   (2,1,98),(2,0,99),(0,2,100)):
        hello(r,a,b,at)
    # Old 0->1 route advertisement remains alive but direct 0->3
    # expires since its reverse last heard at time7.
    advert(r,0,1,101)
    advert(r,1,2,102)
    r.tick(108)
    assert np.all(r.read_routes()["next_hop"][:,0,3]==-1)
    assert np.all(r.read_routes()["next_hop"][:,1,3]==0)
    assert np.all(r.read_routes()["next_hop"][:,2,3]==1)
    return r


def test_three_node_triangle_with_failed_destination_link_has_no_loop():
    r=setup_triangle(1)
    advert(r,2,0,109)
    routes=r.read_routes()["next_hop"]
    assert routes[0,0,3]==-1, "3-cycle: 0->2->1->0"
    assert routes[0,1,3]==0
    assert routes[0,2,3]==1


def test_path_vector_retains_valid_2_hop_route_and_source_trace():
    r=ZhangXueWarpDistanceVector(1,hello_ttl_us=100)
    for a,b,at in ((0,1,0),(1,0,1),(1,2,2),(2,1,3)):
        hello(r,a,b,at)
    advert(r,1,0,4)
    snapshot=r.read_routes()
    assert snapshot["next_hop"][0,0,2]==1
    # Path 0->1->2 includes own source and destination; never 3.
    mask=int(snapshot["path_mask"][0,0,2])
    assert mask&(1<<0)
    assert mask&(1<<1)
    assert mask&(1<<2)
    assert not mask&(1<<3)
    assert snapshot["next_hop"][0,0,3]==-1


def test_three_node_cycle_guard_independent_across_4096_worlds():
    r=setup_triangle(4096)
    advert(r,2,0,109)
    snap=r.read_routes()
    assert np.all(snap["next_hop"][:,0,3]==-1)
    assert np.all(snap["next_hop"][:,1,3]==0)
    assert np.all(snap["next_hop"][:,2,3]==1)
