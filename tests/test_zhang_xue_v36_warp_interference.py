"""Gate 5D: Warp FP32 simultaneous receiver-interference kernel.

This is a limited PHY receiving-window kernel; not MAC contention/backoff,
RTX/ACK, or complete SimPy-Warp parity.
"""
import numpy as np
import pytest

pytest.importorskip("warp")

from uav_network.communication.gpu.classic_v1_interference_warp import (
    compute_simultaneous_link_sinr,
)


def _scenes():
    xyz=np.zeros((3,3,3),np.float32)
    xyz[:]=[[0.,0.,100.],[400.,0.,100.],[800.,0.,100.]]
    tx_src=np.array([[0,2]]*3,np.int32)
    rx_dst=np.array([[1,1]]*3,np.int32)
    channels=np.array([[1,1],[1,1],[1,6]],np.int32)
    start=np.array([[0.,.0002],[0.,.003],[0.,.0002]],np.float32)
    duration=np.full((3,2),.002264,np.float32)
    valid=np.ones((3,2),np.int32)
    return xyz,tx_src,rx_dst,channels,start,duration,valid


def test_warp_detects_hidden_terminal_collision_and_noninterfering_controls():
    xyz,src,dst,ch,begin,duration,valid=_scenes()
    got=compute_simultaneous_link_sinr(
        xyz,src,dst,ch,begin,duration,valid,
        tx_power_w=.5,carrier_frequency_hz=2.4e9,
        noise_power_w=4e-11,snr_threshold_db=6.,device="cpu",
    )
    assert got["sinr_db"].shape==(3,2)
    assert got["success"].shape==(3,2)
    np.testing.assert_array_equal(got["success"],[
        [0,0], # hidden terminals on same channel, overlapping frames
        [1,1], # same channel, but transmissions have no time overlap
        [1,1], # time overlap but orthogonal 1/6 channels
    ])
    assert float(got["sinr_db"][0].max())<6.
    assert float(got["sinr_db"][1].min())>6.
    assert float(got["sinr_db"][2].min())>6.


def test_warp_interfering_sinr_matches_classic_v1_analytic_formula():
    xyz,src,dst,ch,begin,duration,valid=_scenes()
    got=compute_simultaneous_link_sinr(
        xyz,src,dst,ch,begin,duration,valid,
        tx_power_w=.5,device="cpu",
    )
    import math
    distance=400.
    gain=(3e8/(4*math.pi*2.4e9*distance))**2
    received=.5*gain
    ideal_interference_db=10.*math.log10(received/(received+4e-11))
    np.testing.assert_allclose(
        got["sinr_db"][0], [ideal_interference_db]*2,atol=0.001,rtol=0,
    )


def test_invalid_indices_or_negative_duration_cannot_launch():
    xyz,src,dst,ch,begin,duration,valid=_scenes()
    with pytest.raises(ValueError):
        compute_simultaneous_link_sinr(
            xyz,np.full_like(src,6),dst,ch,begin,duration,valid,device="cpu")
    with pytest.raises(ValueError):
        compute_simultaneous_link_sinr(
            xyz,src,dst,ch,begin,-duration,valid,device="cpu")


def test_random_128_worlds_sinr_matches_independent_numpy_phy():
    """Check all scheduled links, not only an illustrative hidden-terminal case."""
    import math
    rng=np.random.default_rng(20261125)
    e,n,m=128,6,4
    xyz=rng.uniform(0,1000,(e,n,3)).astype(np.float32)
    xyz[:,:,2]=100.
    src=np.tile(np.arange(m,dtype=np.int32),(e,1))
    dst=np.full((e,m),4,np.int32)
    ch=rng.choice(np.array([1,3,6,11],np.int32),size=(e,m))
    begin=rng.choice(np.array([0.,.001,.004,.010],np.float32),size=(e,m))
    duration=np.full((e,m),.0024,np.float32)
    valid=rng.choice(np.array([0,1],np.int32),size=(e,m),p=[.12,.88])
    got=compute_simultaneous_link_sinr(
        xyz,src,dst,ch,begin,duration,valid,
        device="cpu",
    )
    expected=np.full((e,m),-1e9,np.float64)
    for world in range(e):
        for i in range(m):
            if valid[world,i]==0:
                continue
            r=xyz[world,dst[world,i]].astype(np.float64)
            t=xyz[world,src[world,i]].astype(np.float64)
            di=np.linalg.norm(r-t)
            power=.5*((3e8/(4*math.pi*2.4e9*di))**2)
            interference=0.
            for j in range(m):
                if i==j or valid[world,j]==0:
                    continue
                if abs(int(ch[world,i])-int(ch[world,j]))>=5:
                    continue
                if not (begin[world,i]<begin[world,j]+duration[world,j] and
                        begin[world,j]<begin[world,i]+duration[world,i]):
                    continue
                tj=xyz[world,src[world,j]].astype(np.float64)
                dj=np.linalg.norm(r-tj)
                interference+=.5*((3e8/(4*math.pi*2.4e9*dj))**2)
            expected[world,i]=10*math.log10(power/(4e-11+interference))
    active=valid.astype(bool)
    np.testing.assert_allclose(got["sinr_db"][active],expected[active],rtol=0,atol=.002)
    np.testing.assert_array_equal(got["success"][active],(expected[active]>=6).astype(np.int32))
    np.testing.assert_array_equal(got["success"][~active],0)
