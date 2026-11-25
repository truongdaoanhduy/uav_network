"""Actual source-level v4.11 regressions: production MAC + actor obs, not audit helpers."""
import numpy as np

from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)
from uav_network.environment.zhang_xue_coop import (
    ZhangXueConfig, ZhangXueCooperativeEnv,
)


def _concurrent_ack_and_cross_channel_data():
    """At t=150us UAV1 is transmitting ACK to UAV0 while UAV2 sends DATA to UAV1.

    Source2 is 800m away from ACK TX UAV1 (>750m source carrier sense).
    Different MAC channels prevent RF SINR collision from masking half-duplex.
    Tx power 2W produces a decodable UAV2->UAV1 isolated DATA link.
    """
    positions=np.zeros((1,6,3),np.float32)
    positions[0,:,0]=[0,500,1300,2600,3000,3500]
    positions[0,:,2]=100
    dest=np.full((1,6),-1,np.int32)
    dest[0,0]=1;dest[0,2]=1
    sizes=np.zeros((1,6),np.int32);sizes[0,[0,2]]=20
    channels=np.ones((1,6),np.int32);channels[0,2]=6
    slots=np.zeros((1,6,5),np.int32);slots[0,2,0]=5
    deadline=np.full((1,6),5000,np.float32)
    deadline[0,2]=240.0  # only first DATA attempt can complete before TTL
    return simulate_one_hop_retry_batch(
        positions,dest,sizes,channels,slots,deadline,
        tx_power_w=2.0,physical_carrier_sense=True,device="cpu",
    )


def test_ack_transmitter_cannot_receive_cross_channel_data_at_same_time():
    got=_concurrent_ack_and_cross_channel_data()
    assert got["start_us"][0,0,0]==50.0
    assert got["start_us"][0,2,0]==150.0
    assert got["data_received"][0,0]==1
    assert got["data_received"][0,2]==0, "ACK transmitter U1 must not decode U2 DATA"
    assert got["collided"][0,2,0]==1


def test_nfz_zero_distance_and_empty_slot_must_be_distinguishable():
    env=ZhangXueCooperativeEnv(seed=44)
    env.positions[0]=[200.,200.]
    env.no_fly_zones=[(200.,200.,50.),(900.,900.,50.)]
    here,_,_=env._build_outputs()
    env.no_fly_zones=[(850.,850.,50.),(900.,900.,50.)]
    absent,_,_=env._build_outputs()
    # Each slot: relative-x, relative-y, distance, valid. Empty is d=FOV and valid=0.
    assert here.shape==(6,121)
    np.testing.assert_allclose(here[0,-8:-4],[0.,0.,0.,1.])
    np.testing.assert_allclose(absent[0,-8:-4],[0.,0.,.3,0.])
    np.testing.assert_allclose(absent[0,-4:],[0.,0.,.3,0.])
    assert env.obs_dim==121


def test_float32_microsecond_clock_precision_is_not_sufficient_for_200s():
    assert np.float32(20_000_000)==np.float32(20_000_001)
    assert np.spacing(np.float32(200_000_000))==16.0
    assert 200_000_000<2_147_483_647


def test_cpu_reference_reset_none_advances_rng_for_episode_diversity():
    a=ZhangXueCooperativeEnv(seed=44)
    b=ZhangXueCooperativeEnv(seed=44)
    first_a=a.target_mask.copy()
    first_b=b.target_mask.copy()
    np.testing.assert_array_equal(first_a,first_b)
    a.reset()
    b.reset()
    np.testing.assert_array_equal(a.target_mask,b.target_mask)
    assert not np.array_equal(first_a,a.target_mask), "reset(None) cannot replay one scene forever"
    a.reset(seed=44)
    np.testing.assert_array_equal(first_a,a.target_mask)
