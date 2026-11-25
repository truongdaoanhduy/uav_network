"""Gate 5G regression: genuine Warp time ordering, retry, blackout and TTL."""
import numpy as np
import pytest
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)


def _worlds(e=4):
    pos=np.full((e,6,3),3000.,np.float32)
    pos[:,:,2]=100.
    pos[:,0,:]=[0,0,100.]
    pos[:,1,:]=[400,0,100.]
    pos[:,2,:]=[800,0,100.]
    dest=np.full((e,6),-1,np.int32)
    dest[:,[0,2]]=1
    sizes=np.zeros((e,6),np.int32)
    sizes[:,[0,2]]=566
    channels=np.ones((e,6),np.int32)
    slots=np.zeros((e,6,5),np.int32)
    ttl=np.full((e,6),10_000_000.,np.float32)
    return pos,dest,sizes,channels,slots,ttl


def test_hidden_terminal_retries_all_five_and_orthogonal_channel_recovers():
    pos,dest,sizes,ch,slots,ttl=_worlds(2)
    ch[1,2]=6
    got=simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    np.testing.assert_array_equal(got["delivered"][:,[0,2]],[[0,0],[1,1]])
    np.testing.assert_array_equal(got["attempts"][:,[0,2]],[[5,5],[1,1]])
    np.testing.assert_array_equal(got["dropped"][:,[0,2]],[[1,1],[0,0]])
    assert got["collided"][0,[0,2],:].all()


def test_sources_in_cca_range_defer_and_both_deliver():
    pos,dest,sizes,ch,slots,ttl=_worlds(1)
    pos[0,2,:]=[200.,0.,100.]
    got=simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    assert np.array_equal(got["delivered"][0,[0,2]],[1,1])
    assert got["cca_deferrals"][0,2]>=1
    assert got["start_us"][0,2,0]>got["end_us"][0,0,0]


def test_blackout_does_not_teleport_data_and_deadline_expires():
    pos,dest,sizes,ch,slots,ttl=_worlds(2)
    dest[:,2]=-1
    black=np.zeros((2,6),np.int32)
    black[0,0]=1
    ttl[1,0]=1000.  # insufficient for 566 B airtime+MAC ACK
    got=simulate_one_hop_retry_batch(
        pos,dest,sizes,ch,slots,ttl,blackout=black,
    )
    assert got["attempts"][0,0]==0
    assert got["expired"][0,0]==1
    assert got["attempts"][1,0]==1
    assert got["expired"][1,0]==1
    assert not got["delivered"].any()


def test_input_validation_rejects_out_of_cw_and_self_destination():
    pos,dest,sizes,ch,slots,ttl=_worlds(1)
    slots[0,0,0]=31
    with pytest.raises(ValueError,match="backoff"):
        simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    slots[0,0,0]=0
    dest[0,0]=0
    with pytest.raises(ValueError,match="destination"):
        simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)


def test_4096_worlds_bounded_without_any_network_leakage():
    pos,dest,sizes,ch,slots,ttl=_worlds(4096)
    ch[2048:,2]=6
    got=simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    assert got["attempts"].shape==(4096,6)
    assert int(got["delivered"].sum())==4096
    assert int(got["dropped"].sum())==4096
    assert int(got["attempts"].sum())==4096*6


def test_mac_ack_can_collide_even_after_data_reaches_relay():
    pos,dest,sizes,ch,slots,ttl=_worlds(2)
    pos[:,2,:]=[760.,0.,100.]
    sizes[:,0]=20  # DATA at t50..130us, ACK at t140..260us
    slots[:,2,0]=5  # hidden transmitter enters at t150us
    ch[1,2]=6       # different channel in world 1 is the control
    got=simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    assert got["collided"][0,0,0]==0
    assert got["ack_collided"][0,0,0]==1
    assert got["attempts"][0,0]>=2
    assert got["ack_collided"][1,0,0]==0
    assert got["delivered"][1,0]==1


def test_128_seeded_worlds_have_bounded_attempts_unique_terminal_status_and_time_order():
    rng=np.random.default_rng(20261125)
    pos,dest,sizes,ch,slots,ttl=_worlds(128)
    for a in range(5):
        slots[:,:,a]=rng.integers(0,32*(2**a)-1,size=(128,6))
    got=simulate_one_hop_retry_batch(pos,dest,sizes,ch,slots,ttl)
    active=dest>=0
    assert np.all((got["attempts"][active]>=1)&(got["attempts"][active]<=5))
    terminal=got["delivered"]+got["dropped"]+got["expired"]
    assert np.array_equal(terminal[active],np.ones(active.sum(),np.int32))
    assert np.array_equal(terminal[~active],np.zeros((~active).sum(),np.int32))
    assert np.array_equal(got["delivered"][~active],np.zeros((~active).sum(),np.int32))
    for i in range(128):
        for u in (0,2):
            n=int(got["attempts"][i,u])
            starts=got["start_us"][i,u,:n]
            ends=got["end_us"][i,u,:n]
            assert np.all(starts>=0) and np.all(ends>starts)
            assert np.all(np.diff(starts)>0)
            np.testing.assert_array_equal(got["start_us"][i,u,n:],-1.)
