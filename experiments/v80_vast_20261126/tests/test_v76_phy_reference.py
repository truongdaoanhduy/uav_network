"""Controlled geometry/PHY gate separate from easy end-to-end packet workload."""
import numpy as np
import pytest
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
from zhang_xue.gpu.runtime_v67 import HAS_WARP


def _positions():
    p = np.zeros((2, 3, 3), dtype=np.float32)
    p[:, 0, :] = (100.0, 100.0, 50.0)
    p[:, 1, :] = (200.0, 100.0, 50.0)
    p[:, 2, :] = (100.0, 200.0, 50.0)
    return p


def test_building_and_blackout_change_link_gain_in_reference():
    p = _positions()
    los = compute_channel_gain_v67_np(p, (), (), 0, grid_w=20, cell_size_m=50.0)
    nlos = compute_channel_gain_v67_np(p, ((150,100,15,100),), (), 0, grid_w=20, cell_size_m=50.0)
    blackout = compute_channel_gain_v67_np(p, (), ((2,2),), 0, grid_w=20, cell_size_m=50.0)
    assert np.all(los[:,0,1] > 0)
    assert np.all(nlos[:,0,1] < los[:,0,1])
    assert np.all(nlos[:,0,2] == los[:,0,2])
    assert np.all(blackout[:,0,:] == 0)
    assert np.all(blackout[:,:,0] == 0)
    assert np.all(los == los.transpose((0,2,1)))


@pytest.mark.skipif(not HAS_WARP, reason='warp-lang missing; numerical CPU reference still tested')
def test_warp_cpu_phy_matches_reference_for_buildings_and_blackout():
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    p = _positions()
    buildings = ((150,100,15,100),)
    for centers in ((), ((2,2),)):
        net = WarpCompressedRadioNetworkV67(2,num_uavs=2,grid_w=20,device='cpu',seed=46,q_cap=64)
        net.update_phy_from_positions(p,buildings,centers,0,50.0)
        actual = net.data['gain'].numpy()
        expected = compute_channel_gain_v67_np(p,buildings,centers,0,20,50.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0)
