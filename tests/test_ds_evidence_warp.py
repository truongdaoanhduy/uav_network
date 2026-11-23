"""Native/Warp DS map parity. The GPU path only reads noisy observation events."""
import numpy as np
import pytest
import warp as wp
from uav_network.environment.ds_evidence import combine_evidence
from uav_network.environment.gpu.ds_evidence import ds_update_observations, ds_reset


@pytest.mark.parametrize('device', ['cpu'] + (['cuda:0'] if wp.is_cuda_available() else []))
def test_warp_ds_mass_matches_native_across_positive_negative_and_missing(device):
    wp.init()
    shape = (2, 2, 4)
    f = wp.zeros(shape, dtype=wp.float32, device=device)
    e = wp.zeros(shape, dtype=wp.float32, device=device)
    u = wp.zeros(shape, dtype=wp.float32, device=device)
    wp.launch(ds_reset, dim=shape, inputs=[f,e,u], device=device)
    obs = np.full(shape,-1,dtype=np.int32)
    obs[0,0,0] = 1
    obs[0,0,1] = 0
    obs[0,1,2] = 1
    pos_c = np.full(shape,0.8,dtype=np.float32)
    neg_c = np.full(shape,0.8,dtype=np.float32)
    pos_c[0,1,2] = 0.6
    for readings in [obs,obs,np.where(obs==1,0,np.where(obs==0,1,-1)).astype('int32')]:
        wp.launch(ds_update_observations, dim=shape, inputs=[f,e,u,
            wp.array(readings,device=device),wp.array(pos_c,device=device),wp.array(neg_c,device=device)], device=device)
    actual = np.stack((f.numpy(),e.numpy(),u.numpy()),axis=-1)
    expected = np.zeros(shape+(3,),dtype=np.float64)
    expected[...,2] = 1.0
    for idx in np.ndindex(shape):
        old = (0.0,0.0,1.0)
        for readings in [obs,obs,np.where(obs==1,0,np.where(obs==0,1,-1)).astype('int32')]:
            observed = readings[idx]
            if observed >= 0:
                c = pos_c[idx] if observed == 1 else neg_c[idx]
                old = combine_evidence(old,bool(observed),c)
        expected[idx] = old
    assert np.allclose(actual,expected,rtol=1e-5,atol=1e-6)
    assert np.allclose(actual.sum(-1),1,atol=1e-6)
