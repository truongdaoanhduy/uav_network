"""Scene sampling contract for offline 4096-env Warp map resets, without SimPy/network."""
import numpy as np
import pytest
from uav_network.environment.zhang_xue_coop import ZhangXueConfig, ZhangXueCooperativeEnv
from uav_network.environment.zhang_xue_map_geometry import sample_scenario_batch


def test_reset_scene_batch_matches_cpu_reference_byte_for_byte():
    config=ZhangXueConfig()
    seeds=[44,45,99,20261125,0,8888]
    scenes=sample_scenario_batch(config,seeds)
    assert scenes["positions"].shape==(len(seeds),6,2)
    assert scenes["target_mask"].shape==(len(seeds),400)
    assert scenes["no_fly_zones"].shape==(len(seeds),2,3)
    assert scenes["blackout_zones"].shape==(len(seeds),2,2)
    assert scenes["positions"].dtype==np.float32
    for e,seed in enumerate(seeds):
        cpu=ZhangXueCooperativeEnv(config,seed=seed)
        np.testing.assert_array_equal(cpu.positions,scenes["positions"][e])
        np.testing.assert_array_equal(cpu.target_mask,scenes["target_mask"][e])
        np.testing.assert_allclose(cpu.no_fly_zones,scenes["no_fly_zones"][e],rtol=0,atol=0)
        np.testing.assert_allclose(cpu.blackout_zones,scenes["blackout_zones"][e],rtol=0,atol=0)


def test_scene_batch_reseed_independent_and_complete_at_1024_envs():
    config=ZhangXueConfig()
    a=sample_scenario_batch(config,np.arange(44,1068))
    b=sample_scenario_batch(config,np.arange(44,1068))
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a["target_mask"].sum(axis=1),np.full(1024,10))
    assert len(np.unique(a["target_mask"].view(np.dtype((np.void,a["target_mask"].dtype.itemsize*400)))))>800


def test_reject_duplicate_scene_seeds():
    with pytest.raises(ValueError):
        sample_scenario_batch(ZhangXueConfig(),[4,4])
