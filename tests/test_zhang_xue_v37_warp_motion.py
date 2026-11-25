"""Warp batched geometry/motion parity against the 1km CPU map environment.

No packet, SimPy or training code involved. Sequential resolution must match
the CPU reference's no-fly segment and 50m inter-UAV separation rules.
"""
import numpy as np
import pytest
pytest.importorskip("warp")

from uav_network.environment.zhang_xue_coop import ZhangXueCooperativeEnv,ZhangXueConfig
from uav_network.environment.zhang_xue_map_geometry import sample_scenario_batch
from uav_network.environment.gpu.zhang_xue_motion_warp import warp_move_batch


def test_parallel_map_movement_matches_cpu_reference_64_random_scenes():
    config=ZhangXueConfig()
    seeds=np.arange(1200,1264)
    scenes=sample_scenario_batch(config,seeds)
    rng=np.random.default_rng(55)
    actions=rng.uniform(-1.25,1.25,(len(seeds),6,2)).astype(np.float32)
    newpos,vel,safety,boundary=warp_move_batch(
        scenes["positions"],scenes["no_fly_zones"],actions,
        dt=1.,max_speed=50.,safe_distance=50.,device="cpu",
    )
    for idx,seed in enumerate(seeds):
        env=ZhangXueCooperativeEnv(config,seed=int(seed))
        cpu_safety,cpu_boundary=env._move(actions[idx])
        np.testing.assert_allclose(newpos[idx],env.positions,atol=.0002,rtol=0)
        np.testing.assert_allclose(vel[idx],env.velocity,atol=.0002,rtol=0)
        assert int(safety[idx])==cpu_safety
        assert int(boundary[idx])==cpu_boundary
    assert np.isfinite(newpos).all()


def test_motion_blocks_border_and_segment_intersection():
    config=ZhangXueConfig()
    env=ZhangXueCooperativeEnv(config,seed=44)
    pos=env.positions[None].copy()
    # Force UAV0 near the right map boundary; UAV1 in path of UAV0.
    pos[0,0]=[975,500]
    pos[0,1]=[800,500]
    # UAV1 starts at x=800 outside the disk; motion toward x=850
    # intersects the disk centered at x=840, radius 30.
    forbidden=np.array([[[100,100,20],[840,500,30]]],np.float32)
    action=np.zeros((1,6,2),np.float32)
    action[0,0]=[1,0]
    action[0,1]=[1,0]
    out,vel,safety,boundary=warp_move_batch(pos,forbidden,action,device="cpu")
    assert int(boundary[0])==1
    np.testing.assert_array_equal(out[0,0],pos[0,0])
    assert int(safety[0])>=1
    np.testing.assert_array_equal(out[0,1],pos[0,1])


def test_motion_input_shape_rejected():
    with pytest.raises(ValueError):
        warp_move_batch(np.zeros((2,6,2)),np.zeros((2,2,3)),np.zeros((6,2)))
