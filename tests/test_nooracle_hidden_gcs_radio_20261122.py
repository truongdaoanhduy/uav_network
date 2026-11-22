"""Local Actor observations may not reveal unobserved 3D obstacle occlusion."""
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.gpu.observations import observe


def test_no_oracle_actor_gcs_signal_feature_does_not_reveal_unobserved_obstacle():
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.num_envs=1
    c.runtime.device='cpu'
    assert c.sensor.report_cell_hypotheses
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        positions=wp.to_torch(v.s.positions)
        gcs=wp.to_torch(v.s.gcs)
        obstacles=wp.to_torch(v.s.obstacles)
        heights=wp.to_torch(v.s.obstacle_heights)
        altitude=wp.to_torch(v.s.altitudes)
        positions[0,0]=torch.tensor([500.,500.],dtype=positions.dtype)
        gcs[0]=torch.tensor([300.,500.],dtype=gcs.dtype)
        altitude[0,0]=50.
        obstacles[0,:,:]=torch.tensor([900.,900.,10.],dtype=obstacles.dtype)
        heights[0,:]=1.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        before=wp.to_torch(v.s.observation)[0,0].clone()
        # An occluder is 100 m away; the UAV's current camera footprint is
        # ~50 m, so it cannot see this obstacle, but the global PHY oracle can.
        obstacles[0,0,:]=torch.tensor([400.,500.,20.],dtype=obstacles.dtype)
        heights[0,0]=150.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        after=wp.to_torch(v.s.observation)[0,0].clone()
        assert torch.equal(before,after),[ (i,float(a),float(b)) for i,(a,b) in enumerate(zip(before,after)) if a!=b ]
    finally:
        v.close()


def test_nooracle_observation_version_changes_when_hidden_gcs_oracle_removed():
    # Equal tensor shape is not enough: old v7 Actor learned the exact radio
    # effect of 3D obstacles it could not see. Never resume that checkpoint.
    import pytest
    from uav_network.environment.observations import observation_version
    from uav_network.algorithms.checkpoints import validate_observation_version
    cfg=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    assert observation_version(cfg) == 8
    historical_policy={'config':cfg.to_dict(),'observation_version':7}
    with pytest.raises(ValueError,match='observation semantics differ'):
        validate_observation_version(historical_policy)
