"""NoOracle: hidden geometry and target truth must not enter policy observations."""
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.observations import observe, centralized
from uav_network.environment.gpu.vector import WarpVectorEnv

CONFIG='configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml'


def test_obstacle_behind_visible_tall_obstacle_does_not_leak_geometry_to_actor():
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=1
    assert c.sensor.obstacle_occlusion and c.sensor.report_cell_hypotheses
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        positions=wp.to_torch(v.s.positions);positions[0,0]=torch.tensor([500.,500.],dtype=positions.dtype)
        wp.to_torch(v.s.altitudes)[0,0]=50.
        obstacles=wp.to_torch(v.s.obstacles);heights=wp.to_torch(v.s.obstacle_heights)
        obstacles[0,:,:]=torch.tensor([900.,900.,5.],dtype=obstacles.dtype)
        heights[0,:]=1.
        obstacles[0,0,:]=torch.tensor([520.,500.,10.],dtype=obstacles.dtype)
        heights[0,0]=120.
        obstacles[0,1,:]=torch.tensor([545.,500.,5.],dtype=obstacles.dtype)
        heights[0,1]=100.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        before=wp.to_torch(v.s.observation)[0,0].clone()
        # Same camera-visible first obstacle; geometry of obstacle behind it
        # changes but is fully occluded by that first obstacle.
        obstacles[0,1,:]=torch.tensor([550.,500.,7.],dtype=obstacles.dtype)
        heights[0,1]=60.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        after=wp.to_torch(v.s.observation)[0,0].clone()
        assert torch.equal(before,after),[(i,float(a),float(b)) for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    finally:v.close()


def test_critic_and_actor_do_not_expose_unobserved_target_coordinates():
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        v._launch(centralized,1,v.s,v.p)
        before_actor=wp.to_torch(v.s.observation).clone()
        before_critic=wp.to_torch(v.s.state).clone()
        # Ground truth is allowed in sensor sampling on the next step only.
        # It must not affect a same-step read of policy features.
        truth=wp.to_torch(v.s.targets)
        truth[0,0]=torch.tensor([125.,975.],dtype=truth.dtype)
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        v._launch(centralized,1,v.s,v.p)
        assert torch.equal(before_actor,wp.to_torch(v.s.observation))
        assert torch.equal(before_critic,wp.to_torch(v.s.state))
    finally:v.close()


def test_native_bayesian_nooracle_must_fail_closed_instead_of_emitting_true_target_ids():
    import pytest
    from uav_network.environment.env import SwarmSearchEnv
    c=load_config(CONFIG)
    with pytest.raises(NotImplementedError,match='NoOracle|oracle|cell ID'):
        SwarmSearchEnv(c)


def test_configuration_rejects_nooracle_native_backend_before_production():
    import pytest
    c=load_config(CONFIG)
    c.runtime.environment_backend='native'
    with pytest.raises(ValueError,match='NoOracle|oracle|warp_packet'):
        c.validate()


def test_new_visibility_science_requires_new_checkpoint_version():
    from uav_network.environment.observations import observation_version
    c=load_config(CONFIG)
    assert observation_version(c)==10


def test_visible_obstacle_ranking_and_heights_remain_aligned_after_occlusion():
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        wp.to_torch(v.s.positions)[0,0]=torch.tensor([500.,500.],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[0,0]=50.
        o=wp.to_torch(v.s.obstacles)
        h=wp.to_torch(v.s.obstacle_heights)
        o[0]=torch.tensor([[900.,900.,5.]]*c.world.num_obstacles,dtype=torch.float64)
        h[0]=1.
        # First visible, second behind tall occluder (hidden), third clear.
        o[0,0]=torch.tensor([520.,500.,10.],dtype=torch.float64)
        h[0,0]=120.
        o[0,1]=torch.tensor([545.,500.,5.],dtype=torch.float64)
        h[0,1]=100.
        o[0,2]=torch.tensor([500.,540.,6.],dtype=torch.float64)
        h[0,2]=30.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        row=wp.to_torch(v.s.observation)[0,0]
        obstacle_offset=10+2*c.world.num_targets+4*c.sensor.visible_target_slots
        # The second slot must hold the third real obstacle, not the hidden one.
        assert row[obstacle_offset+3].item()==1.
        assert row[obstacle_offset+4+3].item()==1.
        assert row[obstacle_offset+8+3].item()==0.
        torch.testing.assert_close(row[obstacle_offset+4:obstacle_offset+6],
                                   torch.tensor([0.,.04]),atol=1e-6,rtol=0)
        height_offset=(obstacle_offset+4*c.sensor.visible_obstacle_slots
                       +6*(c.world.num_uavs-1)
                       +(3*c.world.num_uavs if c.network.link_state_observation else 0)
                       +c.sensor.coverage_grid_cells**2+2)
        assert abs(row[height_offset].item()-120/150)<1e-6
        assert abs(row[height_offset+1].item()-30/150)<1e-6
        assert row[height_offset+2].item()==0.
        # Shortening the first cylinder makes the previously hidden second visible.
        h[0,0]=5.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        now=wp.to_torch(v.s.observation)[0,0]
        assert now[obstacle_offset+8+3].item()==1.
    finally:v.close()


def test_eval_and_train_seeds_are_disjoint_in_new_production_configuration():
    c=load_config(CONFIG)
    assert c.runtime.seed == 44
    assert c.tracking.periodic_evaluation_seed==1_000_000_044
    assert c.runtime.eval_seed==2_000_000_044
    assert c.runtime.seed + c.runtime.episodes < c.tracking.periodic_evaluation_seed
    assert c.tracking.periodic_evaluation_seed + max(c.tracking.periodic_evaluation_episodes,1) < c.runtime.eval_seed


def test_camera_off_at_zero_altitude_cannot_reveal_obstacle_geometry():
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        wp.to_torch(v.s.positions)[0,0]=torch.tensor([500.,500.],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[0,0]=0.
        obstacles=wp.to_torch(v.s.obstacles)
        obstacles[0]=torch.tensor([[900.,900.,5.]]*c.world.num_obstacles,dtype=torch.float64)
        obstacles[0,0]=torch.tensor([505.,500.,10.],dtype=torch.float64)
        heights=wp.to_torch(v.s.obstacle_heights);heights[0]=1.;heights[0,0]=60.
        v._launch(observe,(1,c.world.num_uavs),v.s,v.p)
        row=wp.to_torch(v.s.observation)[0,0]
        offset=10+2*c.world.num_targets+4*c.sensor.visible_target_slots
        assert not bool(row[offset:offset+4*c.sensor.visible_obstacle_slots].any())
    finally:v.close()


def test_quality_gated_nooracle_scoring_metadata_never_mislabels_posterior_only():
    from uav_network.communication.gpu.metrics import world_info
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        info=world_info(v.s,v.p,0)
        assert info['confirmation_source']=='posterior_plus_likelihood_ratio_no_truth_filter'
    finally:v.close()
