"""TDD: positive Bayes posterior .99 is the ONLY confirmation gate (no LR/50m)."""
from pathlib import Path
import numpy as np
import pytest
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.bayesian import confirmation_allowed,confirmation_shaping_altitude,sensor_uniform
from uav_network.environment.gpu.bayesian import build_target_cell_lookup
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.observations import observation_version
from uav_network.environment.gpu.state import params_from_config
from uav_network.algorithms.checkpoints import validate_observation_version

CFG='configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml'
OLD='configs/matd3-nooracle-lr4-map800-h256x256-orthogonal-compile-tx08-learned-hf-20261123.yaml'


def test_posterior_only_new_mdp_config_and_legacy_preserved():
    c=load_config(CFG)
    assert c.sensor.confirmation_mode=='posterior_only'
    assert c.sensor.confirmation_threshold==.99
    assert c.sensor.confirmation_min_likelihood_ratio==0
    assert c.world.altitude_max_m==150
    assert c.sensor.belief_prior==.5
    assert c.network.tx_power_min_w==.001 and c.network.tx_power_max_w==.8
    assert c.network.power_control=='learned'
    assert c.world.map_size_m==800 and c.sensor.grid_cell_m==20
    assert c.matd3.hidden_sizes==[256,256] and c.matd3.initialization=='orthogonal'
    assert c.runtime.compile_models
    assert c.runtime.num_envs==4096 and c.matd3.batch_size==8192
    assert c.matd3.updates_per_step==4
    assert c.runtime.episodes==20000 and c.runtime.eval_episodes==1000
    assert c.tracking.periodic_evaluation_episodes==100
    assert c.tracking.deferred_gpu_eval_collection_stride==100
    assert c.tracking.deferred_gpu_eval_keep_all and c.tracking.deferred_gpu_eval_enabled
    assert c.reward.communication_reference_tx_power_w==.4
    assert c.tracking.checkpoint_repo=='duy95/uav-search-checkpoints'
    assert c.tracking.run_name!=load_config(OLD).tracking.run_name
    assert observation_version(c)==11
    assert observation_version(load_config(OLD))==10
    assert params_from_config(c).confirmation_mode==2
    assert params_from_config(c).confirm_capacity>=100
    assert params_from_config(c).confirmation_shaping_altitude==pytest.approx(150)


def test_posterior_only_never_requires_lr_or_50m_but_legacy_still_does():
    c=load_config(CFG)
    for altitude in (1,50,100,150):
        assert confirmation_allowed(c.sensor,altitude,.7,.3)
        assert confirmation_allowed(c.sensor,altitude,.5,.5)
    old=load_config(OLD)
    assert not confirmation_allowed(old.sensor,150,.7,.3)
    assert not confirmation_allowed(old.sensor,50,.7,.3)
    assert confirmation_allowed(old.sensor,50,.9,.1)
    with pytest.raises(ValueError,match='observation semantics differ'):
        validate_observation_version({'config':c.to_dict(),'observation_version':10})
    validate_observation_version({'config':c.to_dict(),'observation_version':11})


def test_posterior_only_allows_non_lr_threshold_but_rejects_invalid_modes():
    c=load_config(CFG)
    c.sensor.confirmation_min_likelihood_ratio=0
    c.validate()
    c.sensor.confirmation_mode='likelihood_ratio'
    with pytest.raises(ValueError,match='[Ll]ikelihood'):
        c.validate()
    c.sensor.confirmation_mode='garbage'
    with pytest.raises(ValueError):c.validate()


def test_real_warp_can_confirm_at_150_without_lr4_when_posterior_high():
    c=load_config(CFG)
    c.runtime.device='cpu';c.runtime.num_envs=1
    c.world.num_uavs=c.world.num_targets=1
    c.world.num_obstacles=0;c.network.enabled=False
    seed=next(s for s in range(1,10000) if sensor_uniform(s,1,0,820)<.69)
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[seed])
        xy=torch.tensor([412.5,412.5],dtype=torch.float64) # cell (20,20)=820 at width20
        wp.to_torch(v.s.positions)[0,0]=xy
        wp.to_torch(v.s.targets)[0,0]=xy
        wp.to_torch(v.s.altitudes)[0,0]=150.
        wp.to_torch(v.s.belief).fill_(.5)
        wp.to_torch(v.s.belief)[0,0,820]=.99999
        mask=wp.from_torch(torch.ones(1,dtype=torch.int32))
        v._launch(build_target_cell_lookup,1,v.s,v.p,mask,v.target_cell_lookup)
        v.step({'continuous':np.zeros((1,1,4),np.float32),'destination':np.zeros((1,1),np.int64)})
        assert bool(wp.to_torch(v.s.known)[0,0,0]),'High-altitude confirmation must not require LR>=4'
        assert int(wp.to_torch(v.s.report_cell)[0,0])==820
        assert not bool(wp.to_torch(v.s.errors).any())
    finally:v.close()


def test_metrics_label_posterior_only_not_lr():
    from uav_network.communication.gpu.metrics import world_info
    c=load_config(CFG)
    c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        assert world_info(v.s,v.p,0)['confirmation_source']=='posterior_only_no_truth_filter'
    finally:v.close()


def test_posterior_only_investigation_shaping_does_not_reward_descent_by_itself():
    from types import SimpleNamespace
    from uav_network.environment.bayesian import investigation_potential
    c=load_config(CFG)
    belief=np.full((c.world.num_uavs,40,40),.5,dtype=np.float64)
    belief[:,20,20]=.995
    positions=np.full((c.world.num_uavs,3),400.,dtype=np.float64)
    positions[:,2]=150
    sample=SimpleNamespace(config=c,belief=belief,positions=positions,
                           active=np.ones(c.world.num_uavs,dtype=np.bool_))
    high=investigation_potential(sample)
    positions[:,2]=50.
    low=investigation_potential(sample)
    assert high>0
    assert high==pytest.approx(low,abs=1e-12), 'No hidden LR/50m descent preference in shaping'


def test_high_altitude_false_positive_is_not_truth_filtered_in_posterior_only():
    # This characterizes the risk of dropping LR, it is NOT a desirable reward.
    c=load_config(CFG)
    c.runtime.device='cpu';c.runtime.num_envs=1
    c.world.num_uavs=c.world.num_targets=1;c.world.num_obstacles=0;c.network.enabled=False
    seed=next(s for s in range(1,10000) if sensor_uniform(s,1,0,820)<.29)
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[seed])
        wp.to_torch(v.s.positions)[0,0]=torch.tensor([412.5,412.5],dtype=torch.float64)
        wp.to_torch(v.s.targets)[0,0]=torch.tensor([100.,100.],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[0,0]=150.
        wp.to_torch(v.s.belief).fill_(.5)
        wp.to_torch(v.s.belief)[0,0,820]=.99999
        mask=wp.from_torch(torch.ones(1,dtype=torch.int32))
        v._launch(build_target_cell_lookup,1,v.s,v.p,mask,v.target_cell_lookup)
        v.step({'continuous':np.zeros((1,1,4),np.float32),
                'destination':np.zeros((1,1),np.int64)})
        cells=wp.to_torch(v.s.report_cell)[0].tolist()
        assert 820 in cells, 'Camera yes+posterior>=.99 is enough even for false cell'
        from uav_network.communication.gpu.metrics import world_info
        info=world_info(v.s,v.p,0)
        assert info['false_confirmed_cells']>0
        assert info['targets_found']==0
    finally:v.close()
