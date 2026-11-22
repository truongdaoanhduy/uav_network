"""New scientific MDP: 800 m world, H256x256, orthogonal MATD3, torch.compile."""
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from torch import nn

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.observations import observation_version
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.models import initialize_matd3_orthogonal

CONFIG='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml'
HIST='configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml'


def test_new_config_has_exact_requested_science_settings_and_does_not_mutate_old():
    c=load_config(CONFIG)
    old=load_config(HIST)
    assert c.world.map_size_m==800
    assert c.world.obstacle_radius_min_m==24
    assert c.world.obstacle_radius_max_m==56
    assert c.world.obstacle_clearance_m==8
    assert c.world.altitude_max_m==150
    assert c.sensor.grid_cell_m==20
    assert c.sensor.confirmation_mode=='likelihood_ratio'
    assert c.sensor.confirmation_threshold==.99
    assert c.matd3.initialization=='orthogonal'
    assert c.matd3.hidden_sizes==[256,256]
    assert c.runtime.compile_models is True
    assert c.matd3.actor_batch_norm and c.matd3.actor_layer_norm
    assert c.matd3.critic_batch_norm is False
    assert c.runtime.num_envs==4096 and c.matd3.batch_size==8192
    assert c.matd3.updates_per_step==4 and c.runtime.episodes==20_000
    assert c.runtime.eval_episodes==1000
    assert c.network.tx_power_max_w==.4
    assert c.runtime.warm_start_checkpoint=='none'
    assert observation_version(c)==10
    assert old.world.map_size_m==1000 and old.matd3.hidden_sizes==[256,512]
    assert old.runtime.compile_models is False
    assert old.matd3.initialization=='default'
    assert c.tracking.run_name!=old.tracking.run_name


@pytest.mark.parametrize('seed',[44,45,1000000044,2000000044,805,806])
def test_800m_scene_has_all_35_distinct_target_cells_and_valid_geometry(seed):
    c=load_config(CONFIG)
    c.runtime.num_envs=1;c.runtime.device='cpu'
    v=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=v.reset(seed=[seed])
        assert v.p.grid_n==40
        assert obs['observation'].shape[0]==1
        targets=wp.to_torch(v.s.targets)[0].cpu().numpy()
        assert np.isfinite(targets).all()
        assert np.min(targets)>=0 and np.max(targets)<800
        cells=np.floor(targets/20).astype(int)
        assert len(np.unique(cells[:,1]*40+cells[:,0]))==35
        # No world-generation capacity / duplicate-cell internal errors.
        assert not bool(wp.to_torch(v.s.errors).any())
        assert np.linalg.norm(targets-np.array([400.,0.]),axis=1).min()>=200
    finally:v.close()


def test_orthogonal_network_initializes_both_actor_and_critic_and_targets():
    c=load_config(CONFIG)
    c.runtime.device='cpu'
    torch.manual_seed(44)
    from uav_network.environment.observations import observation_dim, state_dim
    n=MATD3(c,observation_dim(c),state_dim(c))
    from math import sqrt
    act=n.actors[0]
    for module in act.encoder.modules():
        if isinstance(module,nn.Linear):
            w=module.weight.detach()/sqrt(2)
            gram=w@w.T if w.shape[0]<=w.shape[1] else w.T@w
            torch.testing.assert_close(gram,torch.eye(gram.shape[0]),rtol=1e-5,atol=1e-5)
    assert all(torch.count_nonzero(v.bias)==0 for v in act.encoder.modules() if isinstance(v,nn.Linear))
    for online,target in ((n.actors,n.target_actors),(n.critics,n.target_critics)):
        for a,b in zip(online.parameters(),target.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_800m_one_real_warp_step_with_radio_and_valid_belief():
    c=load_config(CONFIG);c.runtime.device='cpu';c.runtime.num_envs=2
    v=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=v.reset(seed=[44,45])
        for _ in range(4):
            obs,reward,terminated,truncated,info=v.step({'continuous':np.zeros((2,6,4),np.float32),
                'destination':np.zeros((2,6),np.int64)})
            assert torch.isfinite(reward).all()
            assert not bool(wp.to_torch(v.s.errors).any())
        assert v.p.map_size==800
    finally:v.close()
