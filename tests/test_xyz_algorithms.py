"""Four-slot actions must retain az while masking only RF power."""
import numpy as np
import pytest
import torch
from uav_network.configuration import Config
from uav_network.algorithms.checkpoints import make_learner,save_policy,load_policy
from uav_network.algorithms.action_utils import canonical_continuous,joint_action
from uav_network.experiments.pql_replay import make_replay
from uav_network.environment.observations import observation_dim,state_dim
from uav_network.environment.env import Action


@pytest.mark.parametrize("algorithm",["masac","matd3"])
@pytest.mark.parametrize("actor,critic",[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize("power",["learned","link_budget"])
def test_xyz_policy_replay_and_real_updates(tmp_path,algorithm,actor,critic,power):
    torch.set_num_threads(1);torch.manual_seed(44)
    c=Config();c.world.motion_dimensions=3;c.sensor.model="bayesian";c.world.num_uavs=2
    c.runtime.algorithm=algorithm;c.runtime.vectorize_critics=True;c.runtime.critic_chunk_size=2
    c.network.power_control=power
    a=c.algorithm_config;a.hidden_sizes=[8,8];a.batch_size=4;a.replay_capacity=16
    a.share_actor_parameters=actor;a.share_critic_parameters=critic
    if algorithm=="masac":a.target_continuous_entropy=-4
    c.validate()
    obs=np.zeros((4,2,observation_dim(c)),np.float32);obs[:,:,9]=1;obs[:,:,-3:]=1
    state=np.zeros((4,state_dim(c)),np.float32)
    learner=make_learner(c,observation_dim(c),state_dim(c))
    action=learner.act_batch(obs)
    assert action.continuous.shape==(4,2,4)
    if power=="link_budget": assert np.all(action.continuous[:,:,-1]==0)
    replay=make_replay(c,(2,observation_dim(c)),state_dim(c),"cpu")
    replay.add_batch(obs,state,action,np.ones(4),obs,state,np.zeros(4))
    sample=replay.sample(4)
    assert sample["continuous"].shape==(4,2,4)
    for _ in range(2):
        metrics=learner.update(sample)
        assert all(np.isfinite(v) for v in metrics.values())
    assert learner.actor_updates>0
    p=tmp_path/"policy.pt";save_policy(learner,p)
    assert torch.load(p,weights_only=True)["observation_version"]==4
    restored=load_policy(p)
    np.testing.assert_array_equal(learner.act_batch(obs,True).continuous,restored.act_batch(obs,True).continuous)


def test_xyz_az_is_not_power_or_silence_mask():
    x=torch.ones((1,4),requires_grad=True);obs=torch.zeros((1,20));obs[:,9]=1
    y=canonical_continuous(x,obs,torch.tensor([0]),learn_power=False)
    assert y.tolist()==[[1,1,1,0]]
    y.sum().backward()
    assert x.grad.tolist()==[[1,1,1,0]]
    assert joint_action(x,torch.tensor([1]),3,learn_power=False)[0,:4].tolist()==[1,1,1,0]
