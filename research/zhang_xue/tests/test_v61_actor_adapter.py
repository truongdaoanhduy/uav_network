"""TDD for the ACTOR-OBS/packet-destination mismatch in v6.0.

The active CPU scenario emits 514 features and a *separate* 7-way route mask.
Old MATD3/MASAC require obs[...,9] == alive and the last 7 values == mask.
Preserve the original distance feature while providing a policy-compatible view.
"""
import numpy as np
import pytest
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.environment.configuration import make_v57_config
from zhang_xue.learners.actor_adapter import (
    ACTOR_POLICY_DIM, encode_actor_input, decode_policy_action, make_critic_state
)


def fresh():
    env=ZhangXue7NodePacketEnv(make_v57_config(),seed=44)
    obs,state,mask=env.reset(seed=44)
    return env,obs,state,mask


def test_actor_observation_shape_no_truth_and_observable_mask():
    env,obs,state,mask=fresh()
    assert obs.shape==(6,514)
    assert mask.shape==(6,7)
    x=encode_actor_input(obs,mask)
    assert x.shape==(6,ACTOR_POLICY_DIM)
    assert ACTOR_POLICY_DIM==522
    np.testing.assert_array_equal(x[:,-7:],mask.astype(np.float32))
    np.testing.assert_array_equal(x[:,9],(obs[:,4]>0).astype(np.float32))
    np.testing.assert_allclose(x[:,-8],obs[:,9])
    np.testing.assert_allclose(x[:,:9],obs[:,:9])
    assert make_critic_state(x).shape==(6*ACTOR_POLICY_DIM,)
    # Two target placements shouldn't reveal labels in encoder with the same local evidence
    old_truth=env.true_targets.copy()
    env.true_targets=~env.true_targets
    try:
        np.testing.assert_allclose(x,encode_actor_input(obs,mask))
    finally:
        env.true_targets=old_truth


def test_mask_invalid_action_fails_closed_and_silent_is_legal():
    env,obs,state,mask=fresh()
    invalid=int(np.where(mask[0]==0)[0][0])
    comm=np.zeros(6,dtype=np.int64)
    comm[0]=invalid
    cont=np.zeros((6,3),dtype=np.float64)
    with pytest.raises(ValueError,match="unavailable"):
        decode_policy_action(cont,comm,mask,v_max_mps=env.cfg.v_max_mps)
    vel,dest=decode_policy_action(cont,np.zeros(6,np.int64),mask,v_max_mps=25.)
    np.testing.assert_array_equal(vel,np.zeros((6,2)))
    np.testing.assert_array_equal(dest,np.zeros(6))


def test_commands_bounded_even_for_out_of_range_policy_raw():
    env,obs,st,mask=fresh()
    cont=np.zeros((6,3),dtype=np.float64)
    cont[:,0]=999
    cont[:,1]=-999
    v,d=decode_policy_action(cont,np.zeros(6,np.int64),mask,v_max_mps=25.)
    assert np.max(v)<=25 and np.min(v)>=-25
    assert np.linalg.norm(v,axis=1).max()<=25.+1e-5


def test_end_to_end_actor_action_to_packet_env_one_transition():
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.configuration import LearnerConfig
    env,obs,st,mask=fresh()
    encoded=encode_actor_input(obs,mask)
    config=LearnerConfig().validate()
    model=MATD3(config,ACTOR_POLICY_DIM,6*ACTOR_POLICY_DIM)
    action=model.act(encoded,deterministic=True)
    vel,dest=decode_policy_action(action.continuous,action.destination,mask,v_max_mps=25.)
    next_obs,next_state,next_mask,reward,terminated,truncated,info=env.step(vel,dest)
    assert next_obs.shape==(6,514)
    assert next_mask.shape==(6,7)
    assert np.isfinite(reward)
    assert make_critic_state(encode_actor_input(next_obs,next_mask)).shape==(3132,)
