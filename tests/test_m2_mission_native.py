"""New opt-in 3D mission: actual sensing + actual multi-hop packet delivery."""
import numpy as np
import pytest

from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.environment.m2_mission import CooperativeM2Env


def cfg():
    c=Config();c.world.motion_dimensions=3;c.world.motion_safety="cbf"
    c.world.map_size_m=200.;c.world.num_uavs=2;c.world.num_targets=1
    c.world.num_obstacles=1;c.world.max_steps=25
    c.world.launch_radius_m=90.;c.world.target_exclusion_radius_m=15.
    c.world.obstacle_radius_min_m=5.;c.world.obstacle_radius_max_m=10.
    c.world.obstacle_height_min_m=10.;c.world.obstacle_height_max_m=60.
    c.sensor.model="bayesian";c.sensor.grid_cell_m=25.
    c.network.peer_range_m=c.network.gcs_range_m=200.
    c.network.report_ttl_s=35.;c.energy.enabled=False
    return c


def scene():
    return {"positions":np.array([[80.,90.,100.],[120.,90.,100.]]),
            "targets":np.array([[85.,130.]]),
            "obstacles":np.array([[150.,155.,7.,40.]])}


def test_m2_native_episode_has_autonomous_z_no_oracle_critic_and_gcs_packet_map():
    env=CooperativeM2Env(cfg(),correlation_strength=.5,max_scan_cells=4)
    try:
        obs,info=env.reset(seed=44,options=scene())
        assert obs.shape[0]==2
        assert env.state().size==obs.size  # observer-only critic
        assert env.gcs_map.coverage.sum()==0
        initial_altitudes=env.positions[:,2].copy()
        hold=Action(np.zeros((2,env.config.continuous_dim),dtype=np.float32),
                    np.zeros(2,dtype=np.int64))
        env.step(hold)
        np.testing.assert_array_equal(env.positions[:,2],initial_altitudes)
        assert env.gcs_map.coverage.sum()==0
        transmit=Action(np.zeros((2,env.config.continuous_dim),dtype=np.float32),
                        np.array([2,0],dtype=np.int64))
        env.step(transmit)
        assert env.gcs_map.coverage.sum()>0
        assert env.gcs_map.target_masses.shape[1]==3
        assert env.gcs_map.obstacle_masses.shape[1]==3
        assert 0<=env.info()["gcs_map"]["target_brier"]<=1
        # XYZ action still selects altitude. No hidden verification descent.
        move=Action(np.zeros((2,env.config.continuous_dim),dtype=np.float32),
                    np.zeros(2,dtype=np.int64))
        move.continuous[0,2]=-1
        env.step(move)
        assert env.positions[0,2]<initial_altitudes[0]
    finally:env.close()


def test_m2_rejects_multiple_targets_in_same_grid_cell():
    c=cfg();c.world.num_targets=2;c.world.num_obstacles=0
    options={"positions":scene()["positions"],
             "targets":np.array([[80.,130.],[81.,131.]]),"obstacles":np.empty((0,4))}
    env=CooperativeM2Env(c)
    try:
        with pytest.raises(ValueError,match="same cell"):
            env.reset(seed=44,options=options)
    finally:env.close()


def test_map_evaluation_has_reconstruction_while_critic_cannot_read_truth():
    env=CooperativeM2Env(cfg(),correlation_strength=.5,max_scan_cells=4)
    try:
        obs, info=env.reset(seed=33,options=scene())
        assert "gcs_map" in info
        assert "target_brier" in info["gcs_map"]
        assert not hasattr(env.nodes[0],"targets")
        assert not hasattr(env.nodes[0],"target_truth")
        assert env._truth_target.sum()==1
        assert "true_target_id" not in repr(env.nodes[0].visible_summary(0))
    finally:env.close()


def test_delivery_reward_is_not_paid_twice_for_same_ground_truth_cell():
    from uav_network.communication.m2_protocol import DecisionNotice
    e=CooperativeM2Env(cfg(),correlation_strength=.5,max_scan_cells=1)
    try:
        e.reset(seed=44,options=scene())
        cell=int(np.flatnonzero(e._truth_target)[0])
        e.gcs_map.ingest_notice(DecisionNotice("a:1",0,cell,1,"verified",1),arrived_at=1)
        e.gcs_map.ingest_notice(DecisionNotice("b:1",1,cell,1,"verified",1),arrived_at=1)
        first=e._reward(np.zeros(2),0)
        assert e.last_reward_components["report"]==50
        e.gcs_map.ingest_notice(DecisionNotice("a:2",0,cell,2,"retracted",2),arrived_at=2)
        e.gcs_map.ingest_notice(DecisionNotice("a:3",0,cell,3,"verified",3),arrived_at=3)
        second=e._reward(np.zeros(2),0)
        assert e.last_reward_components["report"]==0
    finally:e.close()


@pytest.mark.parametrize("algorithm",["matd3","masac"])
def test_new_m2_observation_respects_legacy_actor_mask_and_active_positions(algorithm):
    import torch
    from uav_network.algorithms.action_utils import valid_destinations
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.masac import MASAC
    c=cfg();c.matd3.hidden_sizes=[32,32];c.masac.hidden_sizes=[32,32]
    e=CooperativeM2Env(c,correlation_strength=.5,max_scan_cells=2)
    try:
        obs,_=e.reset(seed=44,options=scene())
        assert np.all(obs[:,9]==1), "the policy assumes index 9 = active UAV flag"
        masks=valid_destinations(torch.from_numpy(obs),3)
        assert masks.shape==(2,3)
        assert masks[:,0].all()
        assert masks[:,2].all(),"both UAVs can reach GCS in this scenario"
        model=(MATD3 if algorithm=="matd3" else MASAC)(c,obs.shape[-1],e.state().size,device="cpu")
        action=model.act(obs,deterministic=True)
        assert action.continuous.shape==(2,c.continuous_dim)
        assert action.destination.shape==(2,)
        for i in range(2):assert masks[i,action.destination[i]]
        e.step(action)
    finally:e.close()


@pytest.mark.parametrize("algorithm",["matd3","masac"])
def test_real_m2_rollout_replay_and_actor_critic_update_cpu_smoke(algorithm):
    """One real neural gradient step, not evidence of convergence or quality."""
    import torch
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.masac import MASAC
    from uav_network.algorithms.replay import ReplayBuffer
    c=cfg();c.matd3.hidden_sizes=[32,32];c.masac.hidden_sizes=[32,32]
    e=CooperativeM2Env(c,correlation_strength=.55,max_scan_cells=2)
    try:
        obs,_=e.reset(seed=44,options=scene())
        model=(MATD3 if algorithm=="matd3" else MASAC)(c,obs.shape[-1],e.state().size,device="cpu")
        replay=ReplayBuffer(32,obs.shape,e.state().size,continuous_dim=c.continuous_dim)
        initial=torch.cat([p.detach().flatten() for p in model.actors.parameters()]).clone()
        for _ in range(6):
            state=e.state()
            action=model.act(obs,deterministic=False)
            next_obs,reward,done,_,_=e.step(action)
            replay.add(obs,state,action,reward,next_obs,e.state(),done)
            obs=next_obs
            if done:break
        assert len(replay)>=4
        for _ in range(2):
            model.update(replay.sample(4),collect_metrics=False)
        assert model.updates==2
        assert model.actor_updates>0
        updated=torch.cat([p.detach().flatten() for p in model.actors.parameters()])
        assert torch.isfinite(updated).all()
        assert not torch.allclose(initial,updated,rtol=0,atol=0)
    finally:e.close()
