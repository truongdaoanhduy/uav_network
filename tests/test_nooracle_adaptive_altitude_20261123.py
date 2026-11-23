"""Science-changing opt-in camera quality rule: legacy remains unchanged."""
import numpy as np
import pytest
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.bayesian import sensor_profile, sensor_uniform
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.gpu.bayesian import build_target_cell_lookup
from uav_network.environment.observations import observation_version

CONFIG='configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml'


def candidate_config():
    c=load_config(CONFIG)
    c.runtime.num_envs=1
    c.runtime.device='cpu'
    c.sensor.confirmation_mode='likelihood_ratio'
    c.sensor.confirmation_min_likelihood_ratio=4.0
    c.sensor.confirmation_threshold=.99
    c.world.num_uavs=c.world.num_targets=1
    c.world.num_obstacles=0
    c.network.enabled=False
    c.validate()
    return c


def test_sensor_quality_profile_gives_policy_a_real_altitude_tradeoff():
    c=candidate_config()
    assert observation_version(c)==10
    pd_low,pf_low,r_low=sensor_profile(c.sensor,50)
    pd_mid,pf_mid,r_mid=sensor_profile(c.sensor,75)
    pd_high,pf_high,r_high=sensor_profile(c.sensor,150)
    assert r_high>r_mid>r_low
    assert pd_low>pd_mid>pd_high
    assert pf_low<pf_mid<pf_high
    assert pd_low/pf_low>pd_mid/pf_mid>4>pd_high/pf_high
    assert c.world.altitude_max_m==150


@pytest.mark.parametrize('altitude,expected',[ (25,True),(75,True),(150,False) ])
def test_real_warp_nooracle_confirmation_uses_sensor_evidence_not_hard_50(altitude,expected):
    c=candidate_config()
    seed=next(s for s in range(1,10000) if sensor_uniform(s,1,0,820)<.69)
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[seed])
        wp.to_torch(v.s.positions)[0,0]=torch.tensor([512.5,512.5],dtype=torch.float64)
        wp.to_torch(v.s.targets)[0,0]=torch.tensor([512.5,512.5],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[0,0]=float(altitude)
        wp.to_torch(v.s.belief).fill_(0.5)
        wp.to_torch(v.s.belief)[0,0,820]=.9999
        wp.to_torch(v.s.known).zero_()
        mask=wp.from_torch(torch.ones(1,dtype=torch.int32))
        v._launch(build_target_cell_lookup,1,v.s,v.p,mask,v.target_cell_lookup)
        v.step({'continuous':np.zeros((1,1,4),dtype=np.float32),
                'destination':np.zeros((1,1),dtype=np.int64)})
        assert bool(wp.to_torch(v.s.known)[0,0,0])==expected
    finally:v.close()


def test_default_legacy_keeps_50m_restriction_and_version_8():
    c=load_config(CONFIG)
    assert c.sensor.confirmation_mode=='fine_altitude'
    assert observation_version(c)==8
    assert c.world.altitude_max_m==150 and c.sensor.fine_altitude_m==50


@pytest.mark.parametrize('mode,threshold',[('unknown',4.0),('likelihood_ratio',0),('likelihood_ratio',float('inf'))])
def test_camera_quality_parameters_reject_invalid_science(mode,threshold):
    c=load_config(CONFIG)
    c.sensor.confirmation_mode=mode
    c.sensor.confirmation_min_likelihood_ratio=threshold
    with pytest.raises(ValueError):c.validate()


def test_adaptive_experiment_yaml_is_separate_new_mdp_and_shaping_does_not_force_50m():
    from uav_network.environment.bayesian import confirmation_shaping_altitude
    p='configs/matd3-nooracle-adaptive-altitude-lr4-h256x256-20261123.yaml'
    c=load_config(p)
    assert c.world.altitude_max_m==150 and c.sensor.fine_altitude_m==50
    assert c.sensor.confirmation_threshold==.99
    assert c.sensor.confirmation_mode=='likelihood_ratio'
    assert c.sensor.confirmation_min_likelihood_ratio==4
    assert confirmation_shaping_altitude(c.sensor,150)==pytest.approx(100)
    assert observation_version(c)==10
    original=load_config(CONFIG)
    assert original.sensor.confirmation_mode=='fine_altitude'
    assert observation_version(original)==8


def test_quality_warp_candidate_buffer_reserves_max_fov_and_actor_sees_altitude():
    c=candidate_config()
    v=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=v.reset(seed=[44])
        assert v.p.confirm_capacity>36
        assert v.p.confirmation_shaping_altitude==pytest.approx(100)
        assert v.p.altitude_max==pytest.approx(150)
        # Actor's XYZ extras include own altitude and vertical velocity.
        from uav_network.environment.observations import observation_dim
        assert obs['observation'].shape==(1,1,observation_dim(c))
        first=obs['observation'][0,0].clone()
        wp.to_torch(v.s.altitudes)[0,0]=75.
        from uav_network.environment.gpu.observations import observe
        v._launch(observe,(1,1),v.s,v.p)
        later=wp.to_torch(v.s.observation)[0,0].clone()
        assert not torch.equal(first,later)
    finally:v.close()


def test_adaptive_nooracle_real_radio_delivers_cell_report_not_truth_id():
    c=candidate_config()
    c.network.enabled=True
    c.network.tx_power_min_w=.01
    c.network.tx_power_max_w=.4
    c.network.power_control='learned'
    seed=next(s for s in range(1,10000) if sensor_uniform(s,1,0,820)<.8)
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[seed])
        xy=torch.tensor([512.5,512.5],dtype=torch.float64)
        wp.to_torch(v.s.positions)[0,0]=xy
        wp.to_torch(v.s.gcs)[0]=xy
        wp.to_torch(v.s.targets)[0,0]=xy
        wp.to_torch(v.s.altitudes)[0,0]=75.
        wp.to_torch(v.s.belief).fill_(0.5)
        wp.to_torch(v.s.belief)[0,0,820]=.99999
        mask=wp.from_torch(torch.ones(1,dtype=torch.int32))
        v._launch(build_target_cell_lookup,1,v.s,v.p,mask,v.target_cell_lookup)
        idle={'continuous':np.zeros((1,1,4),np.float32),
              'destination':np.zeros((1,1),np.int64)}
        v.step(idle)
        assert bool(wp.to_torch(v.s.known)[0,0,0])
        assert int(wp.to_torch(v.s.report_cell)[0,0])==820
        tx={'continuous':np.array([[[0,0,0,1]]],np.float32),
            'destination':np.ones((1,1),np.int64)}
        for _ in range(6):
            v.step(tx)
            if bool(wp.to_torch(v.s.delivered)[0,0]):break
        assert bool(wp.to_torch(v.s.delivered)[0,0])
    finally:v.close()


def test_science_changed_policy_does_not_share_observation_version_with_historical():
    from uav_network.algorithms.checkpoints import validate_observation_version
    c=candidate_config()
    payload={'config':c.to_dict(),'observation_version':8}
    with pytest.raises(ValueError,match='observation semantics differ'):
        validate_observation_version(payload)
    payload['observation_version']=9
    with pytest.raises(ValueError,match='observation semantics differ'):
        validate_observation_version(payload)
    payload['observation_version']=10
    validate_observation_version(payload)


def test_actual_cpu_matd3_train_evaluate_checkpoint_adaptive(tmp_path,monkeypatch):
    from uav_network.experiments.training import train
    from uav_network.experiments.evaluation import evaluate
    from pathlib import Path
    torch.set_num_threads(1)
    c=load_config('configs/matd3-nooracle-adaptive-altitude-lr4-h256x256-20261123.yaml')
    c.runtime.device=c.runtime.replay_device='cpu'
    c.runtime.num_envs=4
    c.runtime.episodes=4
    c.runtime.eval_episodes=2
    c.world.max_steps=12
    c.runtime.output_dir=str(tmp_path/'train')
    c.runtime.checkpoint_interval=4
    c.runtime.log_interval_steps=8
    c.runtime.collector_overlap=False
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=False
    c.tracking.online_gpu_eval_enabled=False
    c.tracking.evaluation_interval_steps=0
    c.tracking.evaluation_interval=0
    c.tracking.resumable_checkpoint_interval_seconds=0
    c.matd3.hidden_sizes=[32,32]
    c.matd3.batch_size=16
    c.matd3.replay_capacity=512
    c.matd3.learning_starts=16
    c.matd3.updates_per_step=1
    monkeypatch.setenv('UAV_DIAGNOSTIC_STORE','off')
    monkeypatch.setenv('UAV_FORENSIC_ENABLED','false')
    monkeypatch.setenv('UAV_TRAINING_ONLY','true')
    result=train(c)
    assert result['episodes']==4
    assert result['updates']>=2
    model=Path(result['checkpoint'])
    assert model.is_file()
    saved=torch.load(model,map_location='cpu',weights_only=True)
    assert saved['observation_version']==10
    assert saved['config']['sensor']['confirmation_mode']=='likelihood_ratio'
    evaluation=evaluate(model,episodes=2,seed=9000044,output_dir=tmp_path/'eval')
    assert evaluation['episodes']==2
    assert np.isfinite(evaluation['mean_return'])
    assert (tmp_path/'eval'/'metrics.json').is_file()


def test_quality_rule_requires_declining_pd_rising_pf_and_nooracle():
    c=candidate_config()
    c.sensor.pd=[.7,.9,.8,.7]
    with pytest.raises(ValueError):c.validate()
    c=candidate_config()
    c.sensor.pf=[.01,.2,.1,.3]
    with pytest.raises(ValueError):c.validate()
    c=candidate_config()
    c.sensor.report_cell_hypotheses=False
    with pytest.raises(ValueError):c.validate()


def test_legacy_historical_config_still_validates_without_quality_rule():
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.validate()
    assert c.sensor.confirmation_mode=='fine_altitude'


def test_new_h256x512_adaptive_config_matches_historical_training_dimensions():
    historical=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    adapted=load_config('configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml')
    assert historical.matd3.hidden_sizes==adapted.matd3.hidden_sizes==[256,512]
    assert adapted.runtime.num_envs==4096
    assert adapted.matd3.batch_size==8192
    assert adapted.matd3.updates_per_step==4
    assert adapted.runtime.episodes==20000
    assert adapted.runtime.eval_episodes==1000
    assert adapted.sensor.confirmation_threshold==.99
    assert adapted.world.altitude_max_m==150
    assert adapted.sensor.confirmation_mode=='likelihood_ratio'
    assert observation_version(adapted)==10
    assert observation_version(historical)==8
