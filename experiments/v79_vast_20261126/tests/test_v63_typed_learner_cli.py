"""Typed config, explicit versioned choices and RF pipeline (red-green)."""
import pytest
import numpy as np

from uav_network.algorithms.configuration import LearnerConfig


def test_algorithm_yaml_override_and_strict_types(tmp_path):
    from uav_network.algorithms.configuration import load_learner_config
    file = tmp_path/'train.yaml'
    file.write_text('runtime:\n  algorithm: masac\nmasac:\n  batch_size: 64\n  hidden_sizes: [64, 128]\nnetwork:\n  power_control: learned\n')
    cfg=load_learner_config(file,overrides=['masac.gamma=0.97','runtime.amp_mode=off'])
    assert cfg.runtime.algorithm=='masac'
    assert cfg.network.power_control=='learned'
    assert cfg.masac.hidden_sizes==[64,128]
    assert cfg.masac.batch_size==64
    assert cfg.masac.gamma==0.97


@pytest.mark.parametrize('override',[
    'matd3.batch_size=true', 'matd3.gamma=1.5','matd3.hidden_sizes=[256, true]',
    'runtime.algorithm=ddpg', 'runtime.unknown=100', 'network.power_control=magical',
    'matd3.learning_rate=-0.001', 'masac.initial_alpha=-1', 'world.num_uavs=31',
    'matd3.share_actor_parameters=1', 'matd3.hidden_sizes=[]',
])
def test_invalid_algorithm_settings_rejected(override):
    from uav_network.algorithms.configuration import load_learner_config
    with pytest.raises(ValueError):
        load_learner_config(overrides=[override])


def test_tx_power_actor_wiring_into_environment_and_radio():
    from zhang_xue.environment.configuration import make_v62_config
    from zhang_xue.environment.core import ZhangXue7NodePacketEnv
    from zhang_xue.learners.actor_adapter import step_from_policy_action
    env=ZhangXue7NodePacketEnv(make_v62_config(),seed=44)
    _,_,mask=env.reset(seed=44)
    class Action:
        continuous=np.zeros((6,3))
        destination=np.zeros((6,),dtype=np.int64)
    action=Action()
    action.continuous[0,2]=-1.0
    action.continuous[1,2]=1.0
    result=step_from_policy_action(env,action,mask,power_control='learned')
    assert len(result)==7
    assert env.net.tx_power_per_radio[0]==pytest.approx(env.cfg.tx_power_min_w)
    assert env.net.tx_power_per_radio[1]==pytest.approx(env.cfg.tx_power_max_w)


def test_unimplemented_gpu_layout_reports_active_not_planned():
    from zhang_xue.gpu.layout_v62 import backing_bytes_per_environment_v62
    from zhang_xue.gpu.runtime import WarpSevenRadioNetwork
    assert WarpSevenRadioNetwork.active_soa_bytes_per_environment() != backing_bytes_per_environment_v62()


def test_tx_feature_configuration_mismatch_fails_closed():
    from zhang_xue.environment.configuration import make_v57_config, make_v62_config
    from zhang_xue.environment.core import ZhangXue7NodePacketEnv
    from zhang_xue.learners.actor_adapter import step_from_policy_action
    class Action:
        continuous=np.zeros((6,3))
        destination=np.zeros(6,dtype=np.int64)
    fixed=ZhangXue7NodePacketEnv(make_v57_config(),seed=44)
    _,_,mask=fixed.reset(seed=44)
    with pytest.raises(ValueError,match='power_control'):
        step_from_policy_action(fixed, Action(), mask, power_control='learned')
    learned=ZhangXue7NodePacketEnv(make_v62_config(),seed=44)
    _,_,mask=learned.reset(seed=44)
    with pytest.raises(ValueError,match='power_control'):
        step_from_policy_action(learned, Action(), mask, power_control='fixed')
    with pytest.raises(ValueError,match='power'):
        learned.step(np.zeros((6,2)),np.zeros(6,dtype=np.int64))
    with pytest.raises(ValueError,match='power'):
        fixed.step(np.zeros((6,2)),np.zeros(6,dtype=np.int64),tx_power_w=np.ones(6))


def test_real_cpu_learner_gate_uses_learned_tx_control_when_configured():
    import importlib.util
    from pathlib import Path
    from zhang_xue.environment.configuration import make_v62_config
    from uav_network.algorithms.configuration import LearnerConfig
    path=(Path(__file__).resolve().parents[1]/'research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py')
    spec=importlib.util.spec_from_file_location('v63_cpu_gate',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=LearnerConfig()
    config.network.power_control='learned'
    config.matd3.hidden_sizes=[32,32]
    config.masac.hidden_sizes=[32,32]
    report=module.integration_gate(
        'matd3',seed=44,transitions=2,
        learner_config=config,scenario_config=make_v62_config(),
    )
    assert report['power_control']=='learned'
    assert 0.05 <= report['mean_actor_tx_power_w'] <= 0.5
    assert report['finite_report']
