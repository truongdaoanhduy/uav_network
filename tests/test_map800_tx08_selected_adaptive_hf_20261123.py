"""Selected MAP800 TX0.8W learned-power config with LR4 and deep forensic logging."""
import os
import pytest

from uav_network.configuration import load_config
from uav_network.communication.link_budget import report_reference_tx_energy_j, power_from_action
from uav_network.environment.gpu.state import params_from_config
from scripts.launch_nooracle_map800_tx08_training_with_hf_forensics_20261123 import (
    CONFIG,forensic_environment,validate_production,training_command)

BASE='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml'


def test_tx08_is_learned_and_lr4_is_not_broken():
    c=load_config(CONFIG)
    assert c.network.power_control=='learned'
    assert c.network.tx_power_min_w==.001 and c.network.tx_power_max_w==.8
    assert c.network.tx_power_mapping=='log'
    assert c.sensor.confirmation_mode=='likelihood_ratio'
    assert c.sensor.confirmation_threshold==.99
    assert c.sensor.confirmation_min_likelihood_ratio==4
    assert c.world.altitude_max_m==150 and c.world.map_size_m==800
    assert c.sensor.grid_cell_m==20
    assert c.matd3.hidden_sizes==[256,256]
    assert c.matd3.initialization=='orthogonal'
    assert c.runtime.compile_models is True
    assert c.runtime.num_envs==4096 and c.matd3.batch_size==8192
    assert c.matd3.updates_per_step==4
    assert c.tracking.run_name.startswith('MATD3-NOORACLE-LR4-MAP800-H256x256-TX08-')
    assert c.tracking.checkpoint_repo=='duy95/uav-search-checkpoints'
    assert c.reward.communication_reference_tx_power_w==.4
    assert c.reward.communication_efficiency==1
    assert c.runtime.episodes==20_000


def test_reference_penalty_same_energy_independent_of_hardware_cap():
    c=load_config(CONFIG)
    base=load_config(BASE)
    old=report_reference_tx_energy_j(base.network)
    new=report_reference_tx_energy_j(c.network,reference_power_w=c.reward.communication_reference_tx_power_w)
    assert new==pytest.approx(old)
    assert params_from_config(c).report_reference_tx_energy==pytest.approx(old)
    assert report_reference_tx_energy_j(c.network)>old
    assert power_from_action(-1,c.network)==pytest.approx(.001)
    assert power_from_action(1,c.network)==pytest.approx(.8)
    assert .02 < power_from_action(0,c.network) < .04


def test_launcher_keeps_hf_forensics_and_prevents_cpu_or_overwrites(tmp_path):
    c=load_config(CONFIG)
    env=forensic_environment({'PATH':'/usr/bin'})
    assert env['UAV_DIAGNOSTIC_STORE']=='huggingface_public'
    assert env['UAV_DIAGNOSTIC_HF_REPO']=='duy95/uav-search-diagnostics'
    assert env['UAV_HF_VERIFY_PUBLIC_READBACK']=='true'
    assert env['UAV_ROOT_CAUSE_AUDIT']=='true'
    assert env['UAV_FORENSIC_WORLD_IDS']=='0,1'
    assert int(env['UAV_FORENSIC_STRIDE'])<=16
    assert env['UAV_FORENSIC_FULL_BELIEF']=='true'
    assert env['UAV_WANDB_PLOTS_ONLY']=='false'
    assert env['UAV_TRAINING_ONLY']=='true'
    assert 'WANDB_API_KEY' not in env and 'HF_TOKEN' not in env
    assert training_command()[-2:]==['--config',CONFIG]
    with pytest.raises(RuntimeError,match='two CUDA GPUs'):
        validate_production(c,cuda_count=1)
    c.runtime.output_dir=str(tmp_path/'already_exists')
    (tmp_path/'already_exists').mkdir()
    with pytest.raises(FileExistsError):
        validate_production(c,cuda_count=2)
    c.runtime.output_dir=str(tmp_path/'new')
    validate_production(c,cuda_count=2)
