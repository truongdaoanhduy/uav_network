"""Regression gates for the combined Q1-Clean and structured algorithm package."""
import math

import numpy as np
import pytest

from uav_network.algorithms.configuration import LearnerConfig
from zhang_xue.environment.configuration import (
    make_v63_q1_config,
    make_uavnetsim_reference_config,
)
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import encode_actor_input_scalable, make_critic_state_scalable


def test_upstream_80211b_phy_reference_not_mislabeled_as_minus_70():
    cfg = make_uavnetsim_reference_config()
    expected_noise_dbm = -174.0 + 10.0 * math.log10(22e6) + 7.0
    assert cfg.tx_power_w == pytest.approx(0.1)
    assert cfg.mac_cca_threshold_dbm == -82.0
    assert 10 * math.log10(cfg.mac_noise_power_w / 1e-3) == pytest.approx(expected_noise_dbm)


def test_clean_learners_have_two_motors_and_three_comm_actions():
    cfg = LearnerConfig()
    cfg.continuous_dim = 2
    cfg.world.num_uavs = 8
    cfg.network.destination_mode = "compact_3"
    cfg.network.power_control = "fixed"
    cfg.masac.target_continuous_entropy = -2.0
    cfg.validate()
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.masac import MASAC
    for name, model_cls in [("matd3", MATD3), ("masac", MASAC)]:
        cfg.runtime.algorithm = name
        cfg.matd3.hidden_sizes = [16, 16]
        cfg.masac.hidden_sizes = [16, 16]
        model = model_cls(cfg, 518, 8 * 518, device="cpu")
        assert model.destinations == 3
        assert model.continuous_dim == 2
        obs = np.zeros((8, 518), np.float32)
        obs[:, 9] = 1
        obs[:, -3:] = [1, 0, 0]
        act = model.act(obs, deterministic=True)
        assert act.continuous.shape == (8, 2)
        assert np.max(act.destination) < 3


def test_scalable_8_uav_network_substeps_2_and_no_dead_tx_head():
    cfg = make_v63_q1_config(num_uavs=8, network_substeps=2)
    env = ZhangXue7NodePacketEnv(cfg, seed=32)
    obs, _, mask = env.reset(seed=32)
    encoded = encode_actor_input_scalable(obs, mask)
    state = make_critic_state_scalable(encoded)
    assert encoded.shape == (8, 518)
    assert state.shape == (8 * 518,)
    obs, _, _, rew, terminated, truncated, info = env.step(np.zeros((8, 2)), np.zeros(8, np.int64))
    assert np.isfinite(rew) and obs.shape == (8, 514)
    assert not terminated and not truncated

@pytest.mark.parametrize('algorithm', ['matd3', 'masac'])
def test_real_cpu_learning_update_with_q1_clean_two_motor_compact_action(algorithm):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / 'research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py'
    spec = importlib.util.spec_from_file_location('v64_gate', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    learner = LearnerConfig()
    learner.world.num_uavs = 4
    learner.continuous_dim = 2
    learner.network.destination_mode = 'compact_3'
    learner.network.power_control = 'fixed'
    learner.matd3.hidden_sizes = [16, 16]
    learner.masac.hidden_sizes = [16, 16]
    learner.matd3.batch_size = 2
    learner.masac.batch_size = 2
    learner.masac.target_continuous_entropy = -2.0
    output = module.integration_gate(
        algorithm, seed=44, transitions=2,
        learner_config=learner, scenario_config=make_v63_q1_config(num_uavs=4),
    )
    assert output['gradient_update_count'] >= 1
    assert output['continuous_dim'] == 2
    assert output['destinations'] == 3
    assert output['actor_obs_dim'] == 518
    assert output['joint_critic_dim'] == 4 * 518
    assert output['mean_actor_tx_power_w'] == pytest.approx(0.1)
    assert output['finite_report']


def test_motor_y_not_zeroed_by_fixed_power_legacy_canonicalizer():
    import torch
    from uav_network.algorithms.core.action_utils import canonical_continuous, joint_action

    xy = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
    obs = torch.zeros((1, 518), dtype=torch.float32)
    obs[0, 9] = 1.0
    dst = torch.tensor([0])
    actual = canonical_continuous(xy, obs, dst, learn_power=False)
    assert torch.allclose(actual, xy)
    assert torch.allclose(joint_action(xy, dst, destinations=3, learn_power=False)[:, :2], xy)


def test_default_setup_cpu_gate_profiles_match_q1_fixed_power():
    from pathlib import Path
    from uav_network.algorithms.configuration import load_learner_config
    from zhang_xue.configuration.loader import load_scenario
    root = Path(__file__).resolve().parents[1]
    scenario = load_scenario(root / "research/zhang_xue/configs/scenario_7radio.yaml")
    for algorithm in ("matd3", "masac"):
        cfg = load_learner_config(root / f"configs/learners/{algorithm}_cpu_gate.yaml")
        assert cfg.continuous_dim == 2
        assert cfg.network.power_control == "fixed"
        assert cfg.network.destination_mode == "compact_3"
        assert cfg.world.num_uavs == scenario.num_uavs
        assert scenario.enable_tx_power_control is False
        assert scenario.comm_action_mode == cfg.network.destination_mode
        assert scenario.tx_power_w == 0.1
    setup = (root / "setup.sh").read_text()
    assert "research/zhang_xue/configs/scenario_7radio.yaml" in setup


def test_no_peer_dissemination_ablation_masks_and_blocks_the_action():
    cfg = make_v63_q1_config(enable_peer_dissemination=False, w_p2p=0.0)
    env = ZhangXue7NodePacketEnv(cfg, seed=51)
    _, _, mask = env.reset(seed=51)
    assert np.all(mask[:, 2] == 0)
    assert all(env._resolve_comm_dest(u, 2) == -1 for u in range(cfg.num_uavs))


def test_upstream_phy_sanity_los_one_kilometer_not_fake_249m_range():
    """Only isolated LoS: real experiment includes buildings/blackout and CSMA."""
    from zhang_xue.communication.common import link_path_gain
    cfg = make_uavnetsim_reference_config()
    rx = cfg.tx_power_w * link_path_gain((0, 0, 50), (1000, 0, 50), [], [])
    cca_w = 1e-3 * (10.0 ** (cfg.mac_cca_threshold_dbm / 10.0))
    assert rx > cca_w
    assert rx / cfg.mac_noise_power_w > (10.0 ** (cfg.mac_snr_threshold_db / 10.0))
    # At upstream 802.11b noise, -82 dBm CCA is HIGHER than the thermal noise floor.
    assert cca_w > cfg.mac_noise_power_w


def test_v64_q1_gpu_preflight_fails_closed_for_legacy_seven_radio_runtime():
    from scripts.maintenance.q1_gpu_preflight import q1_gpu_readiness
    status = q1_gpu_readiness()
    assert status['ready'] is False
    assert 'v61_uncompressed' in status['active_layout']
    assert 'hardcoded_6_uavs' in status['blockers']


def test_standalone_cpu_smoke_cli_respects_compact3_destination_and_num_uavs():
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, 'research/zhang_xue/scripts/run_scenario.py',
        '--config', 'research/zhang_xue/configs/scenario_7radio.yaml',
        '--set', 'num_uavs=8', '--steps', '1', '--communication', 'gcs',
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    data = json.loads(result.stdout)
    assert data['steps_executed'] == 1
    assert data['config']['num_uavs'] == 8
