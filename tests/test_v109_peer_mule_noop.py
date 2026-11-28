"""V109: inactive data-mule must not modify an actor's state when PEER is off.

This is a paired-world scientific contract, not merely reward equality. An
inactive action cannot silently scale the acknowledgement feature of the actor.
"""
from pathlib import Path
import numpy as np

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params


STRESS = Path(__file__).resolve().parents[1] / 'research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml'


def _run(mule: bool):
    cfg = load_scenario(STRESS, (
        'enable_peer_dissemination=false',
        f'enable_data_mule_forwarding={str(mule).lower()}',
    ))
    env = ZhangXue7NodePacketEnv(cfg, seed=44)
    obs, _, mask = env.reset(seed=44)
    for _ in range(25):
        vel = np.tile([[0., 10.]], (6, 1))
        obs, _, mask, _, _, _, _ = env.step(vel, mask[:, 1].astype(np.int64))
    return cfg, env, obs


def test_peer_off_mule_flag_is_true_noop_for_nonoracle_actor():
    cfg_a, a, obs_a = _run(False)
    cfg_b, b, obs_b = _run(True)
    assert cfg_b.enable_data_mule_forwarding is True  # preserve requested ablation
    assert a.net.stats['e2e_ack_rx'] > 0  # meaningful workload, not empty equality
    assert a.net.stats['e2e_ack_rx'] == b.net.stats['e2e_ack_rx']
    assert a.net.stats['data_delivered_gcs'] == b.net.stats['data_delivered_gcs']
    np.testing.assert_array_equal(obs_a, obs_b)
    assert a.effective_data_mule_forwarding is False
    assert b.effective_data_mule_forwarding is False


def test_gpu_params_use_effective_not_requested_mule_flag():
    for mule in (False, True):
        cfg = load_scenario(STRESS, (
            'enable_peer_dissemination=false',
            f'enable_data_mule_forwarding={str(mule).lower()}',
        ))
        params = GPUWorldV79Params.from_scenario(cfg, device='cpu')
        assert params.enable_peer is False
        assert params.enable_data_mule is False
