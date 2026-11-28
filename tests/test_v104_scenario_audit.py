"""CPU DS lifecycle regression and active gate defaults (not CUDA proof)."""
from pathlib import Path
import numpy as np
import pytest
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('unlatch', [False, True])
def test_active_precision_and_one_time_reward(unlatch):
    cfg = load_scenario(ROOT / 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml')
    cfg.enable_tp_unlatch = unlatch
    env = ZhangXue7NodePacketEnv(cfg, seed=44)
    env.reset(seed=44)
    cid = int(np.flatnonzero(env.true_targets)[0])
    g = env.gcs_id
    env.net.ledger_gen_ts_s[g, 0, cid] = 1
    env.net.gcs_last_novel_rx_us[cid] = env.net.sim_time_us
    env.net.gcs_latest_u_s[cid] = 1
    a = np.zeros((6, 2), dtype=np.float32)
    d = np.zeros(6, dtype=np.int32)
    env.net.ledger_pos[g, 0, cid] = 2
    first = env.step(a, d)[-1]
    assert first['r_gcs_tp'] == cfg.r_tp
    assert first['recall'] == pytest.approx(1/15, abs=0.0001)
    env.net.ledger_neg[g, 0, cid] = 3
    second = env.step(a, d)[-1]
    assert second['gcs_tp_confirmed'] == 1
    assert second['gcs_tp_active_confirmed'] == 0
    assert second['recall'] == 0
    env.net.ledger_pos[g, 0, cid] = 6
    third = env.step(a, d)[-1]
    assert third['gcs_tp_active_confirmed'] == 1
    assert third['r_gcs_tp'] == 0

@pytest.mark.parametrize('name', ['v99_endurance_sticky_cuda_gate', 'v98_warp_4096x900_endurance_gate', 'v97_warp_scale_smoke'])
def test_gate_default_ds(name):
    src = (ROOT / 'scripts/gpu' / (name + '.py')).read_text()
    assert 'scenario_v103_ds_only_1km.yaml' in src
    assert 'scenario_v91_bayes_1km.yaml' not in src
