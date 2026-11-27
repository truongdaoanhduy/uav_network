"""Independent V101 audit regression: Bayes/DS CPU contract, legal comm actions, DES CLI.

These do not pretend that a pure-Python test validates a GPU Warp kernel.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


def _env(seed: int = 44):
    cfg = load_scenario(ROOT / "research/zhang_xue/configs/scenario_v91_bayes_1km.yaml")
    env = ZhangXue7NodePacketEnv(cfg, seed=seed)
    env.reset(seed=seed)
    return env


def test_cpu_bayes_matches_gpu_analytical_contract_two_positive_measurements():
    env = _env()
    cid = 120
    gcs = env.gcs_id
    prior = env.cfg.bayes_prior_occupancy
    p0, p1, un = env._fused_cell(gcs, cid)
    assert p0 == pytest.approx(prior)
    assert p1 == pytest.approx(1 - prior)
    assert un == pytest.approx(1)

    env.net.ledger_pos[gcs, 0, cid] = 2
    env.net.ledger_gen_ts_s[gcs, 0, cid] = 1
    prob, empty, unknown = env._fused_cell(gcs, cid)
    odds = prior / (1 - prior) * (env.cfg.pd / env.cfg.pf) ** 2
    assert prob == pytest.approx(odds / (1 + odds), abs=1e-6)
    assert prob > env.cfg.confirm_th
    assert empty == pytest.approx(1 - prob)
    assert unknown == pytest.approx(0.25)
    # Two positive reports meet the Bayesian posterior threshold while
    # remaining above the DS-only epistemic-uncertainty ceiling of 0.20.
    assert unknown > env.cfg.ds_uncertainty_ceiling


def test_cpu_local_bayes_map_unknown_cells_preserve_prior():
    env = _env()
    local_map = env._local_belief_features(0).reshape(5, 100)
    assert np.isfinite(local_map).all()
    assert np.any(np.isclose(local_map[0], env.cfg.bayes_prior_occupancy))
    assert np.any(np.isclose(local_map[1], 1 - env.cfg.bayes_prior_occupancy))


def test_cpu_masks_only_routable_gcs_and_keeps_peer_choice_even_with_gcs_route():
    env = _env()
    u, p, gcs = 0, 1, env.gcs_id
    cid = 120
    env.net.ledger_pos[u, u, cid] = 1
    env.net.ledger_gen_ts_s[u, u, cid] = 1
    env.net.routes[u].clear()
    _, _, m = env._build_obs_state_mask()
    assert m[u, 1] == 0, 'unreachable GCS cannot be legal despite dirty evidence'
    assert m[u, 2] == 0
    expire = env.net.sim_time_us + 2_000_000
    env.net.routes[u][gcs] = (gcs, 1, (1 << u) | (1 << gcs), expire)
    env.net.routes[u][p] = (p, 1, (1 << u) | (1 << p), expire)
    _, _, m = env._build_obs_state_mask()
    assert m[u].tolist() == [1, 1, 1], 'PEER is a choice, not prohibited by an available GCS route'


def test_cpu_gcs_bayesian_reward_accepts_two_positive_reports():
    env = _env()
    cid = int(np.flatnonzero(env.true_targets)[0])
    gcs = env.gcs_id
    env.net.ledger_pos[gcs, 0, cid] = 2
    env.net.ledger_gen_ts_s[gcs, 0, cid] = 1
    env.net.gcs_last_novel_rx_us[cid] = env.net.sim_time_us
    env.net.gcs_latest_u_s[cid] = 1
    a = np.zeros((env.actor_n, 2), dtype=np.float64)
    d = np.zeros(env.actor_n, dtype=np.int32)
    _obs, _state, _mask, rew, terminated, truncated, info = env.step(a, d)
    assert env.gcs_tp_latch[cid] == 1
    assert info['gcs_tp_confirmed'] >= 1
    assert rew == pytest.approx(rew)


def test_cli_des_us_changes_both_des_windows(tmp_path):
    out = tmp_path / 'short_des.json'
    p = subprocess.run([
        sys.executable, str(ROOT / 'scripts/gpu/v98_simpy_vs_warp_mac_parity.py'),
        '--des-only', '--nodes', '2', '--des-us', '120000', '--output', str(out),
    ], capture_output=True, text=True, cwd=ROOT, timeout=25)
    assert p.returncode == 0, p.stderr[-1600:]
    result = json.loads(out.read_text())
    assert result['des_duration_us'] == result['hidden_des_duration_us'] == 120000
    assert result['upstream_packet_level_simpy_parity'] == 'NOT_TESTED'
