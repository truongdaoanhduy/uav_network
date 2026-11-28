"""V124: RED/GREEN DS temporal sample counting and CRN; no learned actions."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.sensing.keyed_rng import sensor_uniform

SCENARIO = 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'


def _counts(env):
    idx = np.arange(env.actor_n)
    return int(np.sum(env.net.ledger_pos[idx, idx, :], dtype=np.int64) +
               np.sum(env.net.ledger_neg[idx, idx, :], dtype=np.int64))


def test_cpu_cooldown_does_not_count_suppressed_repeated_samples():
    cfg = replace(load_scenario(SCENARIO), sensor_revisit_cooldown_steps=3)
    env = ZhangXue7NodePacketEnv(cfg, seed=44)
    env.reset(seed=44)
    act = np.zeros((env.actor_n, 2), dtype=np.float64)
    comm = np.ones(env.actor_n, dtype=np.int32)
    totals=[]
    for _ in range(4):
        env.step(act, comm)
        totals.append(_counts(env))
    assert totals[0] > 0
    assert totals[1] == totals[0] and totals[2] == totals[0]
    assert totals[3] > totals[2]
    assert env.cfg.belief_inference_mode == 'dempster_shafer'


def test_canonical_cooldown_zero_still_adds_each_visible_tick():
    env = ZhangXue7NodePacketEnv(load_scenario(SCENARIO), seed=44)
    env.reset(seed=44)
    act = np.zeros((env.actor_n, 2))
    comm = np.ones(env.actor_n, dtype=np.int32)
    vals=[]
    for _ in range(4):
        env.step(act, comm)
        vals.append(_counts(env))
    assert vals[0] > 0 and vals == sorted(vals)
    assert all(b > a for a,b in zip(vals,vals[1:]))


@pytest.mark.parametrize('bad',[-1, 1.5, True, 901])
def test_invalid_cooldown_rejected(bad):
    with pytest.raises(ValueError, match='sensor_revisit_cooldown_steps'):
        replace(load_scenario(SCENARIO), sensor_revisit_cooldown_steps=bad)


def test_counter_keyed_rng_independent_of_fov_and_order():
    observations = [(0, 31, 11), (0, 31, 12), (1, 31, 11), (0, 32, 11)]
    a = {i: sensor_uniform(44,1,t,u,c) for u,t,c in observations for i in [(u,t,c)]}
    b = {i: sensor_uniform(44,1,t,u,c) for u,t,c in reversed(observations) for i in [(u,t,c)]}
    assert a == b
    assert len(set(a.values())) == len(a)
    for i in a:
        assert 0.0 <= a[i] < 1.0


def test_invalid_counter_rng_mode_rejected():
    with pytest.raises(ValueError, match='sensor_rng_mode'):
        replace(load_scenario(SCENARIO),sensor_rng_mode='legacy_bayes')


def test_gpu_parameter_wiring_and_keyed_gpu_fail_closed_without_warp_import():
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params
    cfg = replace(load_scenario(SCENARIO), sensor_revisit_cooldown_steps=9)
    p = GPUWorldV80Params.from_scenario(cfg, envs=1, device='cpu')
    assert p.sensor_revisit_cooldown_steps == 9
    with pytest.raises(ValueError,match='keyed RNG'):
        GPUWorldV80Params.from_scenario(replace(cfg,sensor_rng_mode='counter_keyed'),envs=1,device='cpu')
