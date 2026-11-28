"""V113: evidence-backed CPU safety rollback contract, no RL algorithm edits."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.mobility.flight_safety_energy import trajectory_safety_violations


YAML = Path(__file__).resolve().parents[1] / 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'


def test_v113_infeasible_braking_is_terminal_without_writing_unsafe_position():
    cfg = load_scenario(YAML)
    env = ZhangXue7NodePacketEnv(cfg, seed=44)
    env.reset(seed=44)
    # At 985 m, initial position is legal, but 15 m/s with max -5 m/s2
    # cannot stop short of the physical boundary. V112 committed x=995.
    env.positions[0, :2] = [985.0, 500.0]
    env.velocities[0, :] = [15.0, 0.0]
    env.net.update_positions(env.positions, env.velocities)
    initial_positions = env.positions.copy()
    initial_velocities = env.velocities.copy()
    initial_battery = env.battery_j.copy()
    initial_counts = (env.net.ledger_pos.copy(), env.net.ledger_neg.copy())
    assert not any(trajectory_safety_violations(
        env.positions[:6,:2], np.zeros((6,2)), np.ones(6,dtype=bool), env.buildings,
        map_size_m=cfg.map_size_m, dt_s=cfg.dt_s,
        uav_safe_distance_m=cfg.uav_safe_distance_m,
    ).values())
    velocity = np.zeros((6,2),dtype=np.float64)
    velocity[0] = [15.0, 0.0]
    obs, state, mask, reward, terminated, truncated, info = env.step(
        velocity, np.zeros(6,dtype=np.int64),
    )
    assert terminated and not truncated
    assert info['safety_violation']
    assert info['safety_violations']['boundary'] == [0]
    np.testing.assert_array_equal(env.positions, initial_positions)
    np.testing.assert_array_equal(env.velocities, initial_velocities)
    np.testing.assert_array_equal(env.battery_j, initial_battery)
    np.testing.assert_array_equal(env.net.ledger_pos, initial_counts[0])
    np.testing.assert_array_equal(env.net.ledger_neg, initial_counts[1])
    assert np.isfinite(obs).all() and np.isfinite(state).all()
    assert reward <= -cfg.safety_failure_penalty + 1e-6


def test_v113_safe_move_still_integrates_and_consumes_battery():
    cfg=load_scenario(YAML)
    env=ZhangXue7NodePacketEnv(cfg,seed=47)
    env.reset(seed=47)
    before=env.positions.copy()
    before_battery=env.battery_j.copy()
    velocity=np.zeros((6,2),dtype=np.float64)
    velocity[0,:]=[4.0,0.0]
    *_,terminated,truncated,info=env.step(velocity,np.zeros(6,dtype=np.int64))
    assert not info['safety_violation']
    assert not terminated and not truncated
    assert float(env.positions[0,0]) > float(before[0,0])
    assert np.all(env.battery_j < before_battery)
