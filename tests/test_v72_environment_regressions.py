"""Executable regressions for decentralized actions, aged belief and flight setup."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zhang_xue.communication.common import PKT_DATA, Packet, pack_entry_2x_uint32, unpack_entry_2x_uint32
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import step_from_policy_action
from zhang_xue.mobility.flight_safety_energy import trajectory_safety_violations


def _env(**overrides):
    cfg = EnvConfigV54(num_targets=1, **overrides)
    env = ZhangXue7NodePacketEnv(cfg, seed=31)
    env.reset(seed=31)
    return env


def test_default_compact_peer_action_uses_only_sender_routes_and_evidence():
    # Hidden peer position/battery must not change an actor's executable action.
    env = _env(num_uavs=3, comm_action_mode="compact_3")
    env.net.routes[0].clear()
    env.net.record_local_observation(0, 100, True, 1)
    assert env._resolve_comm_dest(0, 2) == -1
    assert env._build_obs_state_mask()[2][0, 2] == 0
    env.net.routes[0].update({1: (1, 2, 3, 10_000_000), 2: (2, 1, 5, 10_000_000)})
    before_obs, _, before_mask = env._build_obs_state_mask()
    assert env._resolve_comm_dest(0, 2) == 2
    env.positions[1, :2] = env.positions[0, :2] + 1.0
    env.positions[2, :2] = [900.0, 900.0]
    env.battery_j[1:] = 0.0
    after_obs, _, after_mask = env._build_obs_state_mask()
    np.testing.assert_array_equal(before_obs[0], after_obs[0])
    np.testing.assert_array_equal(before_mask[0], after_mask[0])
    assert env._resolve_comm_dest(0, 2) == 2


def test_explicit_legacy_nearest_peer_experiment_remains_selectable():
    env = _env(num_uavs=3, comm_action_mode="compact_3", compact3_decentralized_peer_select=False)
    env.net.routes[0].clear()
    env.net.record_local_observation(0, 100, True, 1)
    env.positions[1, :2] = env.positions[0, :2] + 1.0
    env.positions[2, :2] = [900.0, 900.0]
    assert env._resolve_comm_dest(0, 2) == 1
    env.battery_j[1] = 0.0
    assert env._resolve_comm_dest(0, 2) == 2


@pytest.mark.parametrize("expiry_delta_us", [-1, 0])
def test_compact_peer_skips_expired_short_route_for_live_higher_hop(expiry_delta_us):
    env = _env(num_uavs=3, comm_action_mode="compact_3")
    now = env.net.sim_time_us
    env.net.routes[0] = {
        1: (1, 1, 3, now + expiry_delta_us),
        2: (2, 2, 5, now + 1_000_000),
    }
    env.net.record_local_observation(0, 100, True, 1)
    before = env.net.routes[0].copy()
    assert env._resolve_comm_dest(0, 2) == 2
    assert env._build_obs_state_mask()[2][0, 2] == 1
    assert env.net.routes[0] == before


@pytest.mark.parametrize("hops", [-1, 0])
def test_compact_peer_skips_nonpositive_hop_route(hops):
    env = _env(num_uavs=3, comm_action_mode="compact_3")
    now = env.net.sim_time_us
    env.net.routes[0] = {1: (1, hops, 3, now + 1_000_000),
                         2: (2, 2, 5, now + 1_000_000)}
    env.net.record_local_observation(0, 100, True, 1)
    assert env._resolve_comm_dest(0, 2) == 2


@pytest.mark.parametrize("expiry_delta_us,hops", [(-1, 1), (0, 1), (1_000_000, 0)])
def test_only_expired_or_invalid_peer_does_not_enable_share_mask(expiry_delta_us, hops):
    env = _env(num_uavs=3, comm_action_mode="compact_3")
    now = env.net.sim_time_us
    env.net.routes[0] = {1: (1, hops, 3, now + expiry_delta_us)}
    env.net.record_local_observation(0, 100, True, 1)
    before = env.net.routes[0].copy()
    assert env._build_obs_state_mask()[2][0, 2] == 0
    assert env._resolve_comm_dest(0, 2) == -1
    assert env.net.routes[0] == before


def test_route_telemetry_counts_only_live_positive_hop_routes_without_pruning():
    env = _env(num_uavs=3, comm_action_mode="compact_3")
    now = env.net.sim_time_us
    env.net.routes[0] = {
        1: (1, 0, 3, now + 1_000_000),
        2: (2, 2, 5, now + 1_000_000),
        env.gcs_id: (1, 2, 9, now),
    }
    before = env.net.routes[0].copy()
    obs, _, _ = env._build_obs_state_mask()
    assert obs[0, 10] == 0.0
    assert obs[0, 13] == .25  # Only peer2 remains active among four radio nodes.
    assert env.net.routes[0] == before
    env.net.routes[0][env.gcs_id] = (1, 2, 9, now + 1)
    obs, _, _ = env._build_obs_state_mask()
    assert obs[0, 10] == 1.0
    assert obs[0, 13] == .5


@pytest.mark.parametrize("comm_action_mode", ["compact_3", "per_peer"])
def test_reading_observation_does_not_change_next_packet_payload(comm_action_mode):
    # Extra reads of the same observation must not rotate the relay scheduler.
    envs = [_env(num_uavs=3, comm_action_mode=comm_action_mode,
                 enable_data_mule_forwarding=True) for _ in range(2)]
    for env in envs:
        for origin in (1, 2):
            env.net.ledger_pos[0, origin, :62] = 1
            env.net.ledger_gen_ts_s[0, origin, :62] = 1
    envs[1]._build_obs_state_mask()
    payloads = [e.net.get_dirty_entries_non_oracle(0, e.gcs_id) for e in envs]
    assert len(payloads[0]) == 62
    assert payloads[0] == payloads[1]
    assert set(unpack_entry_2x_uint32(*p)[0] for p in payloads[0]) <= {1, 2}


def test_team_confirmation_ages_each_source_before_combining_effective_counts():
    # Two old positives have mass 1 - .2**(2*exp(-2)) = .353141, below .9.
    env = _env(num_uavs=2, comm_action_mode="compact_3", fov_radius_m=.01,
               aoi_ewma_end_to_end=True)
    env.true_targets.fill(False)
    env.true_targets[100] = True
    for u in range(2):
        env.net.record_local_observation(u, 100, True, 1)
    env.t = 100
    *_, info = env.step(np.zeros((2, 2)), np.zeros(2, dtype=np.int64))
    assert info["local_tp_discovered"] == 0
    assert info["r_loc_team"] == pytest.approx(.1766, abs=5e-5)


def test_continuous_aoi_penalty_uses_current_age_instead_of_running_mean():
    env = _env(num_uavs=2, fov_radius_m=.01, continuous_aoi_penalty_w=60.0)
    action = (np.zeros((2, 2)), np.zeros(2, dtype=np.int64))
    env.step(*action)
    *_, info = env.step(*action)
    assert info["r_continuous_aoi"] == -2.0
    assert info["current_gcs_aoi_mapwide_censored_s"] == 2.0
    assert info["mean_gcs_aoi_mapwide_censored_s"] == 1.5


def test_first_confirmation_uses_latest_novel_reception_even_for_older_support():
    env = _env(num_uavs=2, fov_radius_m=.01, aoi_reward_mode="exponential")
    env.true_targets.fill(False)
    env.true_targets[100] = True

    def receive_positive(origin, generation_s, sequence):
        env.net.radios[origin].active_next_hop = env.gcs_id
        packet = Packet(
            pkt_id=10_000 + sequence, episode_id=env.episode_id,
            pkt_type=PKT_DATA, app_src=origin, app_dst=env.gcs_id,
            app_seq=sequence, create_us=env.net.sim_time_us,
            expire_us=env.net.sim_time_us + 10_000_000,
            packed_entries=(pack_entry_2x_uint32(origin, 100, generation_s, 1, 0, max_origins=2),),
        )
        env.net._handle_rx_frame(env.gcs_id, origin, packet)

    assert env.net.run_until_us(env.epoch_us + 1_000_000)
    env.t = 1
    receive_positive(0, 1, 1)
    action = (np.zeros((2, 2)), np.zeros(2, dtype=np.int64))
    *_, first_info = env.step(*action)
    assert first_info["gcs_tp_confirmed"] == 0  # A single positive has mass .8.
    assert env.net.run_until_us(env.epoch_us + 10_000_000)
    env.t = 10
    receive_positive(1, 0, 2)  # Older novel support raises fused mass above .9.
    *_, info = env.step(*action)
    assert info["gcs_tp_confirmed"] == 1
    assert info["mean_first_tp_confirm_latency_s"] == 9.0
    assert info["r_gcs_aoi"] == pytest.approx(5.1642, abs=5e-5)  # 6 exp(-9/60).
    assert env.net.gcs_last_rx_us[100] == env.epoch_us + 1_000_000
    assert env.net.gcs_last_novel_rx_us[100] == env.epoch_us + 10_000_000


@pytest.mark.parametrize("n,grid", [(2, 10), (6, 10), (30, 20)])
def test_accepted_map_and_fleet_spawn_with_physical_clearance(n, grid):
    # Geometric map scaling must preserve the configured 50m physical spacing.
    env = _env(num_uavs=n, grid_w=grid, map_size_m=50.0 * grid)
    violations = trajectory_safety_violations(
        env.positions[:n, :2], np.zeros((n, 2)), np.ones(n, dtype=bool), env.buildings,
        map_size_m=env.cfg.map_size_m, dt_s=1.0,
        uav_safe_distance_m=env.cfg.uav_safe_distance_m,
    )
    assert not any(violations.values()), violations
    *_, info = env.step(np.zeros((n, 2)), np.zeros(n, dtype=np.int64))
    assert info["safety_violation"] is False


def test_default_six_uav_spawn_uses_v105_braking_safe_geometry():
    # V105 S10 intentionally replaced obsolete 65 m spacing with 120 m.
    # The older expected positions violate the 95 m two-sided braking margin.
    # Active scenario is capped at 15m/s; generic EnvConfigV54 defaults to 25m/s.
    env = _env(v_max_mps=15.0)
    np.testing.assert_array_equal(env.positions[:6, :2], [
        [90., 110.], [210., 140.], [330., 110.],
        [450., 140.], [570., 110.], [690., 140.],
    ])
    braking_margin = env.cfg.uav_safe_distance_m + env.cfg.v_max_mps**2 / env.cfg.a_max_mps2
    for i in range(6):
        for j in range(i + 1, 6):
            assert np.linalg.norm(env.positions[i, :2] - env.positions[j, :2]) >= braking_margin


@pytest.mark.parametrize("n,mode", [(4, "compact_3"), (6, "compact_3"), (8, "per_peer")])
def test_learned_rf_bridge_executes_any_fleet_and_route_mask_size(n, mode):
    # The complete bridge, not just RF decoding, must accept N and D from env.
    env = _env(num_uavs=n, comm_action_mode=mode, enable_tx_power_control=True)
    _, _, mask = env._build_obs_state_mask()
    controls = np.zeros((n, 3))
    controls[:, 2] = np.linspace(-1.0, 1.0, n)
    policy = SimpleNamespace(continuous=controls, destination=np.zeros(n, dtype=np.int64))
    result = step_from_policy_action(env, policy, mask, power_control="learned")
    assert result[0].shape == (n, 514)
    np.testing.assert_allclose(env.net.tx_power_per_radio[:n], np.linspace(.05, .5, n))


def test_queue_coalescing_is_exposed_as_an_explicit_scenario_option():
    env = _env(num_uavs=2, coalesce_duplicate_queue=True)
    env.net.record_local_observation(0, 100, True, 1)
    entries = env.net.get_dirty_entries_non_oracle(0, env.gcs_id)
    first = env.net.enqueue_app_data(0, env.gcs_id, entries)
    second = env.net.enqueue_app_data(0, env.gcs_id, entries)
    assert first == second
    assert len(env.net.radios[0].queue) == 1


@pytest.mark.parametrize("value", ["true", 1, None])
def test_queue_coalescing_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match="coalesce_duplicate_queue"):
        EnvConfigV54(coalesce_duplicate_queue=value)
