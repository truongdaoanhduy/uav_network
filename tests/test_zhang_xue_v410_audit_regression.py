"""Executable v4.10 regression checks distilled from the supplied audit.

Verify behavior in production modules, not a stand-alone reference patch.
"""
import numpy as np

from uav_network.communication.zhang_xue_ref.reward_engine import RewardEngine
from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)
from uav_network.environment.zhang_xue_coop import (
    ZhangXueCooperativeEnv,
    ZhangXueConfig,
)


def test_search_and_shared_reward_count_empty_cell_uncertainty_reduction():
    truth = np.zeros(400, dtype=bool)
    truth[:10] = True
    engine = RewardEngine(6, 400, truth)
    search = np.zeros(400, dtype=np.float32)
    shared = np.zeros(400, dtype=np.float32)
    search[50] = 0.8  # objectively empty, but informative absence
    shared[100] = 0.4
    mf = np.zeros(400, dtype=np.float32)
    result = engine.compute_unified_step_reward(
        search, shared, mf, airtime_fraction=0.0
    )
    assert np.isclose(result, 30.0 * 0.8 + 15.0 * 0.4 / 5, atol=1e-5)


def test_negative_information_delta_cannot_earn_or_cancel_information_credit():
    truth = np.zeros(400, dtype=bool)
    truth[:10] = True
    engine = RewardEngine(6, 400, truth)
    changes = np.zeros(400, np.float32)
    changes[50] = -1.0  # uncertainty increased
    changes[51] = 0.2   # uncertainty decreased
    reward = engine.compute_unified_step_reward(
        changes, np.zeros(400, np.float32), np.zeros(400, np.float32), 0.0
    )
    assert np.isclose(reward, 30.0 * 0.2, atol=1e-5)


def test_actor_has_local_nfz_geometry_but_not_unseen_distant_nfz():
    env = ZhangXueCooperativeEnv(ZhangXueConfig(), seed=44)
    env.positions[0] = [200.0, 200.0]
    env.no_fly_zones = [(260.0, 280.0, 50.0), (900.0, 900.0, 50.0)]
    obs, _, _ = env._build_outputs()
    np.testing.assert_allclose(obs[0, -8:-4], [.06, .08, .1, 1.], atol=1e-6)
    np.testing.assert_allclose(obs[0, -4:], [0., 0., .3, 0.], atol=1e-6)
    env.no_fly_zones[1] = (850.0, 850.0, 50.0)  # still beyond local view
    obs2, _, _ = env._build_outputs()
    np.testing.assert_array_equal(obs, obs2)


def test_carrier_sensing_detects_ack_emitted_at_receiver_not_data_sender():
    pos = np.full((1, 6, 3), 3000.0, dtype=np.float32)
    pos[:, :, 2] = 100.0
    pos[0, 0] = [0.0, 0.0, 100.0]       # DATA sender (distant to UAV2)
    pos[0, 1] = [500.0, 0.0, 100.0]     # MAC ACK transmitter
    pos[0, 2] = [1200.0, 0.0, 100.0]    # senses ACK within 700m
    pos[0, 3] = [1500.0, 0.0, 100.0]
    dest = np.full((1, 6), -1, np.int32)
    dest[0, 0], dest[0, 2] = 1, 3
    sizes = np.zeros((1, 6), np.int32)
    sizes[0, [0, 2]] = 20
    channel = np.ones((1, 6), np.int32)
    slots = np.zeros((1, 6, 5), np.int32)
    slots[0, 2, 0] = 5  # planned start 150us; ACK at 140..260us
    deadline = np.full((1, 6), 10000.0, np.float32)
    got = simulate_one_hop_retry_batch(pos, dest, sizes, channel, slots, deadline,
                                       physical_carrier_sense=True)
    assert got["start_us"][0, 0, 0] == 50.0
    assert got["start_us"][0, 2, 0] >= 310.0
    assert got["cca_deferrals"][0, 2] >= 1


def test_collision_priority_reverses_on_alternating_steps_cpu_and_warp():
    from uav_network.environment.gpu.zhang_xue_motion_warp import warp_move_batch

    starting = np.asarray([
        [100., 100.], [200., 100.], [400., 400.], [600., 400.],
        [400., 700.], [600., 700.],
    ], dtype=np.float32)
    acts = np.zeros((6, 2), np.float32)
    acts[0] = [1., 0.]
    acts[1] = [-1., 0.]
    env = ZhangXueCooperativeEnv(seed=44)
    env.no_fly_zones = []
    for step_index, winning, losing in [(0, 0, 1), (1, 1, 0)]:
        env.t = step_index
        env.positions[:] = starting
        safety, boundary = env._move(acts)
        assert boundary == 0
        assert safety == 1
        assert env.positions[winning, 0] == 150.0
        assert env.positions[losing, 0] == starting[losing, 0]
    # The Warp batch kernel must match both CPU priority orientations.
    for step_index in (0, 1):
        env.t = step_index
        env.positions[:] = starting
        env._move(acts)
        got, _, _, _ = warp_move_batch(
            starting[None], np.zeros((1, 0, 3), np.float32),
            acts[None], step_index=step_index,
        )
        np.testing.assert_array_equal(got[0], env.positions)


def test_mobility_aware_hello_period_is_explicit_and_more_frequent():
    from scripts.verify_zhang_xue_v45_sampled_scene_network import (
        run_sampled_scenario_network_probe,
    )

    fixed = dict(envs=2, steps=6, seed=44, device="cpu")
    original = run_sampled_scenario_network_probe(
        **fixed, hello_period_steps=4, hello_ttl_us=12_000_000
    )
    mobile = run_sampled_scenario_network_probe(
        **fixed, hello_period_steps=1, hello_ttl_us=2_000_000
    )
    assert mobile["real_mac_hello_attempts"] > original["real_mac_hello_attempts"]
    assert original["all_direct_routes_from_received_hello_only"]
    assert mobile["all_direct_routes_from_received_hello_only"]
    assert mobile["uavs_communicating_during_blackout"] == 0


def test_false_confirmation_cannot_profit_from_its_own_information_gain():
    truth = np.zeros(400, dtype=bool)
    truth[:10] = True
    engine = RewardEngine(6, 400, truth, omega_fp_hold=0.1)
    evidence = np.zeros(400, dtype=np.float32)
    evidence[60] = 0.8  # spurious belief update on actually empty cell
    confirmed = np.zeros(400, dtype=np.float32)
    confirmed[60] = 0.96  # new false-positive confirmation
    reward = engine.compute_unified_step_reward(
        evidence, np.zeros(400, np.float32), confirmed, 0.0,
    )
    assert reward <= -(engine.omega_fp + engine.omega_fp_hold)
