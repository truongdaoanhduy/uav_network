"""Contract tests for *new* Zhang/Xue 1-km CPU integrated environment.

These intentionally fail before the production module is implemented.
No inference from these tests may be made about Warp/SimPy parity.
"""
import unittest
import numpy as np
import torch

from uav_network.environment.zhang_xue_coop import (
    ZhangXueCooperativeEnv,
    ZhangXueConfig,
)
from uav_network.algorithms.vanilla_baselines import VanillaMATD3, VanillaMASAC


def still():
    return np.zeros((6, 2), np.float32)


def no_send():
    return np.zeros(6, np.int32)


class TestProductionZhangXue(unittest.TestCase):
    def fresh(self, seed=44, horizon=200):
        env = ZhangXueCooperativeEnv(ZhangXueConfig(horizon_steps=horizon), seed=seed)
        obs, state, mask = env.reset(seed=seed)
        return env, obs, state, mask

    def test_scenario_matches_v32_and_actor_interface(self):
        env, obs, state, mask = self.fresh()
        self.assertEqual(env.config.map_size_meter, 1000.)
        self.assertEqual(env.config.target_count, 10)
        self.assertEqual(env.config.uav_count, 6)
        self.assertEqual(env.config.grid_width, 20)
        self.assertEqual(env.config.grid_height, 20)
        self.assertEqual(len(env.no_fly_zones), 2)
        self.assertEqual(len(env.blackout_zones), 2)
        self.assertEqual(int(env.target_mask.sum()), 10)
        self.assertEqual(obs.shape, (6, env.obs_dim))
        self.assertEqual(state.shape, (env.state_dim,))
        self.assertEqual(mask.shape, (6, 6))
        np.testing.assert_array_equal(mask[:, 0], np.ones(6, np.uint8))
        np.testing.assert_array_equal(mask[:, 1:], np.zeros((6, 5), np.uint8))
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(env.state_dim, 6 * env.obs_dim)

    def test_no_instant_fusion_and_communication_through_action(self):
        env, _, _, _ = self.fresh()
        env.step(still(), no_send())  # own sensing of one cell at t=1
        sender_pos = env.belief.pos[0, 0].copy()
        sender_neg = env.belief.neg[0, 0].copy()
        self.assertEqual(int(sender_pos.sum() + sender_neg.sum()), 1)
        self.assertEqual(int(env.belief.pos[1, 0].sum()+env.belief.neg[1, 0].sum()), 0)
        obs, state, mask = env._build_outputs()
        self.assertEqual(int(mask[0, 1]), 1)  # peer-slot 1 is UAV 1
        actions = no_send()
        actions[0] = 1
        _, _, _, _, _, info = env.step(still(), actions)
        self.assertEqual(int(env.belief.pos[1, 0].sum()+env.belief.neg[1, 0].sum()), 1)
        self.assertTrue((0, 1) in env.net.acked_sequences)
        self.assertGreaterEqual(info["data_generated"], 1)
        self.assertGreater(info["data_pdr"], 0.)
        np.testing.assert_array_equal(env.belief.pos[0, 0] >= sender_pos, np.ones(400, bool))

    def test_sender_blackout_blocks_but_recovery_sends(self):
        env, _, _, _ = self.fresh()
        env.step(still(), no_send())
        x, y = env.positions[0]
        env.blackout_zones = [(int(x // 50), int(y // 50))]
        obs, state, mask = env._build_outputs()
        self.assertEqual(int(mask[0, 1]), 0)
        actions = no_send()
        actions[0] = 1
        env.step(still(), actions)
        self.assertEqual(env.net.metrics.data_generated, 0)
        env.blackout_zones = [(0, 19)]
        obs, state, mask = env._build_outputs()
        self.assertEqual(int(mask[0, 1]), 1)
        env.step(still(), actions)
        self.assertGreaterEqual(env.net.metrics.data_generated, 1)

    def test_send_entries_are_pre_step_cumulative_not_future_sensing(self):
        env, _, _, _ = self.fresh()
        env.step(still(), no_send())
        before = env.belief.pos[0, 0].copy() + env.belief.neg[0, 0].copy()
        acts = no_send()
        acts[0] = 1
        env.step(still(), acts)
        self.assertGreaterEqual(len(env.last_sent_packets), 1)
        pkt = env.last_sent_packets[0]
        for origin, cell, npos, nneg in pkt.entries:
            self.assertEqual(origin, 0)
            self.assertLessEqual(npos+nneg, int(before[cell]))

    def test_masked_action_does_not_generate_packets(self):
        env, _, _, _ = self.fresh()
        acts = np.full(6, 1, np.int32)
        env.step(still(), acts)
        self.assertEqual(env.net.metrics.data_generated, 0)

    def test_unacknowledged_packet_pending_state_expires_after_ack_window(self):
        config = ZhangXueConfig(horizon_steps=9, packet_ttl_seconds=1.)
        env = ZhangXueCooperativeEnv(config, seed=44)
        env.step(still(), no_send())
        env.positions[1] = np.array([999., 999.], dtype=np.float32)
        env._sync_physics()
        acts = no_send()
        acts[0] = 1
        env.step(still(), acts)
        self.assertGreater(len(env._pending), 0)
        for _ in range(3):
            env.step(still(), no_send())
        self.assertEqual(len(env._pending), 0)

    def test_horizon_done_and_reset_clear_all(self):
        env, _, _, _ = self.fresh(horizon=3)
        for k in range(3):
            _, _, _, reward, done, info = env.step(still(), no_send())
            self.assertTrue(np.isfinite(reward))
        self.assertTrue(done)
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])
        with self.assertRaises(RuntimeError):
            env.step(still(), no_send())
        _, _, _ = env.reset(seed=44)
        self.assertEqual(env.t, 0)
        self.assertEqual(int(env.belief.pos.sum() + env.belief.neg.sum()), 0)

    def test_actor_and_critic_cannot_read_ground_truth(self):
        env, obs, state, mask = self.fresh()
        # The reward's truth is private to the reward boundary.
        env.target_mask[:] = ~env.target_mask
        obs2, state2, mask2 = env._build_outputs()
        np.testing.assert_array_equal(obs, obs2)
        np.testing.assert_array_equal(state, state2)
        np.testing.assert_array_equal(mask, mask2)

    def test_remote_out_of_radio_range_position_is_hidden(self):
        env, _, _, _ = self.fresh()
        env.positions[1] = np.array([999., 999.], dtype=np.float32)
        obs, state, mask = env._build_outputs()
        # The actor may choose a peer destination through relays, but must
        # not receive its exact coordinates without a local radio sighting.
        np.testing.assert_array_equal(obs[0, 9:12], np.zeros(3, np.float32))

    def test_remote_peer_inside_blackout_is_not_a_live_position_beacon(self):
        env, _, _, _ = self.fresh()
        env.positions[1] = np.array([800.,800.],dtype=np.float32)
        # Destination is in physical radio range but only it is blackout.
        env.blackout_zones = [(16,16)]
        self.assertFalse(env._blackout(env.positions[0]))
        self.assertTrue(env._blackout(env.positions[1]))
        obs, state, mask = env._build_outputs()
        np.testing.assert_array_equal(obs[0, 9:12], np.zeros(3,np.float32))
        self.assertEqual(int(mask[0,0]),1)

    def test_determinism_seed(self):
        a, *_ = self.fresh(seed=21)
        b, *_ = self.fresh(seed=21)
        for _ in range(5):
            ra = a.step(still(), no_send())
            rb = b.step(still(), no_send())
            for i in range(5):
                np.testing.assert_allclose(ra[i], rb[i], rtol=0, atol=0)

    def test_no_fly_segment_and_collision_safety(self):
        env, *_ = self.fresh()
        x, y = env.positions[0]
        env.no_fly_zones = [(x + 18., y, 50.)]
        old = env.positions.copy()
        act = still()
        act[0] = (1., 0.)
        _, _, _, _, _, info = env.step(act, no_send())
        np.testing.assert_array_equal(env.positions[0], old[0])
        self.assertGreater(info["safety_violations"], 0)

    def test_hybrid_matd3_masac_real_learner_update(self):
        env, obs, state, mask = self.fresh(horizon=8)
        next_obs, next_state, next_mask, reward, done, info = env.step(still(), no_send())
        samples = 2
        batch = {
            "obs": torch.from_numpy(np.stack([obs] * samples)).float(),
            "state": torch.from_numpy(np.stack([state] * samples)).float(),
            "mask": torch.from_numpy(np.stack([mask] * samples)).bool(),
            "act_cont": torch.zeros(samples, 6, 2),
            "act_disc": torch.nn.functional.one_hot(torch.zeros(samples, 6, dtype=torch.long), 6).float(),
            "rewards": torch.full((samples, 1), float(reward)),
            "dones": torch.zeros(samples, 1),
            "next_obs": torch.from_numpy(np.stack([next_obs] * samples)).float(),
            "next_state": torch.from_numpy(np.stack([next_state] * samples)).float(),
            "next_mask": torch.from_numpy(np.stack([next_mask] * samples)).bool(),
        }
        for typ in (VanillaMATD3, VanillaMASAC):
            kw = dict(obs_dim=env.obs_dim, state_dim=env.state_dim, num_agents=6,
                      cont_dim=2, disc_dim=6, hidden_sizes=(32, 32), device="cpu")
            if typ is VanillaMATD3:
                kw["policy_delay"] = 1
            learner = typ(**kw)
            with torch.no_grad():
                cont, disc = learner.act(batch["obs"], batch["mask"], explore=False)
                self.assertEqual(tuple(cont.shape), (samples, 6, 2))
                self.assertEqual(tuple(disc.shape), (samples, 6, 6))
            metrics = learner.update(batch)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


if __name__ == "__main__":
    unittest.main()
