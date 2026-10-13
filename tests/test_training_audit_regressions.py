"""Regressions from the completed single-seed training audit."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from uav_network.configuration import Config
from uav_network.communication.reports import ReportStore
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.experiments.training import train


class TrainingAuditTests(unittest.TestCase):
    def test_mission_deadline_disables_bootstrap_for_both_real_learners(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                c = Config()
                c.network.enabled = False
                c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 1, 0, 1
                c.runtime.algorithm, c.runtime.device = algorithm, "cpu"
                c.runtime.episodes = 1
                c.runtime.output_dir = str(Path(folder) / "train")
                a = c.algorithm_config
                a.hidden_sizes, a.replay_capacity = [16, 16], 2
                a.batch_size = a.learning_starts = 1
                result = train(c)
                record = result["final_episode"]
                self.assertEqual(record["termination_reason"], "time_limit")
                self.assertTrue(record["terminated"])
                self.assertFalse(record["truncated"])
                self.assertEqual(record["learner"]["batch_terminated_fraction"], 1)

    def test_fresh_camera_observation_recreates_expired_report_without_reward(self):
        c = Config()
        c.network.enabled = False
        c.network.report_ttl_s = 2
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 1, 1, 0
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[500, 500]],
                                      "targets": [[500, 500]], "obstacles": []})
            action = Action(np.zeros((1, 3)), np.zeros(1, dtype=np.int64))
            env.step(action)
            env.step(action)
            self.assertFalse(env.reports.buffers[0])
            env.step(action)
            self.assertIn(0, env.reports.buffers[0])
            self.assertEqual(env.reports.buffers[0][0].created_s, 2)
            self.assertEqual(env.reports.expired, 1)
            self.assertEqual(env.reward_components["discovery"], 0)
            self.assertTrue(env.reports.known[0, 0])
        finally:
            env.close()

    def test_buffer_drop_can_recover_after_fresh_observation_and_space(self):
        c = Config()
        c.network.buffer_bytes = c.network.report_bytes
        store = ReportStore(c)
        store.discover(0, 0, 0)
        store.discover(0, 1, 0)
        self.assertEqual(store.dropped, 1)
        self.assertTrue(store.transfer(0, c.world.num_uavs, store.buffers[0][0], 1))
        self.assertFalse(store.discover(0, 1, 2))  # No repeated discovery reward.
        self.assertIn(1, store.buffers[0])
        self.assertEqual(store.buffers[0][1].created_s, 2)
        self.assertEqual(store.dropped, 1)

    def test_nlos_ack_completes_at_the_selected_data_power_without_retries(self):
        c = Config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 1
        c.sensor.altitude_m = 20
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[450, 200], [550, 200]],
                                      "targets": [[450, 200]], "obstacles": [[500, 200, 10]]})
            continuous = np.zeros((2, 3))
            continuous[0, 2] = 1  # 0.4 W: data clears this NLOS link's SINR threshold.
            env.step(Action(continuous, np.array([1, 0])))
            metrics = env.network.metrics()
            self.assertEqual(metrics["received_packets"], 1)
            self.assertIn(0, env.reports.buffers[1])
            self.assertEqual(metrics["phy_failures"], 0)
            self.assertEqual(metrics["phy_successes"], 2)  # One data frame and one ACK.
        finally:
            env.close()

    def test_reobserved_report_can_reach_gcs_through_the_real_network(self):
        c = Config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 1, 1, 0
        c.network.report_ttl_s = 2
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[500, 100]],
                                      "targets": [[500, 100]], "obstacles": []})
            silent = Action(np.zeros((1, 3)), np.array([0]))
            for _ in range(3):
                env.step(silent)
            _, _, terminated, _, info = env.step(Action(np.zeros((1, 3)), np.array([1])))
            self.assertTrue(terminated)
            self.assertTrue(info["success"])
            self.assertEqual(info["expired_reports"], 1)
            self.assertEqual(info["regenerated_reports"], 1)
            self.assertEqual(info["reward_components"]["discovery"], 0)
            self.assertEqual(info["network"]["report_average_hops"], 1)
        finally:
            env.close()

    def test_recovery_owner_is_visible_in_actor_observation_and_critic_state(self):
        from uav_network.environment.observations import local_observations
        c = Config()
        c.network.enabled = False
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 0
        observations, states, next_buffers = [], [], []
        for source, receiver in ((0, 1), (1, 0)):
            env = SwarmSearchEnv(c)
            try:
                env.reset(seed=44, options={"positions": [[500, 300], [500, 150]],
                                          "targets": [[900, 900]], "obstacles": []})
                env.reports.discover(source, 0, 0)
                env.reports.transfer(source, receiver, env.reports.buffers[source][0], 1)
                env.reports.expire(300)
                env.steps = 300
                env.targets[:] = [[500, 300]]
                observations.append(local_observations(env))
                states.append(env.state())
                env.step(Action(np.zeros((2, 3)), np.zeros(2, dtype=np.int64)))
                next_buffers.append(bool(env.reports.buffers[0]))
            finally:
                env.close()
        self.assertEqual(next_buffers, [False, True])
        self.assertFalse(np.array_equal(observations[0], observations[1]))
        self.assertFalse(np.array_equal(states[0], states[1]))

    def test_legacy_checkpoint_keeps_original_report_and_ack_behavior(self):
        import torch
        from uav_network.algorithms.checkpoints import make_learner, load_policy
        from uav_network.environment.observations import observation_dim, state_dim
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                c = Config()
                c.runtime.algorithm = algorithm
                c.algorithm_config.hidden_sizes = [16, 16]
                path = Path(folder) / "legacy.pt"
                make_learner(c, observation_dim(c), state_dim(c)).save(path)
                saved = torch.load(path, weights_only=True)
                del saved["config"]["network"]["rediscovery_recovery"]
                del saved["config"]["network"]["ack_uses_data_power"]
                torch.save(saved, path)
                loaded = load_policy(path)
                self.assertFalse(loaded.config.network.rediscovery_recovery)
                self.assertFalse(loaded.config.network.ack_uses_data_power)
                store = ReportStore(loaded.config)
                store.discover(0, 0, 0)
                store.expire(301)
                store.discover(0, 0, 302)
                self.assertFalse(store.buffers[0])

    def test_successful_custody_transfer_does_not_regenerate_a_duplicate(self):
        c = Config()
        store = ReportStore(c)
        store.discover(0, 0, 0)
        self.assertTrue(store.transfer(0, 1, store.buffers[0][0], 1))
        store.discover(0, 0, 2)
        self.assertFalse(store.buffers[0])
        self.assertTrue(store.transfer(1, c.world.num_uavs, store.buffers[1][0], 3))
        store.discover(1, 0, 4)
        self.assertFalse(store.buffers[1])
        self.assertEqual(store.delivery_hops, [2])


if __name__ == "__main__":
    unittest.main()
