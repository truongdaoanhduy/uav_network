"""Training-pilot tests, not a GPU or full UavNetSim benchmark."""
import unittest
import numpy as np

from uav_network.experiments.train_zhang_xue_pilot import run_cpu_pilot


class TestPilotRunner(unittest.TestCase):
    def test_real_replay_actor_critic_updates_matd3(self):
        summary = run_cpu_pilot(
            algorithm="matd3", seed=14, episodes=2, horizon_steps=7,
            learning_starts=3, batch_size=4, hidden_sizes=(32,32),
            eval_episodes=1,
        )
        self.assertEqual(summary["backend"], "cpu_reference_not_uavnetsim")
        self.assertEqual(summary["algorithm"], "matd3")
        self.assertGreater(summary["updates"], 0)
        self.assertGreater(summary["actor_parameter_delta_norm"], 0.)
        self.assertTrue(np.isfinite(summary["mean_eval_return"]))
        self.assertEqual(summary["total_steps"], 14)

    def test_real_replay_actor_critic_updates_masac(self):
        summary = run_cpu_pilot(
            algorithm="masac", seed=14, episodes=2, horizon_steps=7,
            learning_starts=3, batch_size=4, hidden_sizes=(32,32),
            eval_episodes=1,
        )
        self.assertEqual(summary["algorithm"], "masac")
        self.assertAlmostEqual(summary["target_entropy_disc_ratio"], 0.6)
        self.assertGreater(summary["updates"], 0)
        self.assertGreater(summary["actor_parameter_delta_norm"], 0.)
        self.assertGreaterEqual(summary["mean_eval_pdr"], 0.)
        self.assertLessEqual(summary["mean_eval_pdr"], 1.)

    def test_pilot_is_deterministic_under_fixed_seed(self):
        kwargs = dict(algorithm="matd3", seed=99, episodes=1,
                      horizon_steps=5, learning_starts=3,
                      batch_size=3, hidden_sizes=(16,16), eval_episodes=1)
        a = run_cpu_pilot(**kwargs)
        b = run_cpu_pilot(**kwargs)
        self.assertEqual(a["total_steps"], b["total_steps"])
        self.assertAlmostEqual(a["mean_eval_return"], b["mean_eval_return"],places=6)
        self.assertAlmostEqual(a["mean_eval_pdr"], b["mean_eval_pdr"],places=6)


if __name__ == "__main__":
    unittest.main()
