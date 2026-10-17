"""Receiver masks must agree with report admission, independent of channel loss."""
import unittest
import numpy as np
from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import destination_mask, local_observations, observation_dim


class ReceiverAdmissionTests(unittest.TestCase):
    def test_v2_link_state_layout_progress_buffer_and_mask_tail(self):
        c = Config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 3, 2, 0
        c.sensor.altitude_m = 20
        old_dim = observation_dim(c)
        c.network.peer_range_m = c.network.gcs_range_m = 500
        c.network.tx_power_min_w, c.network.tx_power_max_w = 0.001, 0.002
        c.network.tx_power_mapping = "log"
        c.network.power_limited_admission = True
        c.network.link_state_observation = True
        c.network.buffer_bytes = 2048
        self.assertEqual(observation_dim(c), old_dim + 3 * c.world.num_uavs)
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[500,500],[500,200],[500,700]],
                                        "targets": [[900,900],[900,800]], "obstacles": []})
            env.reports.discover(0, 0, 0)
            env.reports.discover(1, 1, 0)
            obs = local_observations(env)[0]
            n = c.world.num_uavs
            features = obs[-(n + 1 + 3*n):-(n+1)].reshape(n, 3)
            mask = obs[-(n+1):]
            self.assertGreater(features[0,0], 0.0)
            self.assertLess(features[0,0], 1.0)
            self.assertGreater(features[0,1], 0.0)
            self.assertAlmostEqual(features[0,2], 0.5, places=7)
            self.assertEqual(features[1,0], 0.0)
            self.assertLess(features[1,1], 0.0)
            self.assertEqual(features[1,2], 1.0)
            self.assertEqual(features[2,0], 1.0)
            self.assertGreater(features[2,1], 0.0)
            self.assertEqual(features[2,2], 1.0)
            np.testing.assert_array_equal(mask, [1,1,1,0])
        finally:
            env.close()

    def test_power_limited_admission_can_be_disabled_without_changing_range_mask(self):
        c = Config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 1, 1, 0
        c.sensor.altitude_m = 20
        c.network.gcs_range_m = 500
        c.network.tx_power_min_w, c.network.tx_power_max_w = 0.001, 0.002
        c.network.tx_power_mapping = "log"
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[500,500]], "targets": [[900,900]], "obstacles": []})
            env.reports.discover(0, 0, 0)
            self.assertTrue(destination_mask(env)[0,-1])
            c.network.power_limited_admission = True
            self.assertFalse(destination_mask(env)[0,-1])
        finally:
            env.close()

    def test_known_target_full_buffer_and_expired_reports_are_not_advertised(self):
        c=Config();c.world.num_uavs=3;c.world.num_targets=2;c.world.num_obstacles=0
        c.sensor.altitude_m=20;c.network.buffer_bytes=1024
        env=SwarmSearchEnv(c)
        try:
            env.reset(seed=44,options={"positions":[[500,100],[600,100],[700,100]],
                                     "targets":[[900,900],[900,800]],"obstacles":[]})
            env.reports.discover(0,0,0)
            self.assertTrue(destination_mask(env)[0,1])
            env.reports.known[1,0]=True
            self.assertFalse(destination_mask(env)[0,1])
            env.reports.known[1,0]=False
            env.reports.discover(1,1,0)
            self.assertFalse(destination_mask(env)[0,1])
            env.reports.buffers[1].clear()
            self.assertTrue(destination_mask(env)[0,1])
            env.steps=301
            self.assertFalse(destination_mask(env)[0,1])
            self.assertTrue(destination_mask(env)[0,0])
        finally:env.close()

    def test_repeated_native_cpu_training_matches_weights_and_scientific_records(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_vector import small_parallel_config
        from uav_network.experiments.training import train
        for algorithm in ("masac","matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                c=small_parallel_config(algorithm);c.runtime.episodes=4
                results=[]
                for name in ("first","repeat"):
                    c.runtime.output_dir=str(Path(folder)/name)
                    results.append(train(c)["reproducibility"])
                self.assertEqual(results[0]["model_hash"],results[1]["model_hash"])
                self.assertEqual(results[0]["episode_hash"],results[1]["episode_hash"])

    def test_failure_logging_works_with_env_credentials_without_local_key_file(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from uav_network.experiments.diagnostics import failure_report
        with TemporaryDirectory() as folder:
            c=Config();c.tracking.credentials_file=str(Path(folder)/"absent.txt")
            with patch.dict("os.environ",{"WANDB_API_KEY":"private-test-value"}):
                p=failure_report(c,folder,RuntimeError("private-test-value"),{})
            self.assertNotIn("private-test-value",p.read_text())
