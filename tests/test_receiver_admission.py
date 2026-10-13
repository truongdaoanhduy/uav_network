"""Receiver masks must agree with report admission, independent of channel loss."""
import unittest
from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import destination_mask


class ReceiverAdmissionTests(unittest.TestCase):
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
