"""Real policies, counters and collection boundaries across a warm restart."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

import torch

from uav_network.configuration import Config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.training import train


def small_config(folder, algorithm="matd3"):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
    c.world.max_steps = 4
    c.network.enabled = False
    c.runtime.algorithm, c.runtime.device, c.runtime.replay_device = algorithm, "cpu", "cpu"
    c.runtime.num_envs, c.runtime.episodes, c.runtime.output_dir = 2, 2, str(folder)
    c.algorithm_config.hidden_sizes = [16, 16]
    c.algorithm_config.batch_size = c.algorithm_config.learning_starts = 4
    c.algorithm_config.replay_capacity = 64
    return c


class ContinuationTests(unittest.TestCase):
    def test_real_restart_keeps_prefix_and_update_schedule_but_refills_replay(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = small_config(Path(directory) / "first", algorithm)
                first = train(c)
                c.runtime.episodes, c.runtime.output_dir = 4, str(Path(directory) / "second")
                c.runtime.warm_start_checkpoint = first["checkpoint"]
                second = train(c)
                self.assertEqual(second["initial_episodes"], first["episodes"])
                self.assertEqual(second["episodes"], 4)
                self.assertEqual(second["segment_episodes"], 2)
                self.assertEqual(second["initial_global_step"], first["global_step"])
                self.assertEqual(second["new_updates"], first["updates"])
                self.assertEqual(second["updates"], first["updates"] * 2)
                saved = torch.load(second["checkpoint"], map_location="cpu", weights_only=True)
                records = saved["progress"]["records"]
                self.assertEqual([r["episode"] for r in records], [1, 2, 3, 4])
                self.assertEqual(len({r["seed"] for r in records}), 4)
                self.assertTrue(second["episode_budget_complete"])

    def test_restore_keeps_all_targets_and_counters_with_empty_adam_state(self):
        from uav_network.experiments.continuation import restore_learner
        with TemporaryDirectory() as directory:
            c = small_config(Path(directory) / "first")
            first = train(c)
            saved = torch.load(first["checkpoint"], weights_only=True)
            c.runtime.episodes = 4
            c.runtime.warm_start_checkpoint = first["checkpoint"]
            learner, progress = restore_learner(c, "cpu")
            self.assertEqual(learner.updates, saved["updates"])
            self.assertEqual(progress["global_step"], first["global_step"])
            for key, value in learner.state_dict().items():
                torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
            self.assertFalse(learner.critic_optimizer.state)
            self.assertTrue(all(not optimizer.state for optimizer in learner.actor_optimizers))

    def test_missing_progress_or_changed_scenario_does_not_silently_restart(self):
        from uav_network.experiments.continuation import restore_learner
        with TemporaryDirectory() as directory:
            c = small_config(Path(directory) / "first")
            plain = Path(directory) / "plain.pt"
            make_learner(c, observation_dim(c), state_dim(c)).save(plain)
            c.runtime.warm_start_checkpoint = str(plain)
            with self.assertRaisesRegex(ValueError, "progress"):
                restore_learner(c, "cpu")
            c.runtime.warm_start_checkpoint = "none"
            first = train(c)
            changed = deepcopy(c)
            changed.runtime.episodes = 4
            changed.runtime.warm_start_checkpoint = first["checkpoint"]
            changed.sensor.altitude_m = 50
            with self.assertRaisesRegex(ValueError, "sensor"):
                restore_learner(changed, "cpu")

    def test_final_upload_failure_retains_local_training_summary(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "upload-failed"
            tracker = SimpleNamespace(episode=lambda record: None,
                progress=lambda *args, **kwargs: None,
                identity=lambda: {"wandb_run_id": "test", "wandb_url": "https://wandb.ai/test"},
                complete=lambda result: (_ for _ in ()).throw(TimeoutError("upload timeout")),
                close=lambda **kwargs: None)
            with patch("uav_network.experiments.training.start_tracking", return_value=tracker):
                with self.assertRaises(TimeoutError):
                    train(small_config(folder))
            result = json.loads((folder / "summary.json").read_text())
            self.assertEqual(result["episodes"], 2)
            self.assertEqual(result["tracking_status"], "failed")
            self.assertTrue((folder / "final.pt").is_file())

    def test_tiny_deadline_preserves_a_finite_policy_without_inventing_an_episode(self):
        with TemporaryDirectory() as directory:
            c = small_config(Path(directory) / "limited")
            c.world.max_steps = 50
            c.runtime.max_train_seconds = 1e-9
            result = train(c)
            self.assertEqual(result["stop_reason"], "time_budget")
            self.assertFalse(result["episode_budget_complete"])
            self.assertEqual(result["episodes"], 0)
            self.assertIsNone(result["final_episode"])
            saved = torch.load(result["checkpoint"], weights_only=True)
            self.assertEqual(saved["progress"]["records"], [])
            self.assertTrue(all(torch.isfinite(p).all() for p in saved["model"].values()))


if __name__ == "__main__":
    unittest.main()
