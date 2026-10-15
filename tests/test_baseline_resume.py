"""Fixed-schedule checkpoints preserve learner/replay clocks across environment resets."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from importlib.util import find_spec

import numpy as np
import torch

from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.experiments.continuation import restore_learner
from uav_network.experiments.tracking import OnlineTracker
from uav_network.experiments.training import train


def small_config(folder, algorithm="matd3"):
    config = Config()
    config.world.num_uavs, config.world.num_targets, config.world.num_obstacles = 2, 2, 1
    config.world.max_steps = 3
    config.network.enabled = False
    config.runtime.algorithm, config.runtime.device, config.runtime.replay_device = algorithm, "cpu", "cpu"
    config.runtime.num_envs, config.runtime.episodes, config.runtime.output_dir = 1, 2, str(folder)
    config.runtime.collector_overlap = False
    config.algorithm_config.hidden_sizes = [8, 8]
    config.algorithm_config.batch_size = config.algorithm_config.learning_starts = 2
    config.algorithm_config.replay_capacity = 32
    return config


def assert_nested_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_nested_equal(test, actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected):
            assert_nested_equal(test, value, reference)
    else:
        test.assertEqual(actual, expected)


class BaselineResumeTests(unittest.TestCase):
    def test_vector_episode_batch_writes_full_replay_once_per_collection(self):
        from unittest.mock import patch
        from uav_network.experiments.fixed_checkpoints import save_fixed_checkpoint
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                config = small_config(Path(directory), algorithm)
                config.network.enabled = True
                config.runtime.environment_backend = "warp_packet"
                config.runtime.num_envs, config.runtime.episodes = 2, 4
                config.world.max_steps, config.runtime.checkpoint_interval = 1, 1
                completed = []

                def record_write(*args, **kwargs):
                    result = save_fixed_checkpoint(*args, **kwargs)
                    saved = torch.load(args[1], map_location="cpu", weights_only=True)
                    completed.append(saved["progress"]["episodes"])
                    return result

                with patch("uav_network.experiments.training.save_fixed_checkpoint", side_effect=record_write):
                    result = train(config)
                self.assertEqual(completed, [2, 4])
                self.assertEqual(result["episodes"], 4)
                saved = torch.load(result["resume_checkpoint"], map_location="cpu", weights_only=True)
                self.assertEqual(len(saved["progress"]["records"]), 4)

    def test_segment_resume_preserves_weights_adam_replay_and_learning_schedule(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                config = small_config(Path(directory) / "first", algorithm)
                first = train(config)
                self.assertTrue(first.get("resume_checkpoint"), "baseline must emit full-state resume checkpoint")
                saved = torch.load(first["resume_checkpoint"], weights_only=True)
                light = torch.load(first["checkpoint"], weights_only=True)
                self.assertEqual(light["version"], 2)
                self.assertNotIn("training", light)
                self.assertEqual(saved["version"], 3)
                self.assertEqual(saved["training"]["trainer"], "fixed_schedule")
                self.assertEqual(saved["training"]["replay"]["size"], first["global_step"])
                self.assertTrue(saved["training"]["optimizers"]["critic"]["state"])
                config.runtime.episodes, config.runtime.output_dir = 3, str(Path(directory) / "second")
                config.runtime.warm_start_checkpoint = first["resume_checkpoint"]
                learner, progress = restore_learner(config, "cpu")
                self.assertEqual(learner.updates, saved["updates"])
                self.assertEqual(learner.actor_updates, saved["actor_updates"])
                self.assertEqual(progress["episodes"], 2)
                assert_nested_equal(self, learner.state_dict(), saved["model"])
                assert_nested_equal(self, learner.critic_optimizer.state_dict(), saved["training"]["optimizers"]["critic"])
                for optimizer, state in zip(learner.actor_optimizers, saved["training"]["optimizers"]["actors"]):
                    assert_nested_equal(self, optimizer.state_dict(), state)
                if algorithm == "masac":
                    assert_nested_equal(self, learner.alpha_optimizer.state_dict(), saved["training"]["optimizers"]["alpha"])
                second = train(config)
                self.assertEqual(second["state_restore"], "training_state_environment_reset")
                self.assertTrue(second["optimizer_replay_restored"])
                self.assertEqual(second["initial_episodes"], first["episodes"])
                self.assertEqual(second["initial_global_step"], first["global_step"])
                self.assertEqual(second["episodes"], 3)
                self.assertEqual(second["new_updates"], second["collection_steps"] - first["collection_steps"])
                resumed = torch.load(second["resume_checkpoint"], weights_only=True)
                old_replay, new_replay = saved["training"]["replay"], resumed["training"]["replay"]
                for key, value in old_replay["data"].items():
                    torch.testing.assert_close(new_replay["data"][key][:old_replay["size"]], value, rtol=0, atol=0)
                self.assertEqual(resumed["progress"]["records"][:2], saved["progress"]["records"])
                self.assertGreater(resumed["progress"]["records"][-1]["seed"], saved["progress"]["records"][-1]["seed"])

    def test_overlap_resume_does_not_repeat_already_drained_final_collection_update(self):
        with TemporaryDirectory() as directory:
            config = small_config(Path(directory) / "first")
            config.runtime.num_envs, config.runtime.collector_overlap = 2, True
            first = train(config)
            config.runtime.episodes, config.runtime.output_dir = 4, str(Path(directory) / "second")
            config.runtime.warm_start_checkpoint = first["resume_checkpoint"]
            second = train(config)
            self.assertEqual(second["new_updates"], second["collection_steps"] - first["collection_steps"])
            self.assertEqual(second["updates"], first["updates"] * 2)

    def test_replay_wraparound_and_random_generators_continue_after_restore(self):
        self.assertIsNotNone(find_spec("uav_network.experiments.fixed_checkpoints"), "fixed-schedule full-state helper is required")
        from uav_network.experiments.fixed_checkpoints import replay_state, restore_replay, runtime_rng_state, restore_runtime_rng
        replay = TensorReplayBuffer(5, (2, 3), 4, seed=13)
        values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        action = Action(np.ones((2, 2, 3)), np.zeros((2, 2), dtype=np.int64))
        for i in range(4):
            replay.add_batch(values + i, np.ones((2, 4)) * i, action, np.ones(2) * i,
                             values + i + 1, np.ones((2, 4)) * (i + 1), np.zeros(2))
        replay.sample(2)
        state = deepcopy(replay_state(replay))
        rng = np.random.default_rng(71)
        rng.random(3)
        runtime = deepcopy(runtime_rng_state(rng))
        expected = (replay.sample(4), torch.rand(3), rng.random(3), np.random.random(3))
        replacement = TensorReplayBuffer(5, (2, 3), 4, seed=99)
        replacement_rng = np.random.default_rng(99)
        restore_replay(replacement, state)
        restore_runtime_rng(runtime, replacement_rng)
        self.assertEqual((replacement.index, replacement.size), (replay.index, replay.size))
        assert_nested_equal(self, replacement.sample(4), expected[0])
        torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)
        np.testing.assert_array_equal(replacement_rng.random(3), expected[2])
        np.testing.assert_array_equal(np.random.random(3), expected[3])

    def test_device_collector_warmup_generator_continues_after_restore(self):
        from uav_network.experiments.fixed_checkpoints import runtime_rng_state, restore_runtime_rng
        collector = torch.Generator().manual_seed(31)
        torch.rand(4, generator=collector)
        rng = np.random.default_rng(45)
        try:
            saved = runtime_rng_state(rng, action_generator=collector)
        except TypeError:
            self.fail("Full-state RNG checkpoints must include the device collector generator")
        expected = torch.rand(5, generator=collector)
        restored = torch.Generator().manual_seed(99)
        restore_runtime_rng(saved, np.random.default_rng(99), action_generator=restored)
        torch.testing.assert_close(torch.rand(5, generator=restored), expected, rtol=0, atol=0)

    def test_incompatible_algorithm_schedule_or_replay_settings_are_refused(self):
        with TemporaryDirectory() as directory:
            config = small_config(Path(directory) / "first")
            first = train(config)
            self.assertTrue(first.get("resume_checkpoint"), "baseline must emit full-state resume checkpoint")
            config.runtime.warm_start_checkpoint, config.runtime.episodes = first["resume_checkpoint"], 3
            for name in ("algorithm", "num_envs", "collector_overlap", "learning_starts", "replay_capacity"):
                with self.subTest(setting=name):
                    changed = deepcopy(config)
                    if name == "algorithm":
                        changed.runtime.algorithm = "masac"
                    elif name in ("num_envs", "collector_overlap"):
                        setattr(changed.runtime, name, 2 if name == "num_envs" else True)
                    else:
                        setattr(changed.algorithm_config, name, getattr(changed.algorithm_config, name) + 1)
                    with self.assertRaises(ValueError):
                        restore_learner(changed, "cpu")

    def test_non_multiple_vector_budget_reports_exactly_requested_episode_prefix(self):
        with TemporaryDirectory() as directory:
            config = small_config(Path(directory) / "budget")
            config.runtime.episodes, config.runtime.num_envs = 6, 4
            config.algorithm_config.batch_size = config.algorithm_config.learning_starts = 4
            result = train(config)
            self.assertEqual(result["episodes"], 6)
            self.assertEqual(result["segment_episodes"], 6)
            saved = torch.load(result["checkpoint"], weights_only=True)
            self.assertEqual([row["episode"] for row in saved["progress"]["records"]], list(range(1, 7)))
            self.assertEqual(len((Path(config.runtime.output_dir) / "episodes.jsonl").read_text().splitlines()), 6)

    def test_periodic_checkpoint_saves_resume_before_final_completion(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "training"
            config = small_config(output)
            config.runtime.checkpoint_interval = 1
            periodic = []
            def checkpoint(path, progress):
                resume = output / "resume_latest.pt"
                self.assertTrue(resume.is_file(), "periodic saves must retain complete resume state")
                periodic.append(torch.load(resume, weights_only=True)["progress"]["episodes"])
            tracker = SimpleNamespace(episode=lambda record: None, learner=lambda *args: None,
                diagnostic=lambda *args: None, failure=lambda *args: None, progress=lambda *args, **kwargs: None,
                identity=lambda: {"wandb_run_id": "test"}, checkpoint=checkpoint,
                complete=lambda result: {}, close=lambda **kwargs: None)
            from unittest.mock import patch
            with patch("uav_network.experiments.training.start_tracking", return_value=tracker):
                train(config)
            self.assertEqual(periodic, [1, 2])

    def test_final_fixed_artifact_includes_resume_but_legacy_policy_only_remains_valid(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            config = small_config(output / "training")
            result = train(config)
            (output / "episodes.jsonl").touch()
            artifacts = []
            def artifact(*args, **kwargs):
                files = {}
                instance = SimpleNamespace(add_file=lambda path, name: files.update({name: path}), files=files)
                artifacts.append(instance)
                return instance
            tracker = OnlineTracker.__new__(OnlineTracker)
            tracker.config, tracker.output = config, output
            tracker.wandb = SimpleNamespace(Artifact=artifact)
            tracker.run = SimpleNamespace(id="test", save=lambda *args, **kwargs: None,
                log_artifact=lambda *args, **kwargs: SimpleNamespace(wait=lambda **kwargs: None, qualified_name="model:final"))
            tracker.checkpoint(result["checkpoint"], result, final=True)
            self.assertEqual(set(artifacts[-1].files), {"checkpoint.pt", "resume.pt"})
            tracker.checkpoint(result["checkpoint"], {"episodes": 2, "global_step": 6}, final=True)
            self.assertEqual(set(artifacts[-1].files), {"checkpoint.pt"})
            with self.assertRaisesRegex(ValueError, "resume"):
                tracker.checkpoint(result["checkpoint"], {**result, "resume_checkpoint": str(output / "missing.pt")}, final=True)


if __name__ == "__main__":
    unittest.main()
