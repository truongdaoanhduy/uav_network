"""W&B must not report a clean finish before learner resources have shut down."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.experiments import training


class RecordingTracker:
    def __init__(self, order):
        self.order = order
        self.closed = False

    def identity(self):
        return {"wandb_run_id": "test", "wandb_url": "https://wandb.ai/test"}

    def episode(self, *args, **kwargs):
        pass

    def learner(self, *args, **kwargs):
        pass

    def diagnostic(self, *args, **kwargs):
        pass

    def progress(self, *args, **kwargs):
        pass

    def checkpoint(self, *args, **kwargs):
        pass

    def failure(self, *args, **kwargs):
        pass

    def complete(self, result):
        self.order.append("tracker_complete")
        return {**self.identity(), "checkpoint_artifact": "owner/project/model:v1"}

    def close(self, failed=False):
        if not self.closed:
            self.closed = True
            self.order.append(("tracker_close", failed))


def base_config(folder):
    config = Config()
    config.world.num_uavs = 2
    config.world.num_targets = 2
    config.world.num_obstacles = 1
    config.world.max_steps = 1
    config.network.enabled = False
    config.runtime.algorithm = "matd3"
    config.runtime.device = config.runtime.replay_device = "cpu"
    config.runtime.environment_backend = "warp_packet"
    config.runtime.num_envs = config.runtime.episodes = 2
    config.runtime.output_dir = str(folder)
    config.algorithm_config.hidden_sizes = [16, 16]
    config.algorithm_config.batch_size = config.algorithm_config.learning_starts = 2
    config.algorithm_config.replay_capacity = 64
    return config


def trace_vector_close(module, order):
    real_make_vector = module.make_vector

    def make_vector(*args, **kwargs):
        vector = real_make_vector(*args, **kwargs)
        real_close = vector.close
        closed = False

        def close():
            nonlocal closed
            if not closed:
                closed = True
                order.append("vector_close")
            return real_close()

        vector.close = close
        return vector

    return make_vector


@unittest.skipUnless(importlib.util.find_spec("warp"), "Warp optional dependency unavailable")
class TrackingCleanupOrderTests(unittest.TestCase):
    def test_base_closes_vector_before_marking_wandb_finished(self):
        order = []
        tracker = RecordingTracker(order)
        with TemporaryDirectory() as directory, \
             patch.object(training, "start_tracking", return_value=tracker), \
             patch.object(training, "make_vector", side_effect=trace_vector_close(training, order)):
            result = training.train(base_config(Path(directory) / "base"))
        self.assertEqual(result["tracking_status"], "finished")
        self.assertLess(order.index("vector_close"), order.index(("tracker_close", False)), order)

    def test_pql_still_closes_vector_and_marks_wandb_failed_when_worker_cleanup_fails(self):
        from test_pql_system import pql_config
        from uav_network.experiments import pql_training

        order = []
        tracker = RecordingTracker(order)
        config = pql_config("masac", "synchronous")
        config.runtime.environment_backend = "warp_packet"
        config.runtime.episodes = config.runtime.num_envs
        config.world.max_steps = 1
        config.runtime.checkpoint_interval = config.runtime.episodes + 1
        real_system = pql_training.PQLSystem

        def make_system(*args, **kwargs):
            engine = real_system(*args, **kwargs)
            real_close = engine.close

            def close():
                order.append("engine_close")
                real_close()
                raise RuntimeError("cleanup failure")

            engine.close = close
            return engine

        with TemporaryDirectory() as directory:
            config.runtime.output_dir = str(Path(directory) / "pql-cleanup-failure")
            with patch.object(pql_training, "start_tracking", return_value=tracker), \
                 patch.object(pql_training, "make_vector", side_effect=trace_vector_close(pql_training, order)), \
                 patch.object(pql_training, "PQLSystem", side_effect=make_system):
                with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
                    training.train(config)
            summary = __import__("json").loads((Path(config.runtime.output_dir) / "summary.json").read_text())
            self.assertEqual(summary["tracking_status"], "failed")
        self.assertIn("vector_close", order, order)
        self.assertLess(order.index("vector_close"), order.index(("tracker_close", True)), order)

    def test_pql_closes_workers_and_vector_before_marking_wandb_finished(self):
        from test_pql_system import pql_config
        from uav_network.experiments import pql_training

        order = []
        tracker = RecordingTracker(order)
        config = pql_config("masac", "synchronous")
        config.runtime.environment_backend = "warp_packet"
        config.runtime.output_dir = "unused"
        config.runtime.episodes = config.runtime.num_envs
        config.world.max_steps = 1
        config.runtime.checkpoint_interval = config.runtime.episodes + 1

        real_system = pql_training.PQLSystem

        def make_system(*args, **kwargs):
            engine = real_system(*args, **kwargs)
            real_close = engine.close
            closed = False

            def close():
                nonlocal closed
                if not closed:
                    closed = True
                    order.append("engine_close")
                return real_close()

            engine.close = close
            return engine

        with TemporaryDirectory() as directory:
            config.runtime.output_dir = str(Path(directory) / "pql")
            with patch.object(pql_training, "start_tracking", return_value=tracker), \
                 patch.object(pql_training, "make_vector", side_effect=trace_vector_close(pql_training, order)), \
                 patch.object(pql_training, "PQLSystem", side_effect=make_system):
                result = training.train(config)
        self.assertEqual(result["tracking_status"], "finished")
        tracker_index = order.index(("tracker_close", False))
        self.assertLess(order.index("engine_close"), tracker_index, order)
        self.assertLess(order.index("vector_close"), tracker_index, order)


if __name__ == "__main__":
    unittest.main()
