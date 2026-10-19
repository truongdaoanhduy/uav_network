import unittest
from unittest.mock import patch

import torch

from uav_network.configuration import Config


class BaseDeviceRoleTests(unittest.TestCase):
    def config(self):
        c = Config()
        c.runtime.device = "cuda"
        c.runtime.replay_device = "cuda"
        return c

    def test_one_visible_gpu_keeps_all_base_roles_on_cuda0(self):
        from uav_network.experiments.training import resolve_base_devices
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            collector, learner, replay = resolve_base_devices(self.config())
        self.assertEqual((str(collector), str(learner), str(replay)), ("cuda:0", "cuda:0", "cuda:0"))

    def test_two_visible_gpus_split_base_collector_from_learner_and_replay(self):
        from uav_network.experiments.training import resolve_base_devices
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            collector, learner, replay = resolve_base_devices(self.config())
        self.assertEqual((str(collector), str(learner), str(replay)), ("cuda:0", "cuda:1", "cuda:1"))

    def test_explicit_cpu_replay_is_preserved_with_two_visible_gpus(self):
        from uav_network.experiments.training import resolve_base_devices
        c = self.config()
        c.runtime.replay_device = "cpu"
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            collector, learner, replay = resolve_base_devices(c)
        self.assertEqual((str(collector), str(learner), str(replay)), ("cuda:0", "cuda:1", "cpu"))

    def test_base_collector_refresh_copies_new_actor_weights(self):
        from uav_network.algorithms.checkpoints import make_learner
        from uav_network.environment.observations import observation_dim, state_dim
        from uav_network.experiments.training import refresh_base_collector
        c = Config()
        c.runtime.algorithm = "matd3"
        c.matd3.hidden_sizes = [8, 8]
        source = make_learner(c, observation_dim(c), state_dim(c), "cpu")
        collector = make_learner(c, observation_dim(c), state_dim(c), "cpu")
        with torch.no_grad():
            for parameter in source.actors.parameters():
                parameter.add_(1.0)
        refresh_base_collector(collector, source)
        for left, right in zip(collector.actors.parameters(), source.actors.parameters()):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
