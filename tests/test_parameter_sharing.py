"""Parameter-sharing topology and compatibility contracts."""
from dataclasses import replace
import unittest

import torch

from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3


class ParameterSharingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)

    def test_off_policy_parameter_sharing_flags_default_false(self):
        config = Config()
        self.assertIs(config.masac.share_actor_parameters, False)
        self.assertIs(config.masac.share_critic_parameters, False)
        self.assertIs(config.matd3.share_actor_parameters, False)
        self.assertIs(config.matd3.share_critic_parameters, False)

    def _learner(self, algorithm, share_actor, share_critic):
        config = Config()
        config.runtime.algorithm = algorithm
        group = getattr(config, algorithm)
        setattr(group, "hidden_sizes", [16, 16])
        setattr(group, "share_actor_parameters", share_actor)
        setattr(group, "share_critic_parameters", share_critic)
        cls = MASAC if algorithm == "masac" else MATD3
        return cls(config, observation_dim(config), state_dim(config), "cpu")

    def test_abcd_topologies_have_expected_module_counts(self):
        variants = {
            "A": (False, False, 6, 6),
            "B": (True, False, 1, 6),
            "C": (False, True, 6, 1),
            "D": (True, True, 1, 1),
        }
        for algorithm in ("masac", "matd3"):
            for name, (share_actor, share_critic, actors, critics) in variants.items():
                with self.subTest(algorithm=algorithm, variant=name):
                    learner = self._learner(algorithm, share_actor, share_critic)
                    self.assertEqual(len(learner.actors), actors)
                    self.assertEqual(len(learner.critics), critics)
                    self.assertEqual(len(learner.target_critics), critics)
                    if algorithm == "matd3":
                        self.assertEqual(len(learner.target_actors), actors)



if __name__ == "__main__":
    unittest.main()
