"""Parameter-sharing topology and compatibility contracts."""
from dataclasses import replace
import unittest

import numpy as np
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


    def _observations(self, config, batch=3):
        obs = np.zeros((batch, config.world.num_uavs, observation_dim(config)), dtype=np.float32)
        obs[:, :, 9] = 1
        obs[:, :, -(config.world.num_uavs + 1):] = 1
        return obs

    def test_shared_actor_is_agent_conditioned_and_emits_all_uav_actions(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = self._learner(algorithm, True, False)
                actor = learner.actors[0]
                self.assertEqual(actor.encoder[0].in_features, learner.observation_dim + learner.num_uavs)
                self.assertEqual(len(learner.actor_optimizers), 1)
                action = learner.act_batch(self._observations(learner.config), deterministic=True)
                self.assertEqual(action.continuous.shape, (3, 6, 3))
                self.assertEqual(action.destination.shape, (3, 6))


    def _batch(self, config, batch=8):
        rng = np.random.default_rng(44)
        obs = self._observations(config, batch)
        n = config.world.num_uavs
        return {
            "observation": obs,
            "next_observation": obs.copy(),
            "state": rng.normal(size=(batch, state_dim(config))).astype(np.float32),
            "next_state": rng.normal(size=(batch, state_dim(config))).astype(np.float32),
            "continuous": rng.uniform(-1, 1, size=(batch, n, 3)).astype(np.float32),
            "destination": rng.integers(0, n + 1, size=(batch, n)),
            "reward": rng.normal(size=batch).astype(np.float32),
            "terminated": np.array([0, 1] * (batch // 2), dtype=np.float32),
        }

    def test_shared_actor_update_uses_one_optimizer_and_changes_shared_weights(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = self._learner(algorithm, True, False)
                before = next(learner.actors.parameters()).detach().clone()
                metrics = learner.update(self._batch(learner.config))
                if algorithm == "matd3":
                    metrics = learner.update(self._batch(learner.config))
                self.assertEqual(len(learner.actor_optimizers), 1)
                self.assertFalse(torch.equal(before, next(learner.actors.parameters())))
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


    def test_matd3_shared_team_critic_updates_for_c_and_d(self):
        for share_actor in (False, True):
            for vectorized in (False, True):
                with self.subTest(share_actor=share_actor, vectorized=vectorized):
                    learner = self._learner("matd3", share_actor, True)
                    learner.config.runtime.vectorize_critics = vectorized
                    critic_before = next(learner.critics.parameters()).detach().clone()
                    actor_before = next(learner.actors.parameters()).detach().clone()
                    first = learner.update(self._batch(learner.config))
                    second = learner.update(self._batch(learner.config))
                    self.assertFalse(first["policy_updated"])
                    self.assertTrue(second["policy_updated"])
                    self.assertEqual(len(learner.critics), 1)
                    self.assertEqual(len(learner.target_critics), 1)
                    self.assertFalse(torch.equal(critic_before, next(learner.critics.parameters())))
                    self.assertFalse(torch.equal(actor_before, next(learner.actors.parameters())))
                    self.assertTrue(all(np.isfinite(value) for value in second.values()))


    def test_masac_shared_team_critic_updates_for_c_and_d(self):
        for share_actor in (False, True):
            for vectorized in (False, True):
                with self.subTest(share_actor=share_actor, vectorized=vectorized):
                    learner = self._learner("masac", share_actor, True)
                    learner.config.runtime.vectorize_critics = vectorized
                    critic_before = next(learner.critics.parameters()).detach().clone()
                    actor_before = next(learner.actors.parameters()).detach().clone()
                    alpha_c_before = learner.log_alpha_continuous.detach().clone()
                    metrics = learner.update(self._batch(learner.config))
                    self.assertEqual(len(learner.critics), 1)
                    self.assertEqual(len(learner.target_critics), 1)
                    self.assertFalse(torch.equal(critic_before, next(learner.critics.parameters())))
                    self.assertFalse(torch.equal(actor_before, next(learner.actors.parameters())))
                    self.assertFalse(torch.equal(alpha_c_before, learner.log_alpha_continuous))
                    self.assertEqual(learner.log_alpha_continuous.shape, (6,))
                    self.assertEqual(learner.log_alpha_discrete.shape, (6,))
                    self.assertTrue(all(np.isfinite(value) for value in metrics.values()))



if __name__ == "__main__":
    unittest.main()
