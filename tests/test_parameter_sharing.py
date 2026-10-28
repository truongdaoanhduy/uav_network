"""Parameter-sharing topology and compatibility contracts."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.environment.env import Action
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


    def test_masac_shared_critic_reuses_one_policy_sample_per_agent(self):
        learner = self._learner("masac", True, True)
        learner.config.runtime.vectorize_critics = False
        with patch.object(learner, "_sample", wraps=learner._sample) as sample:
            learner.update_actor(self._batch(learner.config), collect_metrics=False)
        # Six detached base-policy samples + six differentiable actor samples.
        # Q alternatives and entropy/log-prob must use the same differentiable sample.
        self.assertEqual(sample.call_count, 12)


    def test_pql_inference_copy_supports_shared_actor(self):
        from uav_network.experiments.pql_workers import inference_copy
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = self._learner(algorithm, True, True)
                collector = inference_copy(learner, "cpu")
                action = collector.act_batch(self._observations(learner.config), deterministic=True)
                self.assertEqual(action.continuous.shape, (3, 6, 3))
                self.assertEqual(action.destination.shape, (3, 6))
                self.assertEqual(len(collector.actors), 1)


    def test_shared_checkpoint_round_trip_preserves_topology_and_actions(self):
        for algorithm, cls in (("masac", MASAC), ("matd3", MATD3)):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                learner = self._learner(algorithm, True, True)
                batch = self._batch(learner.config)
                learner.update(batch)
                if algorithm == "matd3":
                    learner.update(batch)
                observation = self._observations(learner.config)
                before = learner.act_batch(observation, deterministic=True)
                path = Path(directory) / "shared.pt"
                learner.save(path)
                restored = cls.load(path, "cpu")
                after = restored.act_batch(observation, deterministic=True)
                self.assertTrue(restored.share_actor_parameters)
                self.assertTrue(restored.share_critic_parameters)
                self.assertEqual(len(restored.actors), 1)
                self.assertEqual(len(restored.critics), 1)
                np.testing.assert_array_equal(before.continuous, after.continuous)
                np.testing.assert_array_equal(before.destination, after.destination)

    def test_legacy_checkpoint_without_sharing_flags_loads_as_independent(self):
        for algorithm, cls in (("masac", MASAC), ("matd3", MATD3)):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                learner = self._learner(algorithm, False, False)
                path = Path(directory) / "current.pt"
                learner.save(path)
                payload = torch.load(path, weights_only=True)
                payload["config"][algorithm].pop("share_actor_parameters")
                payload["config"][algorithm].pop("share_critic_parameters")
                legacy = Path(directory) / "legacy.pt"
                torch.save(payload, legacy)
                restored = cls.load(legacy, "cpu")
                self.assertFalse(restored.share_actor_parameters)
                self.assertFalse(restored.share_critic_parameters)
                self.assertEqual(len(restored.actors), 6)
                self.assertEqual(len(restored.critics), 6)

    def test_shared_d_topology_runs_through_compile_plumbing(self):
        from uav_network.algorithms.optimization import compile_models
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = self._learner(algorithm, True, True)
                with patch("torch.compile", side_effect=lambda function=None, **kwargs: function):
                    compile_models(learner)
                    metrics = learner.update(self._batch(learner.config))
                    if algorithm == "matd3":
                        metrics = learner.update(self._batch(learner.config))
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


    def _pql_config(self, algorithm, mode):
        config = Config()
        config.runtime.algorithm = algorithm
        config.runtime.device = config.runtime.replay_device = "cpu"
        config.runtime.num_envs = 4
        config.runtime.log_interval_steps = 4
        group = getattr(config, algorithm)
        group.hidden_sizes = [16, 16]
        group.batch_size = 4
        group.replay_capacity = 16
        group.share_actor_parameters = True
        group.share_critic_parameters = True
        config.pql.enabled = True
        config.pql.mode = mode
        config.pql.collector_device = "cpu"
        config.pql.learner_device = "cpu"
        config.pql.policy_device = "cpu"
        config.pql.warmup_collection_steps = 1
        config.pql.critic_sample_ratio = 1
        config.pql.worker_timeout_s = 10
        return config.validate()

    def test_pql_shared_variants_update_in_synchronous_and_async_modes(self):
        from uav_network.experiments.pql_system import PQLSystem
        for algorithm in ("masac", "matd3"):
            for mode in ("synchronous", "async"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    config = self._pql_config(algorithm, mode)
                    engine = PQLSystem(config)
                    try:
                        observation = self._observations(config, batch=4)
                        action = engine.act_batch(observation, deterministic=True)
                        self.assertEqual(action.continuous.shape, (4, 6, 3))
                        batch = self._batch(config, batch=4)
                        transition = (batch["observation"], batch["state"],
                                      Action(batch["continuous"], batch["destination"]),
                                      batch["reward"], batch["next_observation"],
                                      batch["next_state"], batch["terminated"])
                        engine.submit(transition)
                        engine.submit(transition)
                        engine.flush()
                        self.assertGreaterEqual(engine.critic.updates, 2)
                        self.assertGreaterEqual(engine.policy.actor_updates,
                                                2 if algorithm == "masac" else 1)
                        self.assertEqual(len(engine.critic.critics), 1)
                        self.assertEqual(len(engine.policy.actors), 1)
                    finally:
                        engine.close()



if __name__ == "__main__":
    unittest.main()
