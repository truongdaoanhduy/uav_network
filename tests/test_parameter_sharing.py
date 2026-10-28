"""Parameter-sharing topology and compatibility contracts."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from torch.distributions import Categorical

import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.environment.env import Action
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.action_utils import canonical_continuous, joint_action, valid_destinations


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

    def test_masac_independent_actor_preserves_legacy_rng_order(self):
        learner = self._learner("masac", False, False)
        observations = torch.as_tensor(self._observations(learner.config, batch=4))
        torch.manual_seed(2468)
        actual = learner.act_batch(observations.numpy(), deterministic=False, return_tensors=True)
        torch.manual_seed(2468)
        expected = []
        for i, actor in enumerate(learner.actors):
            local = observations[:, i]
            continuous, _, probs, _ = learner._sample(actor, local, False, i)
            destination = Categorical(probs=probs, validate_args=False).sample()
            expected.append(joint_action(canonical_continuous(continuous, local, destination),
                                         destination, learner.destinations))
        expected = torch.stack(expected, 1)
        torch.testing.assert_close(actual.continuous, expected[..., :3], rtol=0, atol=0)
        torch.testing.assert_close(actual.destination, expected[..., 3:].argmax(-1), rtol=0, atol=0)


    def test_matd3_independent_actor_preserves_legacy_rng_order(self):
        learner = self._learner("matd3", False, False)
        observations = torch.as_tensor(self._observations(learner.config, batch=4))
        step = 123
        a = learner.config.matd3

        torch.manual_seed(1234)
        actual = learner.act_batch(observations.numpy(), deterministic=False, step=step, return_tensors=True)
        torch.manual_seed(1234)
        expected = []
        fraction = min(1, max(0, step) / a.epsilon_decay_steps)
        epsilon = a.initial_epsilon + fraction * (a.final_epsilon - a.initial_epsilon)
        for i, actor in enumerate(learner.actors):
            local = observations[:, i]
            mask = valid_destinations(local, learner.destinations)
            continuous, probs = actor(local, mask)
            destination = probs.argmax(-1)
            continuous = (continuous + torch.randn_like(continuous) * a.exploration_noise_std).clamp(-1, 1)
            random_destination = torch.multinomial(mask.float(), 1).squeeze(-1)
            explore = torch.rand(len(local)) < epsilon
            destination = torch.where(explore, random_destination, destination)
            continuous = canonical_continuous(continuous, local, destination)
            expected.append(joint_action(continuous, destination, learner.destinations))
        expected = torch.stack(expected, 1)
        torch.testing.assert_close(actual.continuous, expected[..., :3], rtol=0, atol=0)
        torch.testing.assert_close(actual.destination, expected[..., 3:].argmax(-1), rtol=0, atol=0)

        torch.manual_seed(5678)
        actual_target = learner.target_actions(observations)
        torch.manual_seed(5678)
        expected_target = []
        for i, actor in enumerate(learner.target_actors):
            local = observations[:, i]
            continuous, probs = actor(local, valid_destinations(local, learner.destinations))
            noise = (torch.randn_like(continuous) * a.target_noise_std).clamp(
                -a.target_noise_clip, a.target_noise_clip)
            destination = probs.argmax(-1)
            continuous = canonical_continuous((continuous + noise).clamp(-1, 1), local, destination)
            expected_target.append(joint_action(continuous, destination, learner.destinations))
        torch.testing.assert_close(actual_target, torch.stack(expected_target, 1), rtol=0, atol=0)


    def test_shared_actor_batches_all_uavs_into_one_inference_forward(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = self._learner(algorithm, True, False)
                forwards = []
                hook = learner.actors[0].encoder.register_forward_hook(
                    lambda module, args, output: forwards.append(tuple(args[0].shape)))
                try:
                    learner.act_batch(self._observations(learner.config, batch=3), deterministic=True)
                finally:
                    hook.remove()
                self.assertEqual(forwards, [(18, learner.observation_dim + learner.num_uavs)])


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
        # One Bx6 detached base-policy sample + one Bx6 differentiable actor sample.
        # Q alternatives and entropy/log-prob must reuse the latter sample.
        self.assertEqual(sample.call_count, 2)


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
