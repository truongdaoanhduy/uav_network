"""Original update oracle, independent stage ownership, and target clocks."""
from copy import deepcopy
import unittest
import torch

from uav_network.configuration import Config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.observations import observation_dim, state_dim
from legacy_updates import legacy_masac_update, legacy_matd3_update


def small_learner(algorithm="masac", vectorized=False):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
    c.runtime.algorithm, c.runtime.vectorize_critics = algorithm, vectorized
    c.algorithm_config.hidden_sizes = [16, 16]
    torch.manual_seed(44)
    return make_learner(c, observation_dim(c), state_dim(c))


def replay_batch(learner, count=4):
    generator = torch.Generator().manual_seed(100)
    obs = torch.rand((count, learner.num_uavs, learner.observation_dim), generator=generator)
    obs[:, :, 9] = 1
    obs[:, :, -learner.destinations:] = 1
    obs[0, 0, 9] = 0
    obs[0, 0, -learner.destinations:] = 0
    obs[0, 0, -learner.destinations] = 1
    state = torch.rand((count, learner.state_dim), generator=generator)
    return dict(observation=obs, next_observation=obs.clone(),
                state=state, next_state=state.clone(),
                continuous=torch.zeros(count, learner.num_uavs, 3),
                destination=torch.zeros(count, learner.num_uavs, dtype=torch.long),
                reward=torch.arange(count).float(), terminated=torch.tensor([0, 0, 1, 0]).float())


def optimizers(learner):
    result = [learner.critic_optimizer, *learner.actor_optimizers]
    if learner.algorithm == "masac":
        result.append(learner.alpha_optimizer)
    return [o.state_dict() for o in result]


class SplitUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_nested_close(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_close(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_close(a, b)
        else:
            self.assertEqual(left, right)

    def test_combined_update_matches_frozen_original_equations(self):
        for algorithm, oracle in (("masac", legacy_masac_update), ("matd3", legacy_matd3_update)):
            for vectorized in (False, True):
                with self.subTest(algorithm=algorithm, vectorized=vectorized):
                    actual = small_learner(algorithm, vectorized)
                    original = deepcopy(actual)
                    batch = replay_batch(actual)
                    for step in range(3):
                        torch.manual_seed(500 + step)
                        expected = oracle(original, batch)
                        rng = torch.get_rng_state().clone()
                        torch.manual_seed(500 + step)
                        result = actual.update(batch)
                        self.assertEqual(result, expected)
                        self.assert_nested_close(actual.state_dict(), original.state_dict())
                        self.assert_nested_close(optimizers(actual), optimizers(original))
                        self.assertEqual(actual.updates, original.updates)
                        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_critic_and_policy_stages_only_mutate_owned_parameters(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm):
                learner = small_learner(algorithm)
                self.assertTrue(callable(getattr(learner, "update_critic", None)), "Missing critic-only API")
                self.assertTrue(callable(getattr(learner, "update_actor", None)), "Missing policy-only API")
                self.assertTrue(callable(getattr(learner, "update_targets", None)), "Missing target API")
                batch = replay_batch(learner)
                actors = deepcopy(learner.actors.state_dict())
                targets = deepcopy(learner.target_critics.state_dict())
                learner.update_critic(batch)
                self.assert_nested_close(actors, learner.actors.state_dict())
                self.assert_nested_close(targets, learner.target_critics.state_dict())
                self.assertEqual(learner.updates, 1)
                critics = deepcopy(learner.critics.state_dict())
                learner.update_actor(batch)
                self.assert_nested_close(critics, learner.critics.state_dict())
                self.assertEqual(learner.updates, 1)
                self.assertEqual(learner.actor_updates, 1)


if __name__ == "__main__":
    unittest.main()
