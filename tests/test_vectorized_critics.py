"""Independent live critic parameters retain outputs and gradients under vmap."""
from copy import deepcopy
import importlib
import unittest

import torch
from torch import nn

from uav_network.algorithms.models import TwinCritic
from uav_network.algorithms.checkpoints import make_learner
from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim


def check_vectorized_learners(device="cpu"):
    c = Config()
    c.masac.hidden_sizes = c.matd3.hidden_sizes = [32, 32]
    torch.manual_seed(44)
    obs = torch.randn(8, 6, observation_dim(c), device=device)
    obs[:, :, 9] = 1
    obs[:, :, -7:] = 1
    batch = {"observation": obs, "next_observation": obs.clone(),
             "state": torch.randn(8, state_dim(c), device=device),
             "next_state": torch.randn(8, state_dim(c), device=device),
             "continuous": torch.zeros(8, 6, 3, device=device),
             "destination": torch.zeros(8, 6, dtype=torch.long, device=device),
             "reward": torch.ones(8, device=device),
             "terminated": torch.tensor([0., 1.] * 4, device=device)}
    for algorithm in ("masac", "matd3"):
        c.runtime.algorithm = algorithm
        c.runtime.vectorize_critics = False
        reference = make_learner(c, observation_dim(c), state_dim(c), device)
        c.runtime.vectorize_critics = True
        actual = make_learner(c, observation_dim(c), state_dim(c), device)
        actual.load_state_dict(reference.state_dict())
        for step in range(3):
            torch.manual_seed(44 + step)
            expected_metrics = reference.update(batch)
            torch.manual_seed(44 + step)
            actual_metrics = actual.update(batch)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], value, rtol=2e-5, atol=2e-5)
            for key in expected_metrics:
                torch.testing.assert_close(torch.tensor(actual_metrics[key]),
                                           torch.tensor(expected_metrics[key]), rtol=2e-5, atol=2e-5)


class VectorizedCriticTests(unittest.TestCase):
    def test_compiled_critic_inside_eager_vmap_is_rejected_before_training(self):
        c = Config()
        c.runtime.vectorize_critics = c.runtime.compile_models = True
        with self.assertRaisesRegex(ValueError, "vectorize_critics.*compile_models"):
            c.validate()

    def test_live_independent_outputs_and_parameter_input_gradients(self):
        try:
            module = importlib.import_module("uav_network.algorithms.parallel_critics")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "Missing batched independent critic computation")
        torch.set_num_threads(1)
        for per_critic in (False, True):
            torch.manual_seed(44)
            reference = nn.ModuleList(TwinCritic(5, 3, 4, [16, 16]).double() for _ in range(3))
            actual = deepcopy(reference)
            state = torch.randn(4, 5, dtype=torch.float64)
            actions = torch.randn(*((3,) if per_critic else ()), 4, 3, 7, dtype=torch.float64)
            expected_input = actions.clone().requires_grad_()
            actual_input = actions.clone().requires_grad_()
            pairs = [critic(state, expected_input[i] if per_critic else expected_input)
                     for i, critic in enumerate(reference)]
            expected = tuple(torch.stack([p[k] for p in pairs], 1) for k in (0, 1))
            observed = module.critic_values(actual, state, actual_input, True, per_critic)
            for old, new in zip(expected, observed):
                torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
            sum(v.square().mean() for v in expected).backward()
            sum(v.square().mean() for v in observed).backward()
            torch.testing.assert_close(expected_input.grad, actual_input.grad, rtol=1e-12, atol=1e-12)
            for old, new in zip(reference.parameters(), actual.parameters()):
                self.assertIsNotNone(new.grad)
                torch.testing.assert_close(old.grad, new.grad, rtol=1e-12, atol=1e-12)
            with torch.no_grad():
                actual[1].q1[-1].bias.add_(1)
            updated = module.critic_values(actual, state, actual_input, True, per_critic)
            torch.testing.assert_close(updated[0][:, 1], observed[0][:, 1] + 1, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(updated[0][:, 0], observed[0][:, 0], rtol=0, atol=0)

    def test_both_complete_updates_match_legacy_with_fp32_tolerance(self):
        torch.set_num_threads(1)
        check_vectorized_learners()


if __name__ == "__main__":
    unittest.main()
