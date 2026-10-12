"""Numerical safety remains active between scalar-metric logging intervals."""
import unittest
import torch

from uav_network.algorithms.optimization import step_optimizer


class OptimizationTests(unittest.TestCase):
    def test_nonfinite_gradient_is_rejected_before_adam_when_metrics_are_disabled(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.Adam([parameter], lr=.01)
        with self.assertRaises(RuntimeError):
            step_optimizer(parameter.sqrt(), optimizer, [parameter], 10, check_finite=False)
        self.assertEqual(parameter.item(), 0.)
        self.assertEqual(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
