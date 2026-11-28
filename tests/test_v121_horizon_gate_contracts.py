"""A reset-based or partially observed run cannot certify a900-step flight."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/gpu/v121_uninterrupted_cuda_gate.py'


class HorizonGateContracts(unittest.TestCase):
    def decision(self, **changes):
        spec = importlib.util.spec_from_file_location('v121_gate', PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        row = dict(device_is_cuda=True, envs=4096, requested_steps=900,
                   observed_steps=900, reset_calls=0, min_episode_steps=900,
                   first_terminal_min_step=900, terminal_worlds=4096,
                   ever_safety_failure_worlds=0, ever_overflow_worlds=0,
                   nonfinite_worlds=0, energy_increase_worlds=0, physical_violation_worlds=0,
                   max_reward_sum_error=0.0, max_speed_excess=0.0,
                   source_sha256='a' *64, scenario_sha256='b' *64)
        row.update(changes)
        return mod.evaluate_run(row)

    def test_complete_nonreset_cuda_run_is_accepted(self):
        self.assertEqual(self.decision(), [])

    def test_reset_or_early_terminal_or_cpu_never_certifies_full_flight(self):
        for change in (dict(reset_calls=1), dict(first_terminal_min_step=500),
                       dict(min_episode_steps=899), dict(observed_steps=899),
                       dict(device_is_cuda=False), dict(ever_safety_failure_worlds=1),
                       dict(energy_increase_worlds=1), dict(max_speed_excess=.01),
                       dict(source_sha256=''), dict(terminal_worlds=4095)):
            with self.subTest(change=change):
                self.assertTrue(self.decision(**change))

    def test_small_smoke_cannot_certify_4096x900(self):
        self.assertTrue(self.decision(envs=4, requested_steps=8, observed_steps=8,
                                      min_episode_steps=8, terminal_worlds=0,
                                      first_terminal_min_step=0))


if __name__ == '__main__':
    unittest.main()
