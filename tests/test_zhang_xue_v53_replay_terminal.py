"""Real CPU pilot must bootstrap from an artificial battery-mode time cap."""
import tempfile
import unittest
from pathlib import Path

import yaml
from uav_network.experiments.train_zhang_xue_pilot import run_cpu_pilot


class ReplayTerminalTest(unittest.TestCase):
    def _run(self, **cfg):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pilot.yaml'
            p.write_text(yaml.safe_dump({'termination_regime':'physical_battery',
                                        'horizon_steps':1, **cfg}))
            return run_cpu_pilot(episodes=1, eval_episodes=1,
                                 learning_starts=1000, config_path=p)

    def test_time_cap_ends_episode_without_zeroing_bootstrap(self):
        r = self._run()
        self.assertEqual(r['total_steps'], 1)
        self.assertEqual(r['truncated_train_episodes'], 1)
        self.assertEqual(r['replay_terminal_rows'], 0)

    def test_depleted_battery_zeroes_bootstrap_even_at_time_cap(self):
        r = self._run(usable_battery_e_max_j=1.)
        self.assertEqual(r['replay_terminal_rows'], 1)
        self.assertEqual(r['truncated_train_episodes'], 0)


if __name__ == '__main__':
    unittest.main()
