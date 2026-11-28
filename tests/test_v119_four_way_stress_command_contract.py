"""Red/green V119 CLI gate: do not accidentally test only one stress pattern."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v119_four_arms', ROOT / 'scripts/gpu/v119_four_way_cuda_gate.py')
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


class TestV119FourWayStressCommandContract(unittest.TestCase):
    def test_four_distinct_scenario_mode_cases_and_required_output(self):
        commands = mod.build_commands(Path('outputs/sample'), envs=4096, steps=900, device='cuda:0')
        self.assertEqual(list(commands), ['base_random', 'rf_random', 'rf_boundary', 'rf_churn'])
        for name, argv in commands.items():
            self.assertIn('--output', argv)
            self.assertEqual(argv[argv.index('--output') + 1], f'outputs/sample/{name}.json')
            self.assertEqual(argv[argv.index('--envs') + 1], '4096')
            self.assertEqual(argv[argv.index('--steps') + 1], '900')
        self.assertEqual([commands[k][commands[k].index('--mode') + 1] for k in commands],
                         ['random_exploration', 'random_exploration', 'boundary_push', 'command_churn'])
        self.assertEqual([commands[k][commands[k].index('--scenario') + 1].split('/')[-1] for k in commands],
                         ['scenario_v103_ds_only_1km.yaml'] + ['scenario_v111_ds_rf_stress_no_peer_1km.yaml'] * 3)

    def test_dry_run_does_not_depend_on_cuda(self):
        self.assertEqual(mod.run(['--output-dir', 'outputs/nonexistent', '--dry-run']), 0)


if __name__ == '__main__':
    unittest.main()
