"""Calibration must exercise the real CPU action units and state its limits."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YAML = str(ROOT / 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml')


def load_script(name):
    path = ROOT / 'scripts/maintenance' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CalibrationContracts(unittest.TestCase):
    def test_patrol_commands_use_cpu_metres_per_second(self):
        from zhang_xue.environment.core import ZhangXue7NodePacketEnv
        for name in ('v120_ds_correlated_mission_calibration',
                     'v120_cpu_rf_substeps_energy_calibration'):
            mod = load_script(name)
            env = ZhangXue7NodePacketEnv(mod._load_cfg(YAML))
            env.reset(seed=44)
            command = mod._sector_patrol_xy(env, 0)
            np.testing.assert_allclose(np.linalg.norm(command, axis=1),
                                       env.cfg.v_max_mps, rtol=1e-6)
            before = env.positions[:6, :2].copy()
            env.step(command, np.zeros(6, dtype=np.int32))
            # From rest, a_max=5m/s^2 permits5m in the first one-second step.
            np.testing.assert_allclose(np.linalg.norm(env.positions[:6, :2] - before, axis=1),
                                       5.0, atol=1e-5)

    def test_idealized_energy_estimate_cannot_certify_mission(self):
        mod = load_script('v120_cpu_rf_substeps_energy_calibration')
        result = mod.compute_c08_c12_energy_upper_bound(YAML, [44])
        self.assertIs(result.get('mission_feasibility_proven'), False)
        self.assertIn('obstacle_avoidance', result.get('ignored_constraints', []))
        self.assertIn('network_delivery', result.get('ignored_constraints', []))


if __name__ == '__main__':
    unittest.main()
