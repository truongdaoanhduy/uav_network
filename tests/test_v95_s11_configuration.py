"""Regression for the V94 hidden S11 YAML-contract defect (no CUDA required)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params

CFG = ROOT / "research/zhang_xue/configs"

class TestV95SharedCoverageContract(unittest.TestCase):
    def test_original_explicitly_disabled(self):
        path = CFG / "scenario_v91_bayes_1km.yaml"
        scene = load_scenario(path)
        self.assertIs(scene.shared_coverage_reward, False)
        params = GPUWorldV79Params.from_yaml(path, envs=1, seed=44)
        self.assertIs(params.shared_coverage_reward, False)
        self.assertTrue(params.bayesian_count_mode)

    def test_ds_bayes_pair_loads_true_through_runtime(self):
        for suffix, expected_bayes in (("ds", False), ("bayes", True)):
            path = CFG / f"scenario_v95_{suffix}_shared_coverage_1km.yaml"
            scene = load_scenario(path)
            self.assertIs(scene.shared_coverage_reward, True)
            params = GPUWorldV79Params.from_yaml(path, envs=1, seed=44)
            self.assertIs(params.shared_coverage_reward, True)
            self.assertIs(params.bayesian_count_mode, expected_bayes)
            self.assertEqual((params.pd, params.pf, params.confirm_th), (.85, .03, .9))

    def test_typed_override_true_and_bad_values_fail_closed(self):
        path = CFG / "scenario_v91_bayes_1km.yaml"
        self.assertTrue(load_scenario(path, ("shared_coverage_reward=true",)).shared_coverage_reward)
        for bad in ("1", "0", '"true"', "null"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_scenario(path, (f"shared_coverage_reward={bad}",))
        params = GPUWorldV79Params(shared_coverage_reward="true")
        with self.assertRaisesRegex(ValueError, "shared_coverage_reward"):
            params.validate()

if __name__ == "__main__":
    unittest.main()
