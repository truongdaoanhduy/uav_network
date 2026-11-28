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
        path = CFG / "scenario_v103_ds_only_1km.yaml"
        scene = load_scenario(path)
        self.assertIs(scene.shared_coverage_reward, False)
        params = GPUWorldV79Params.from_yaml(path, envs=1, seed=44)
        self.assertIs(params.shared_coverage_reward, False)

    def test_ds_shared_coverage_loads_and_bayes_fails_closed(self):
        ds_path = CFG / "scenario_v95_ds_shared_coverage_1km.yaml"
        scene = load_scenario(ds_path)
        self.assertIs(scene.shared_coverage_reward, True)
        params = GPUWorldV79Params.from_yaml(ds_path, envs=1, seed=44)
        self.assertIs(params.shared_coverage_reward, True)
        self.assertEqual((params.pd, params.pf, params.confirm_th), (.85, .03, .9))
        # V103+ DS-only contract: Warp rejects historical Bayesian YAML fail-closed
        with self.assertRaises(ValueError):
            GPUWorldV79Params.from_yaml(CFG / "scenario_v95_bayes_shared_coverage_1km.yaml", envs=1, seed=44)

    def test_typed_override_true_and_bad_values_fail_closed(self):
        path = CFG / "scenario_v103_ds_only_1km.yaml"
        # V107 first_visit and V95 shared_coverage are mutually exclusive.
        # An isolated override must fail closed; a paired legacy-ablation override works.
        with self.assertRaisesRegex(ValueError, "Cannot combine shared_coverage_reward"):
            load_scenario(path, ("shared_coverage_reward=true",))
        legacy = load_scenario(path, (
            "shared_coverage_reward=true", "exploration_reward_mode=ds_ignorance",
        ))
        self.assertTrue(legacy.shared_coverage_reward)
        self.assertEqual(legacy.exploration_reward_mode, "ds_ignorance")
        for bad in ("1", "0", '"true"', "null"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_scenario(path, (f"shared_coverage_reward={bad}",))
        params = GPUWorldV79Params(shared_coverage_reward="true")
        with self.assertRaisesRegex(ValueError, "shared_coverage_reward"):
            params.validate()

if __name__ == "__main__":
    unittest.main()
