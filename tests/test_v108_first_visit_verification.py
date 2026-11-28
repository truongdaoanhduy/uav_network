"""V108 pure-unittest regression: hygiene fixes and first_visit runtime verification."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))
sys.path.insert(0, str(ROOT / "scripts/maintenance"))

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params
from v107_ds_first_passage import passage


class TestV108FirstVisitAndHygiene(unittest.TestCase):
    BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
    STRESS_YAML = ROOT / "research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml"

    def test_both_active_yamls_use_ds_only_first_visit_and_fixed_tx(self):
        for yml in (self.BASE_YAML, self.STRESS_YAML):
            cfg = load_scenario(yml)
            self.assertEqual(cfg.belief_inference_mode, "dempster_shafer")
            self.assertEqual(cfg.exploration_reward_mode, "first_visit")
            self.assertAlmostEqual(cfg.tx_power_w, 0.1)
            self.assertFalse(cfg.enable_tx_power_control)
            self.assertTrue(cfg.obs_schema_v2)
            p = GPUWorldV79Params.from_yaml(yml, envs=2, device="cpu", seed=44)
            self.assertEqual(p.exploration_reward_mode, "first_visit")

    def test_cpu_first_visit_eliminates_hover_farming(self):
        cfg = load_scenario(self.BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=44)
        env.reset(seed=44)
        a = np.zeros((6, 2), dtype=np.float32)
        d = np.zeros(6, dtype=np.int64)
        *_, info1 = env.step(a, d)
        *_, info2 = env.step(a, d)
        self.assertGreater(float(info1["r_loc_team"]), 0.0)
        self.assertEqual(float(info2["r_loc_team"]), 0.0)

    def test_v107_first_passage_script_creates_parent_dir_and_matches_math(self):
        src = (ROOT / "scripts/maintenance/v107_ds_first_passage.py").read_text(encoding="utf-8")
        self.assertIn("out.parent.mkdir(parents=True, exist_ok=True)", src)
        old_900 = passage(0.03, 0.90)["900"]
        new_900 = passage(0.03, 0.99)["900"]
        self.assertAlmostEqual(passage(0.03, 0.90, 2)["2"], 0.03**2, places=9)
        self.assertLess(new_900, old_900 / 30.0)
        self.assertGreater(passage(0.85, 0.99)["10"], 0.97)
        self.assertEqual(passage(0.03, 0.99, 2)["2"], 0.0)


if __name__ == "__main__":
    unittest.main()
