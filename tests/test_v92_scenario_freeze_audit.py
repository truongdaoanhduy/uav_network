"""V92 scenario-first freeze contracts: non-oracle frontier controller, blackout exit, YAML freeze checks."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/gpu")]

from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params

NON_ORACLE_SCRIPT = ROOT / "scripts/gpu/v92_non_oracle_mission_probe.py"
NET_PEER_SCRIPT = ROOT / "scripts/gpu/v92_network_and_peer_probe.py"
SCENARIO_V91 = ROOT / "research/zhang_xue/configs/scenario_v91_bayes_1km.yaml"
SCENARIO_V89 = ROOT / "research/zhang_xue/configs/scenario_v89_static_1km.yaml"


class TestV92ScenarioFreezeAudit(unittest.TestCase):
    def test_v92_non_oracle_frontier_steers_to_high_uncertainty_and_respects_mask(self):
        obs = np.zeros((1, 6, 549), dtype=np.float32)
        mask = np.zeros((1, 6, 3), dtype=np.uint8)
        obs[0, :, 0] = 0.5  # x=500m
        obs[0, :, 1] = 0.5  # y=500m
        obs[0, :, 5] = 1.0  # radio alive
        # Put highest uncertainty at block (cx=9, cy=9) -> bid=99 -> (950m, 950m)
        obs[0, 0, 214 + 99] = 1.0
        mask[0, 0] = [1, 1, 1]  # GCS available -> should choose 1
        mask[0, 1] = [1, 0, 1]  # Peer only -> should choose 2
        mask[0, 2] = [1, 0, 0]  # Silent only -> should choose 0
        cont, dest = non_oracle_frontier_actions_np(obs, mask)
        self.assertGreater(float(cont[0, 0, 0]), 0.5)
        self.assertGreater(float(cont[0, 0, 1]), 0.5)
        self.assertEqual(int(dest[0, 0]), 1)
        self.assertEqual(int(dest[0, 1]), 2)
        self.assertEqual(int(dest[0, 2]), 0)

    def test_v92_non_oracle_frontier_exits_blackout_toward_gcs_when_dirty(self):
        obs = np.zeros((1, 6, 549), dtype=np.float32)
        mask = np.zeros((1, 6, 3), dtype=np.uint8)
        obs[0, 0, 0] = 0.8  # x=800m
        obs[0, 0, 1] = 0.8  # y=800m
        obs[0, 0, 5] = 0.0  # radio disconnected (inside blackout)
        obs[0, 0, 11] = 0.05  # dirty GCS evidence >= 0.015
        obs[0, 0, 214 + 99] = 1.0  # high uncertainty further away at (950, 950)
        cont, dest = non_oracle_frontier_actions_np(obs, mask)
        # Must head back toward GCS (50, 50) -> negative vx, negative vy
        self.assertLess(float(cont[0, 0, 0]), -0.5)
        self.assertLess(float(cont[0, 0, 1]), -0.5)
        self.assertEqual(int(dest[0, 0]), 0)

    def test_v92_v91_bayesian_and_v89_ds_yaml_contracts_and_500_line_policy(self):
        p91 = GPUWorldV80Params.from_yaml(SCENARIO_V91, envs=2, device="cpu", seed=44)
        p89 = GPUWorldV80Params.from_yaml(SCENARIO_V89, envs=2, device="cpu", seed=44)
        self.assertTrue(p91.static_raw_count and p91.obs_schema_v2)
        self.assertTrue(p89.static_raw_count and p89.obs_schema_v2)
        self.assertEqual(p91.uav_safe_distance_m, 50.0)
        self.assertEqual(p91.v_max_mps, 15.0)
        self.assertEqual(p91.a_max_mps2, 5.0)
        self.assertEqual(p91.dt_s, 1.0)
        for path in (NON_ORACLE_SCRIPT, NET_PEER_SCRIPT):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")


if __name__ == "__main__":
    unittest.main()
