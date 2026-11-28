"""V118 pure-unittest contract: clean canonical 2D scenario free of V117 unproven 96-combo discrete overlay."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import (
    ACTOR_POLICY_DIM_V2,
    RAW_DIM_V2,
    encode_actor_input_scalable,
)


class TestV118CleanCanonicalContract(unittest.TestCase):
    BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
    RF_ONLY_YAML = ROOT / "research/zhang_xue/configs/scenario_v111_ds_rf_stress_no_peer_1km.yaml"

    def test_canonical_env_has_no_unproven_v117_dual_action_overlay(self):
        cfg = load_scenario(self.BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=44)
        sig = inspect.signature(env.step)
        self.assertNotIn("evidence_actions", sig.parameters)
        self.assertNotIn("relay_actions", sig.parameters)
        self.assertFalse(hasattr(env, "experimental_network_action_view"))

        obs, _state, mask = env.reset(seed=44)
        self.assertEqual(obs.shape, (6, RAW_DIM_V2))  # 549
        self.assertEqual(mask.shape, (6, 3))
        enc = encode_actor_input_scalable(obs, mask)
        self.assertEqual(enc.shape, (6, ACTOR_POLICY_DIM_V2))  # 553

    def test_pure_gcs_canonical_profiles_disable_peer_mule_and_fix_tx_power(self):
        for yml in (self.BASE_YAML, self.RF_ONLY_YAML):
            cfg = load_scenario(yml)
            self.assertEqual(cfg.belief_inference_mode, "dempster_shafer")
            self.assertEqual(cfg.exploration_reward_mode, "first_visit")
            self.assertFalse(cfg.enable_peer_dissemination)
            self.assertFalse(cfg.enable_data_mule_forwarding)
            self.assertFalse(cfg.enable_tx_power_control)
            self.assertAlmostEqual(cfg.tx_power_w, 0.1)


if __name__ == "__main__":
    unittest.main()
