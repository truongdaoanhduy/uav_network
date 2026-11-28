"""V111 pure-unittest verification for C19 (549/3 ABI), C20 (unconfounded YAMLs), C27 (non-oracle taint), and 0.90==0.95."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))

from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_stable
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params
from zhang_xue.learners.actor_adapter import (
    ACTOR_POLICY_DIM_V2,
    ALIVE_FEATURE_INDEX,
    RAW_DIM_V2,
    SOURCE_SOC_INDEX,
    encode_actor_input_scalable,
    make_critic_state_scalable,
)


class TestV111AdapterTaintAndFactorial(unittest.TestCase):
    CFG_DIR = ROOT / "research/zhang_xue/configs"
    BASE_YAML = CFG_DIR / "scenario_v103_ds_only_1km.yaml"
    STRESS_YAML = CFG_DIR / "scenario_v105_ds_rf_stress_1km.yaml"
    RF_NO_PEER = CFG_DIR / "scenario_v111_ds_rf_stress_no_peer_1km.yaml"
    RF_PEER_ONLY = CFG_DIR / "scenario_v111_ds_rf_stress_peer_only_1km.yaml"
    BASE_PEER_ONLY = CFG_DIR / "scenario_v111_ds_base_peer_only_1km.yaml"
    TH099 = CFG_DIR / "scenario_v111_ds_ablation_th099_1km.yaml"

    def test_c19_scalable_adapter_accepts_active_549_3_and_preserves_blackout_alive(self):
        cfg = load_scenario(self.BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=44)
        obs, _state, mask = env.reset(seed=44)
        self.assertEqual(obs.shape, (6, RAW_DIM_V2))
        self.assertEqual(mask.shape, (6, 3))

        # Simulate UAV 0 inside RF blackout (obs[5]=0) while still having battery (obs[4]>0)
        obs_blackout = obs.copy()
        obs_blackout[0, SOURCE_SOC_INDEX] = 0.85
        obs_blackout[0, 5] = 0.0
        obs_blackout[0, ALIVE_FEATURE_INDEX] = 0.42  # normalized GCS distance

        enc = encode_actor_input_scalable(obs_blackout, mask)
        self.assertEqual(enc.shape, (6, ACTOR_POLICY_DIM_V2))  # 553
        self.assertEqual(float(enc[0, ALIVE_FEATURE_INDEX]), 1.0)
        self.assertAlmostEqual(float(enc[0, RAW_DIM_V2]), 0.42, places=6)
        np.testing.assert_array_equal(enc[:, -3:], mask.astype(np.float32))

        critic_state = make_critic_state_scalable(enc)
        self.assertEqual(critic_state.shape, (6 * ACTOR_POLICY_DIM_V2,))  # 3318

    def test_c27_non_oracle_taint_mutating_unseen_targets_leaves_obs_and_state_identical(self):
        cfg = load_scenario(self.BASE_YAML)
        env_a = ZhangXue7NodePacketEnv(cfg, seed=44)
        env_b = ZhangXue7NodePacketEnv(cfg, seed=44)
        obs_a, st_a, mask_a = env_a.reset(seed=44)
        obs_b, st_b, mask_b = env_b.reset(seed=44)

        # Flip ground-truth target flags in far-away cells (>250m from all UAVs)
        far_cells = []
        for cid in range(400):
            cx, cy = (cid % 20 + 0.5) * 50.0, (cid // 20 + 0.5) * 50.0
            if all(np.hypot(cx - p[0], cy - p[1]) > 250.0 for p in env_b.positions[:6]):
                far_cells.append(cid)
        self.assertGreater(len(far_cells), 20)
        env_b.true_targets[far_cells] = ~env_b.true_targets[far_cells]

        a = np.zeros((6, 2), dtype=np.float32)
        d = np.ones(6, dtype=np.int64)
        out_a = env_a.step(a, d)
        out_b = env_b.step(a, d)
        np.testing.assert_array_equal(out_a[0], out_b[0])  # raw obs 549
        np.testing.assert_array_equal(out_a[1], out_b[1])  # state
        np.testing.assert_array_equal(out_a[2], out_b[2])  # mask 3
        np.testing.assert_array_equal(
            encode_actor_input_scalable(out_a[0], out_a[2]),
            encode_actor_input_scalable(out_b[0], out_b[2]),
        )

    def test_c20_unconfounded_factorial_and_ablation_yamls_single_variable_diffs(self):
        base = load_scenario(self.BASE_YAML)
        rf_np = load_scenario(self.RF_NO_PEER)
        rf_p = load_scenario(self.RF_PEER_ONLY)
        rf_pm = load_scenario(self.STRESS_YAML)
        b_p = load_scenario(self.BASE_PEER_ONLY)
        th99 = load_scenario(self.TH099)

        # Pure RF noise isolation: base -> rf_np
        self.assertAlmostEqual(rf_np.mac_noise_power_w, 4.0e-11)
        self.assertFalse(rf_np.enable_peer_dissemination)
        self.assertFalse(rf_np.enable_data_mule_forwarding)

        # Pure PEER isolation under RF-stress: rf_np -> rf_p
        self.assertAlmostEqual(rf_p.mac_noise_power_w, 4.0e-11)
        self.assertTrue(rf_p.enable_peer_dissemination)
        self.assertFalse(rf_p.enable_data_mule_forwarding)

        # Pure Data-Mule isolation under RF-stress: rf_p -> rf_pm (v105)
        self.assertAlmostEqual(rf_pm.mac_noise_power_w, 4.0e-11)
        self.assertTrue(rf_pm.enable_peer_dissemination)
        self.assertTrue(rf_pm.enable_data_mule_forwarding)

        # Pure PEER isolation under Open-RF: base -> b_p
        self.assertAlmostEqual(b_p.mac_noise_power_w, base.mac_noise_power_w)
        self.assertTrue(b_p.enable_peer_dissemination)
        self.assertFalse(b_p.enable_data_mule_forwarding)

        # Pure confirm_th=0.99 isolation: base -> th99
        self.assertAlmostEqual(th99.confirm_th, 0.99)
        self.assertFalse(th99.enable_peer_dissemination)

        # All 4 new YAMLs validate cleanly on Warp GPUWorldV79Params
        for yml in (self.RF_NO_PEER, self.RF_PEER_ONLY, self.BASE_PEER_ONLY, self.TH099):
            p = GPUWorldV79Params.from_yaml(yml, envs=2, device="cpu", seed=44)
            self.assertEqual(p.exploration_reward_mode, "first_visit")

    def test_ds_threshold_090_and_095_are_algebraically_equivalent_for_integer_counts(self):
        for p in range(0, 40):
            for n in range(0, 40):
                mt, _mf, mu = counts_to_canonical_ds_stable(float(p), float(n), 0.80, 0.80)
                if mu <= 0.20:
                    self.assertEqual(mt >= 0.90, mt >= 0.95, f"Mismatch at p={p}, n={n}, mt={mt}")


if __name__ == "__main__":
    unittest.main()
