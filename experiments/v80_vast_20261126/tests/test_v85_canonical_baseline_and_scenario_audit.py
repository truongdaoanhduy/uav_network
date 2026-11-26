"""V85 tests: canonical MATD3/MASAC baseline contracts, event-driven comm, and scenario physics audit."""
import ast
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/gpu")]

from rollout_v82_contracts import configure_learner
from uav_network.algorithms.baselines.canonical_continuous import event_driven_comm_action_np
from uav_network.algorithms.configuration.loader import load_learner_config
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params
from zhang_xue.mobility.flight_safety_energy import ZengBatteryConfig, zeng_propulsion_power_w

CANONICAL_MOD = ROOT / "uav_network/algorithms/baselines/canonical_continuous.py"
CANONICAL_YAML = ROOT / "configs/learners/v85_canonical_matd3_masac_baseline.yaml"
RUNNER = ROOT / "scripts/gpu/run_v81_real_rollout.py"
CONTRACTS = ROOT / "scripts/gpu/rollout_v82_contracts.py"
SCENARIO_V77 = ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml"


class _DummyArgs:
    def __init__(self, **kw):
        defaults = dict(
            algorithm="matd3", envs=64, steps=100, updates_per_step=1, batch_size=128,
            replay_capacity=4096, learning_starts=1024, log_interval=10, hidden=None,
            allow_no_updates=False, replay_action_mode="raw", shield_penalty_lambda=0.0,
            baseline_mode="canonical_continuous",
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestV85CanonicalBaselineAndScenarioAudit(unittest.TestCase):
    def test_v85_event_driven_comm_priority_zero_oracle(self):
        """GCS (1) > PEER (2) > SILENT (0), driven strictly by local observation tail mask."""
        masks = np.array([
            [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0]],
        ], dtype=np.float32)
        dest = event_driven_comm_action_np(masks)
        np.testing.assert_array_equal(dest, np.array([[1, 1, 2, 0]], dtype=np.int64))
        # Also works when passed full 518-dim observation whose last 3 dims are the mask
        obs = np.zeros((1, 4, 518), dtype=np.float32)
        obs[..., -3:] = masks
        obs[..., 10] = 1.0  # GCS route genuinely available in this fixture
        np.testing.assert_array_equal(event_driven_comm_action_np(obs), np.array([[1, 1, 2, 0]], dtype=np.int64))

    def test_v85_canonical_yaml_disables_all_custom_modifications(self):
        cfg = load_learner_config(CANONICAL_YAML)
        self.assertEqual(cfg.network.power_control, "fixed")
        self.assertEqual(cfg.continuous_dim, 2)
        for alg in ("matd3", "masac"):
            hp = getattr(cfg, alg)
            self.assertEqual(hp.activation, "relu")
            self.assertFalse(hp.actor_layer_norm)
            self.assertFalse(hp.critic_layer_norm)
            self.assertFalse(hp.actor_batch_norm)
            self.assertFalse(hp.critic_batch_norm)
            self.assertEqual(hp.critic_huber_delta, 0.0)
            self.assertFalse(hp.sync_critic_loss_scale)
            configure_learner(load_learner_config(CANONICAL_YAML), _DummyArgs(algorithm=alg))
        self.assertEqual(cfg.matd3.pre_tanh_reg, 0.0)
        self.assertEqual(cfg.matd3.actor_output_regularization, 0.0)
        self.assertEqual(cfg.matd3.initialization, "default")

    def test_v85_canonical_mode_rejects_custom_replay_or_modified_learner_knobs(self):
        # Reject dual/executed/raw_penalized replay in canonical_continuous mode
        for bad_mode in ("dual", "executed", "raw_penalized"):
            with self.assertRaises(ValueError):
                configure_learner(load_learner_config(CANONICAL_YAML), _DummyArgs(replay_action_mode=bad_mode))
        # Reject customized V81 YAML (SiLU + LayerNorm + Huber + pre_tanh_reg) in canonical_continuous mode
        v81_cfg = load_learner_config(ROOT / "configs/learners/v81_matd3_masac_stable.yaml")
        with self.assertRaises(ValueError):
            configure_learner(v81_cfg, _DummyArgs(algorithm="matd3", baseline_mode="canonical_continuous"))

    def test_v85_scenario_physics_tx_power_and_battery_audit(self):
        """Verifies tx_power is rejected on GPU and RF PA draw is <1% of Zeng 2019 900s budget."""
        with self.assertRaises(ValueError):
            GPUWorldV80Params.from_yaml(
                SCENARIO_V77, envs=1, device="cpu",
                scenario_overrides=("enable_tx_power_control=true",),
            )
        cfg = ZengBatteryConfig()
        e_hover_900 = (zeng_propulsion_power_w(0.0, cfg) + cfg.avionics_w) * 900.0
        e_cruise_900 = (zeng_propulsion_power_w(10.0, cfg) + cfg.avionics_w) * 900.0
        pa_dc_900 = (0.5 + 0.1 / 0.20) * 900.0
        self.assertLess(e_hover_900, 180_000.0)
        self.assertLess(e_cruise_900, 180_000.0)
        self.assertLess(pa_dc_900 / 180_000.0, 0.01)
        # LoS 6 dB range at 4.39e-13 W exceeds full 1000x1000m diagonal (1414.2m)
        g0 = (299_792_458.0 / (4.0 * math.pi * 2.4e9)) ** 2
        d_snr6 = math.sqrt(0.1 * g0 / (4.38957709293155e-13 * (10.0 ** 0.6)))
        self.assertGreater(d_snr6, math.hypot(1000.0, 1000.0))

    def test_v85_files_parse_and_respect_500_line_policy(self):
        for path in (CANONICAL_MOD, RUNNER, CONTRACTS):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")


if __name__ == "__main__":
    unittest.main()
