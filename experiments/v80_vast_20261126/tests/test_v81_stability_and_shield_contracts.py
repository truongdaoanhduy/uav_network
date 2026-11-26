"""V81 regression tests: symmetric LayerNorm/SiLU/Huber/pre_tanh_reg, checkpoint compat, dead-UAV symmetry."""
import ast
from pathlib import Path
import unittest

from uav_network.algorithms.configuration.loader import load_learner_config
from uav_network.algorithms.configuration.schema import LearnerConfig
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params

ROOT = Path(__file__).resolve().parents[1]
V81_LEARNER_YAML = ROOT / "configs/learners/v81_matd3_masac_stable.yaml"
V81_HIGHNOISE_YAML = ROOT / "research/zhang_xue/configs/scenario_v81_nfz_multi_highnoise_1km.yaml"
MODELS_PY = ROOT / "uav_network/algorithms/core/models.py"
SCHEMA_PY = ROOT / "uav_network/algorithms/configuration/schema.py"
MATD3_PY = ROOT / "uav_network/algorithms/matd3/learner.py"
MASAC_PY = ROOT / "uav_network/algorithms/masac/learner.py"
SHIELD_PY = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py"


class TestV81StabilityAndShieldContracts(unittest.TestCase):
    def test_v81_stable_learner_yaml_loads_and_validates(self):
        cfg = load_learner_config(V81_LEARNER_YAML)
        self.assertTrue(cfg.matd3.actor_layer_norm)
        self.assertTrue(cfg.matd3.critic_layer_norm)
        self.assertEqual(cfg.matd3.activation, "silu")
        self.assertAlmostEqual(cfg.matd3.pre_tanh_reg, 0.001)
        self.assertAlmostEqual(cfg.matd3.critic_huber_delta, 10.0)
        self.assertTrue(cfg.matd3.sync_critic_loss_scale)
        self.assertTrue(cfg.masac.actor_layer_norm)
        self.assertTrue(cfg.masac.critic_layer_norm)
        self.assertEqual(cfg.masac.activation, "silu")
        self.assertAlmostEqual(cfg.masac.critic_huber_delta, 10.0)

    def test_v81_from_dict_backward_compatible_with_v80_checkpoints(self):
        cfg = LearnerConfig()
        payload = cfg.to_dict()
        for k in ("critic_layer_norm", "activation", "sync_critic_loss_scale", "critic_huber_delta"):
            payload["matd3"].pop(k)
            payload["masac"].pop(k)
        payload["matd3"].pop("pre_tanh_reg")
        restored = LearnerConfig.from_dict(payload)
        self.assertFalse(restored.matd3.critic_layer_norm)
        self.assertEqual(restored.matd3.activation, "relu")
        self.assertEqual(restored.matd3.pre_tanh_reg, 0.0)
        self.assertEqual(restored.masac.critic_huber_delta, 0.0)

    def test_v81_invalid_stability_fields_fail_closed(self):
        cfg = LearnerConfig()
        cfg.matd3.activation = "gelu"
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg2 = LearnerConfig()
        cfg2.matd3.pre_tanh_reg = -0.1
        with self.assertRaises(ValueError):
            cfg2.validate()
        cfg3 = LearnerConfig()
        cfg3.masac.critic_huber_delta = -1.0
        with self.assertRaises(ValueError):
            cfg3.validate()

    def test_v81_highnoise_ablation_yaml_supported_on_gpu(self):
        params = GPUWorldV80Params.from_yaml(V81_HIGHNOISE_YAML, envs=64, device="cuda:0")
        self.assertAlmostEqual(params.noise_power_w, 4.0e-11)
        self.assertAlmostEqual(params.per_building_loss_db, 12.0)

    def test_v81_files_under_500_lines_and_dead_uav_symmetry_present(self):
        for path in (MODELS_PY, SCHEMA_PY, MATD3_PY, MASAC_PY, SHIELD_PY):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")
        s_src = SHIELD_PY.read_text(encoding="utf-8")
        self.assertIn("if battery[e, u] <= 0.0 and battery[e, peer] <= 0.0:", s_src)
        m_src = MODELS_PY.read_text(encoding="utf-8")
        self.assertIn("def pre_tanh_penalty(", m_src)
        self.assertIn("self.layer_norm = bool(layer_norm)", m_src)


if __name__ == "__main__":
    unittest.main()
