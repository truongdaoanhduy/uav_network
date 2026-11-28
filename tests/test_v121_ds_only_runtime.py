"""DS-only configuration boundaries for the CPU production environment."""
from pathlib import Path
import unittest

import numpy as np

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "research/zhang_xue/configs"
BASE = CONFIGS / "scenario_v103_ds_only_1km.yaml"
RF_STRESS = CONFIGS / "scenario_v105_ds_rf_stress_1km.yaml"


class TestV121DsOnlyRuntime(unittest.TestCase):
    def test_dataclass_rejects_bayesian_backend(self):
        with self.assertRaisesRegex(ValueError, "belief_inference_mode"):
            EnvConfigV54(
                belief_inference_mode="bayesian_logodds",
                static_evidence_mode="raw_u16_count", ds_aoi_eta=0.0,
            )

    def test_active_yaml_cannot_enable_bayes_through_cli_override(self):
        for path in (BASE, RF_STRESS):
            with self.subTest(scenario=path.name):
                with self.assertRaisesRegex(ValueError, "belief_inference_mode"):
                    load_scenario(path, ("belief_inference_mode=bayesian_logodds",))

    def test_historical_bayesian_yamls_are_rejected_at_cpu_boundary(self):
        for filename in (
            "scenario_v91_bayes_1km.yaml",
            "scenario_v95_bayes_shared_coverage_1km.yaml",
        ):
            with self.subTest(scenario=filename):
                with self.assertRaisesRegex(ValueError, "belief_inference_mode"):
                    load_scenario(CONFIGS / filename)

    def test_legacy_prior_field_does_not_change_ds_observations_or_transitions(self):
        for path in (BASE, RF_STRESS):
            with self.subTest(scenario=path.name):
                cfg_a = load_scenario(path)
                cfg_b = load_scenario(path, ("bayes_prior_occupancy=0.9",))
                env_a = ZhangXue7NodePacketEnv(cfg_a)
                env_b = ZhangXue7NodePacketEnv(cfg_b)
                for a, b in zip(env_a.reset(seed=44), env_b.reset(seed=44)):
                    np.testing.assert_array_equal(a, b)
                velocity = np.zeros((6, 2), dtype=np.float64)
                communication = np.ones(6, dtype=np.int32)
                for _ in range(3):
                    result_a = env_a.step(velocity, communication)
                    result_b = env_b.step(velocity, communication)
                    for a, b in zip(result_a[:3], result_b[:3]):
                        np.testing.assert_array_equal(a, b)
                    self.assertEqual(result_a[3:], result_b[3:])


if __name__ == "__main__":
    unittest.main()
