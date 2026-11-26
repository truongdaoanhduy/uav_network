"""Independent governance and scientific-config regression for v5.9."""
from __future__ import annotations
import ast
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork as ModularNetwork
from zhang_xue.environment.core import ZhangXue7NodePacketEnv as ModularEnv
from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_stable
from zhang_xue.compat.network import SevenRadioCsmaCaNetwork
from zhang_xue.compat.env import ZhangXue7NodePacketEnv
from zhang_xue.compat.belief_mobility import counts_to_canonical_ds_stable as LegacyDS


class V59ArchitectureTests(unittest.TestCase):
    def test_backward_compatible_imports(self):
        self.assertIs(ModularNetwork, SevenRadioCsmaCaNetwork)
        self.assertIs(ModularEnv, ZhangXue7NodePacketEnv)
        self.assertIs(LegacyDS, counts_to_canonical_ds_stable)

    def test_modular_source_each_at_most_500_lines(self):
        for path in SRC.rglob("*.py"):
            with self.subTest(file=path.relative_to(SRC)):
                self.assertLessEqual(len(path.read_text().splitlines()), 500)
                ast.parse(path.read_text())

    def test_yaml_matches_validated_dataclass(self):
        cfg = load_scenario(ROOT / "configs/scenario_7radio.yaml")
        self.assertEqual((cfg.map_size_m, cfg.num_targets), (1000., 15))
        self.assertEqual(cfg.safety_mode, "kinematic_shield")
        other = load_scenario(overrides=("num_targets=4", "enable_accel_energy=false"))
        self.assertEqual(other.num_targets, 4)
        self.assertFalse(other.enable_accel_energy)
        self.assertIn("confirm_th", asdict(cfg))

    def test_unknown_or_bad_type_are_rejected(self):
        for overrides in [("num_targts=12",), ("num_targets=true",), ("pd='0.2'",),
                          ("safety_failure_terminate=1",), ("num_targets=[1,2]",)]:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    load_scenario(overrides=overrides)

    def test_study_three_km_profile_cannot_silently_run_20x20_kernel(self):
        with self.assertRaises(ValueError):
            load_scenario(overrides=("map_size_m=3000.0", "num_targets=35"))

    def test_configurable_values_do_not_mutate_previous_defaults(self):
        a = load_scenario(overrides=("num_targets=7",))
        b = load_scenario()
        self.assertEqual(a.num_targets, 7)
        self.assertEqual(b.num_targets, EnvConfigV54(**asdict(b)).num_targets)
        self.assertEqual(b.num_targets, 15)


if __name__ == "__main__":
    unittest.main()
