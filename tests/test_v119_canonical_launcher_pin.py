"""Red/green V119 regression: canonical launcher cannot silently accept modified scenarios."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/gpu/run_v115_canonical_ds_baseline.py"
spec = importlib.util.spec_from_file_location("v119_launcher_under_test", MODULE)
launcher = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(launcher)


def args(scenario=None, learner=None):
    return Namespace(
        algorithm="matd3", scenario=scenario or launcher.DEFAULT_SCENARIO,
        learner_config=learner or launcher.DEFAULT_LEARNER_CFG,
        envs=64, steps=300, batch_size=256, replay_capacity=65536,
        world_device="cuda:0", learner_device="cuda:0", seed=44,
        eval_episodes=0, allow_experimental_physics=True,
        output=Path("outputs/v119_test.json"),
    )


class TestV119CanonicalLauncherPin(unittest.TestCase):
    def test_missing_physics_ack_is_blocked_before_training(self):
        obj = args()
        obj.allow_experimental_physics = False
        with self.assertRaisesRegex(ValueError, "--allow-experimental-physics"):
            launcher.build_command(obj)

    def test_unmodified_pinned_yaml_and_baseline_flags(self):
        cmd = launcher.build_command(args())
        self.assertEqual(cmd[cmd.index("--baseline-mode") + 1], "canonical_continuous")
        self.assertEqual(cmd[cmd.index("--replay-action-mode") + 1], "raw")
        self.assertEqual(cmd[cmd.index("--learner-config") + 1], str(launcher.DEFAULT_LEARNER_CFG))
        self.assertIn("--allow-experimental-physics", cmd)

    def test_rf_stress_yaml_cannot_masquerade_as_canonical(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            launcher.build_command(args(scenario=ROOT / "research/zhang_xue/configs/scenario_v111_ds_rf_stress_no_peer_1km.yaml"))

    def test_same_ds_yaml_but_peer_enabled_cannot_masquerade(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "scenario.yaml"
            contents = launcher.DEFAULT_SCENARIO.read_text()
            path.write_text(contents.replace("enable_peer_dissemination: false", "enable_peer_dissemination: true"))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                launcher.build_command(args(scenario=path))

    def test_same_ds_yaml_but_bayes_or_reward_edited_cannot_masquerade(self):
        with TemporaryDirectory() as td:
            for old, new in (("dempster_shafer", "bayesian_logodds"),
                             ("first_visit", "ds_ignorance")):
                path = Path(td) / "scenario.yaml"
                path.write_text(launcher.DEFAULT_SCENARIO.read_text().replace(old, new))
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    launcher.build_command(args(scenario=path))

    def test_modified_learner_yaml_cannot_masquerade(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "learner.yaml"
            path.write_text(launcher.DEFAULT_LEARNER_CFG.read_text().replace("policy_delay: 2", "policy_delay: 1"))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                launcher.build_command(args(learner=path))


if __name__ == "__main__":
    unittest.main()
