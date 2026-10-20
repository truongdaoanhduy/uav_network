"""Fixed-policy communication audit and backend-aware privileged feasibility."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from uav_network.configuration import Config, load_config
from uav_network.environment.env import Action
from uav_network.environment.observations import observation_dim

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
    c.world.max_steps = 2
    c.runtime.algorithm = "matd3"
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.num_envs, c.runtime.episodes = 1, 20000
    c.runtime.eval_seed = 2000000044
    c.matd3.hidden_sizes = [8, 8]
    c.matd3.replay_capacity = 8
    c.matd3.batch_size = c.matd3.learning_starts = 2
    c.network.enabled = False
    return c.validate()


def snapshot(folder, config):
    from uav_network.algorithms.checkpoints import make_learner
    from uav_network.environment.observations import state_dim
    from uav_network.experiments.evaluation_snapshots import save_evaluation_snapshot
    learner = make_learner(config, observation_dim(config), state_dim(config), device="cpu")
    return save_evaluation_snapshot(config, folder, learner.actors.state_dict(), 0,
                                   {"episodes": 0, "global_step": 123})


class RelayAuditTests(unittest.TestCase):
    def test_controls_copy_preserve_movement_and_direct_admission(self):
        from uav_network.experiments.relay_audit import communication_control
        c = small_config()
        continuous = np.array([[.2, -.3, -.4], [.1, .8, .7]], np.float32)
        obs = np.zeros((2, observation_dim(c)), np.float32)
        obs[:, -1] = [1, 0]
        action = Action(continuous.copy(), np.array([1, 1]))
        result = communication_control(action, obs, c, "direct_only")
        np.testing.assert_array_equal(result.destination, [2, 0])
        np.testing.assert_array_equal(result.continuous, continuous)
        result.continuous[0, 0] = 0
        np.testing.assert_array_equal(action.continuous, continuous)
        for destinations in ([0, 2], [2, 0]):
            action.destination = np.array(destinations)
            np.testing.assert_array_equal(
                communication_control(action, obs, c, "direct_only").destination, destinations)
        learned = communication_control(action, obs, c, "learned")
        self.assertIsNot(learned, action)
        self.assertFalse(np.shares_memory(learned.destination, action.destination))

    def test_fixed_power_inverts_existing_mapping_and_rejects_bad_values(self):
        from uav_network.experiments.relay_audit import communication_control
        from uav_network.communication.link_budget import power_from_action
        c = small_config()
        c.network.tx_power_mapping = "log"
        c.network.tx_power_min_w, c.network.tx_power_max_w = .001, .4
        action = Action(np.array([[.2, .3, -.5], [.4, .5, .1]], np.float32), np.array([1, 0]))
        obs = np.zeros((2, observation_dim(c)))
        for power in (c.network.tx_power_min_w, .02, c.network.tx_power_max_w, None):
            result = communication_control(action, obs, c, "fixed_power", power)
            expected = c.network.tx_power_max_w if power is None else power
            for value in result.continuous[:, 2]:
                self.assertAlmostEqual(power_from_action(value, c.network), expected, places=6)
            np.testing.assert_array_equal(result.destination, action.destination)
            np.testing.assert_array_equal(result.continuous[:, :2], action.continuous[:, :2])
        for power in (0, -1, float("nan"), float("inf"), c.network.tx_power_max_w + 1):
            with self.assertRaises(ValueError):
                communication_control(action, obs, c, "fixed_power", power)

    def test_real_pairing_restart_protocol_and_incomplete_recovery(self):
        from uav_network.experiments import relay_audit as audit
        with TemporaryDirectory() as folder:
            c = small_config()
            checkpoint = snapshot(Path(folder) / "policies", c)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = Path(folder) / "audit"
            real = audit.run_episode
            calls = []
            def interrupt(*args, **kwargs):
                calls.append(kwargs["seed"])
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return real(*args, **kwargs)
            with patch.object(audit, "run_episode", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    audit.run_relay_audit(checkpoint, [101, 102], ["learned", "direct_only", "fixed_power"], output_dir=output, trace=True)
            progress = json.loads((output / "summary.json").read_text())
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["completed_cases"], 1)
            files = list((output / "cases").glob("*.json"))
            self.assertEqual(len(files), 1)
            before = files[0].read_bytes()
            # An incomplete case and a leftover atomic-write temporary are not reusable.
            (output / "cases" / "direct_only-101.json").write_text('{"complete": false}')
            (output / "cases" / "learned-102.json.tmp").write_text("{")
            with patch.object(audit, "run_episode", wraps=real) as run:
                result = audit.run_relay_audit(checkpoint, [101, 102], ["learned", "direct_only", "fixed_power"], output_dir=output, trace=True)
            self.assertEqual(run.call_count, 5)
            self.assertEqual(files[0].read_bytes(), before)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_cases"], 6)
            self.assertEqual(result["protocol"]["config"], c.to_dict())
            self.assertEqual(result["protocol"]["training_budget"]["episodes_completed"], 0)
            self.assertFalse(result["protocol"]["training_budget"]["episode_budget_complete"])
            self.assertEqual({r["seed"] for r in result["records"]}, {101, 102})
            for seed in (101, 102):
                paired = [r for r in result["records"] if r["seed"] == seed]
                self.assertEqual(len({r["layout_hash"] for r in paired}), 1)
                self.assertEqual(len({r["return"] for r in paired}), 1)
                for record in paired:
                    self.assertEqual(record["communication_trace"]["events"], [])
                    self.assertIn("not a per-packet", record["communication_trace"]["semantics"])
            for mode in result["by_mode"].values():
                self.assertIsNone(mode["pooled"]["energy/j_per_gcs_delivery"])
                self.assertIsNone(mode["mission_energy_j_per_delivered_report"])
                self.assertIsNone(mode["uav_radio_energy_j_per_delivered_report"])
                self.assertEqual(mode["total_mission_uav_radio_energy_j"], 0)
                self.assertGreater(mode["total_mission_uav_energy_j"], 0)
            with patch.object(audit, "run_episode", wraps=real) as run:
                audit.run_relay_audit(checkpoint, [101, 102], ["learned", "direct_only", "fixed_power"], output_dir=output, trace=True)
                run.assert_not_called()
            with self.assertRaises(ValueError):
                audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output, trace=True)
            other = snapshot(Path(folder) / "other", c)
            with self.assertRaises(ValueError):
                audit.run_relay_audit(other, [101, 102], ["learned", "direct_only", "fixed_power"], output_dir=output, trace=True)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), digest)
            for seeds, modes in (([], ["learned"]), ([1, 1], ["learned"]), ([1], []),
                                 ([1], ["learned", "learned"]), ([1], ["unknown"])):
                with self.assertRaises(ValueError):
                    audit.run_relay_audit(checkpoint, seeds, modes, output_dir=Path(folder) / "bad")

    def test_initial_protocol_sigkill_and_partial_staging_recover(self):
        import signal
        import subprocess
        import sys
        from uav_network.experiments import relay_audit as audit
        with TemporaryDirectory() as folder:
            checkpoint = snapshot(Path(folder) / "policy", small_config())
            output = Path(folder) / "audit"
            child = (
                "import os, signal, sys; from unittest.mock import patch; "
                "from uav_network.experiments import relay_audit as audit; "
                "kill = lambda _: os.kill(os.getpid(), signal.SIGKILL); "
                "context = patch.object(audit.os, 'fsync', side_effect=kill); "
                "context.start(); "
                "audit.run_relay_audit(sys.argv[1], [101], ['learned'], output_dir=sys.argv[2])"
            )
            process = subprocess.run([sys.executable, "-c", child, str(checkpoint), str(output)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, -signal.SIGKILL, process.stderr)
            self.assertFalse((output / "protocol.json").exists())
            staged = list(output.iterdir())
            self.assertEqual(len(staged), 1)
            self.assertTrue(staged[0].name.startswith("protocol.json."))
            # Earlier loss during the write can also leave a partial/empty staging file.
            (output / "protocol.json.abcdefgh.tmp").write_text("{")
            (output / "protocol.json.12345678.tmp").write_text("")
            result = audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
            self.assertTrue(result["complete"])
            self.assertEqual(list(output.glob("protocol.json.*.tmp")), [])

    def test_initial_protocol_recovery_rejects_unknown_or_incompatible_content(self):
        from uav_network.experiments import relay_audit as audit
        with TemporaryDirectory() as folder:
            checkpoint = snapshot(Path(folder) / "policy", small_config())
            for name in ("notes.txt", "protocol.json.unrelated.tmp", "cases"):
                with self.subTest(name=name):
                    output = Path(folder) / name.replace(".", "_")
                    output.mkdir()
                    staging = output / "protocol.json.abcdefgh.tmp"
                    staging.write_text("{")
                    unrelated = output / name
                    if name == "cases":
                        unrelated.mkdir()
                    else:
                        unrelated.write_text("unrelated")
                    with self.assertRaises(ValueError):
                        audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
                    self.assertTrue(staging.exists())
                    self.assertTrue(unrelated.exists())
            output = Path(folder) / "foreign"
            output.mkdir()
            staged = output / "protocol.json.abcdefgh.tmp"
            staged.write_text('{"fingerprint": "another-protocol"}')
            with self.assertRaises(ValueError):
                audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
            self.assertTrue(staged.exists())

    def test_valid_json_malformed_case_roots_recover_but_foreign_complete_case_rejects(self):
        from uav_network.experiments import relay_audit as audit
        with TemporaryDirectory() as folder:
            checkpoint = snapshot(Path(folder) / "policy", small_config())
            output = Path(folder) / "audit"
            audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
            path = output / "cases" / "learned-101.json"
            original_case = json.loads(path.read_text())
            for value in (None, [], 42, "invalid case", True):
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with patch.object(audit, "run_episode", wraps=audit.run_episode) as run:
                        result = audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
                    self.assertEqual(run.call_count, 1)
                    self.assertTrue(result["complete"])
            case = original_case
            case["fingerprint"] = "another-protocol"
            path.write_text(json.dumps(case))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                audit.run_relay_audit(checkpoint, [101], ["learned"], output_dir=output)
            self.assertEqual(path.read_bytes(), before)

    def test_case_shard_runner_reuses_full_protocol_fingerprint_and_matches_full_audit(self):
        from uav_network.experiments import relay_audit as audit
        with TemporaryDirectory() as folder:
            checkpoint = snapshot(Path(folder) / "policy", small_config())
            full = Path(folder) / "full"
            audit.run_relay_audit(checkpoint, [101, 102],
                                  ["learned", "direct_only", "fixed_power"],
                                  output_dir=full, trace=True)
            expected = json.loads((full / "cases" / "fixed_power-102.json").read_text())

            shard = Path(folder) / "shard"
            shard.mkdir()
            (shard / "protocol.json").write_bytes((full / "protocol.json").read_bytes())
            result = audit.run_relay_audit_cases(
                checkpoint, shard, [(102, "fixed_power")])

            actual = json.loads((shard / "cases" / "fixed_power-102.json").read_text())
            self.assertEqual(actual, expected)
            self.assertEqual(result["completed_cases"], 1)
            self.assertEqual(result["skipped_cases"], 0)
            self.assertEqual(actual["fingerprint"],
                             json.loads((full / "protocol.json").read_text())["fingerprint"])
            with patch.object(audit, "run_episode", wraps=audit.run_episode) as run:
                repeated = audit.run_relay_audit_cases(
                    checkpoint, shard, [(102, "fixed_power")])
                run.assert_not_called()
            self.assertEqual(repeated["completed_cases"], 0)
            self.assertEqual(repeated["skipped_cases"], 1)

    def test_audit_cli_runs_saved_policy_and_writes_complete_case(self):
        import subprocess
        import sys
        with TemporaryDirectory() as folder:
            checkpoint = snapshot(Path(folder) / "policy", small_config())
            output = Path(folder) / "audit"
            process = subprocess.run(
                [sys.executable, "scripts/audit_relay.py", str(checkpoint),
                 "--output-dir", str(output), "--seeds", "101", "--modes", "learned", "--trace"],
                cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((output / "summary.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["protocol"]["seeds"], [101])
            self.assertEqual(report["records"][0]["episode_length"], 2)

    def test_future_profile_only_changes_horizon_and_run_labels(self):
        old = load_config(ROOT / "configs/comparison-20k-steps2000-scenario-v3.yaml").to_dict()
        c = load_config(ROOT / "configs/comparison-20k-steps1800-scenario-v3.yaml")
        self.assertEqual((c.runtime.num_envs, c.runtime.episodes, c.world.max_steps, c.runtime.seed,
                          c.world.num_targets, c.world.num_obstacles, c.network.report_bytes),
                         (4096, 20000, 1800, 44, 35, 25, 1000000))
        new = c.to_dict()
        new["world"]["max_steps"] = old["world"]["max_steps"]
        for section, key in (("runtime", "output_dir"), ("tracking", "group"), ("tracking", "tags")):
            new[section][key] = old[section][key]
        self.assertEqual(old, new)

    def test_warp_oracle_direct_far_peer_and_masked_gcs(self):
        from uav_network.environment.gpu.vector import WarpMissionEnv
        from uav_network.experiments.feasibility import oracle_action
        c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 3, 1, 0
        c.runtime.num_envs = 1
        c.sensor.altitude_m, c.sensor.obstacle_occlusion = 20, False
        c.runtime.device = c.runtime.replay_device = "cpu"
        env = WarpMissionEnv(c)
        self.addCleanup(env.close)
        env.reset(seed=44)
        state = env.vector.s
        for positions, expected in (([[500, 100], [500, 400], [900, 900]], 3),
                                    ([[500, 800], [500, 400], [900, 900]], 1),
                                    ([[500, 900], [950, 950], [900, 950]], 0)):
            state.positions.assign(np.asarray([positions], np.float64))
            state.targets.assign(np.asarray([[positions[0]]], np.float64))
            present = np.zeros((1, 3, 1), np.int32)
            present[0, 0, 0] = 1
            state.report_present.assign(present)
            state.known.assign(present)
            state.delivered.zero_()
            obs = env.vector._publish(env.vector._buffers[env.vector._buffer])["observation"][0].numpy()
            env._refresh()
            before = {name: getattr(state, name).numpy().copy()
                      for name in ("positions", "report_present", "known", "battery")}
            action = oracle_action(env, observation=obs)
            self.assertEqual(int(action.destination[0]), expected)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), value)

    def test_warp_oracle_view_exposes_report_recovery_state(self):
        from uav_network.environment.gpu.vector import WarpMissionEnv
        from uav_network.experiments.feasibility import _warp_oracle_view
        c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 0
        c.runtime.num_envs = 1
        c.runtime.device = c.runtime.replay_device = "cpu"
        env = WarpMissionEnv(c)
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=44)
        state = env.vector.s
        recovery = np.zeros((1, 2, 1), np.int32)
        recovery[0, 0, 0] = 1
        state.recovery.assign(recovery)

        view = _warp_oracle_view(env, obs)

        self.assertTrue(hasattr(view.reports, "recovery_pending"))
        self.assertTrue(bool(view.reports.recovery_pending[0, 0]))
        self.assertTrue(hasattr(view.reports, "delivered"))

    def test_warp_oracle_revisits_known_undelivered_target_after_custody_loss(self):
        from uav_network.environment.gpu.vector import WarpMissionEnv
        from uav_network.experiments.feasibility import oracle_action
        c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 0
        c.runtime.num_envs = 1
        c.sensor.altitude_m, c.sensor.obstacle_occlusion = 20, False
        c.runtime.device = c.runtime.replay_device = "cpu"
        env = WarpMissionEnv(c)
        self.addCleanup(env.close)
        env.reset(seed=44)
        state = env.vector.s
        state.positions.assign(np.asarray([[[500., 100.], [900., 900.]]], np.float64))
        state.velocities.zero_()
        state.targets.assign(np.asarray([[[500., 400.]]], np.float64))
        state.report_present.zero_()
        state.known.assign(np.ones((1, 2, 1), np.int32))
        state.delivered.zero_()
        recovery = np.zeros((1, 2, 1), np.int32)
        recovery[0, 0, 0] = 1
        state.recovery.assign(recovery)
        obs = env.vector._publish(env.vector._buffers[env.vector._buffer])["observation"][0].numpy()
        env._refresh()

        action = oracle_action(env, observation=obs)

        self.assertGreater(float(action.continuous[0, 1]), 0.0)

    def test_real_warp_feasibility_preserves_config_and_reports_backend_layout(self):
        from uav_network.experiments.feasibility import run_feasibility
        c = small_config()
        c.runtime.environment_backend = "warp_packet"
        c.runtime.num_envs = 4096
        c.network.enabled = True
        c.network.report_bytes, c.network.buffer_bytes = 1000, 3000
        original = c.to_dict()
        progress = []
        result = run_feasibility(c, [44], on_progress=progress.append)
        self.assertEqual(c.to_dict(), original)
        self.assertTrue(result["not_a_policy_baseline"])
        self.assertTrue(result["complete"])
        record = result["records"][0]
        self.assertEqual(record["environment_backend"], "warp_packet")
        self.assertEqual(record["episode_length"], 2)
        self.assertEqual(len(record["layout_hash"]), 64)
        self.assertEqual(record["scenario"]["world"]["num_targets"], 2)
        self.assertEqual(progress[-1]["status"], "episode_complete")
        self.assertIn("comparison_metrics", record)


if __name__ == "__main__":
    unittest.main()
