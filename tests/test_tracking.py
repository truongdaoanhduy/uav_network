"""Metric values come from real mission states, without exposing credentials."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import local_observations
from uav_network.experiments.tracking import episode_metrics, progress_metrics, final_metrics
from uav_network.experiments.tracking import OnlineTracker
from uav_network.cloud.credentials import load_credentials


class StoredArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.metadata, self.committed = name, metadata, False
        self.aliases = []

    def add_file(self, path, name):
        self.data = Path(path).read_bytes()

    def wait(self, timeout):
        self.committed = True

    def save(self):
        assert self.committed, "Aliases can only be saved after artifact confirmation"

    @property
    def qualified_name(self):
        if not self.committed:
            raise RuntimeError("Artifact is not committed yet")
        return f"owner/project/{self.name}:v0"


class OnlineRun:
    offline = False
    id, entity, project, url = "run123", "owner", "project", "https://wandb.ai/owner/project/runs/run123"

    def __init__(self):
        self.summary, self.logged, self.finished = {}, [], False

    def save(self, *args, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        pass

    def log_artifact(self, artifact, aliases):
        return artifact

    def log(self, values):
        self.logged.append(values)

    def finish(self, exit_code=0):
        self.finished = True


class TrackingTests(unittest.TestCase):
    def test_online_tracker_does_not_sync_large_raw_run_files(self):
        run = OnlineRun()
        saved = []
        settings = {}
        run.save = lambda path, **kwargs: saved.append(Path(path).name)
        def make_settings(**kwargs):
            settings.update(kwargs)
            return kwargs
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=make_settings, Artifact=StoredArtifact)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            policy = Path(directory) / "policy.pt"
            policy.write_bytes(b"policy")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                tracker.checkpoint(policy, {"episodes": 1, "global_step": 10})
                tracker.close()
        self.assertEqual(settings.get("console"), "wrap")
        self.assertNotIn("diagnostics.jsonl", saved)
        self.assertNotIn("episodes.jsonl", saved)

    def test_periodic_uploads_are_paced_but_final_full_state_always_uploads(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=StoredArtifact)
        descriptor = {'schema':1,'backend':'huggingface','repo_id':'owner/private','revision':'a'*40,
                      'files':{'checkpoint.pt':{'path':'run/checkpoint.pt','size':6,'sha256':'b'*64}}}
        with TemporaryDirectory() as directory:
            config = Config(); config.tracking.credentials_file = str(Path(directory)/'absent.txt')
            policy = Path(directory)/'policy.pt'; policy.write_bytes(b'policy')
            resume = Path(directory)/'resume.pt'; resume.write_bytes(b'full replay and optimizer')
            clock = [0]
            with patch.dict(sys.modules, {'wandb':sdk}), patch.dict(os.environ, {
                    'WANDB_API_KEY':'test-key','WANDB_ENTITY':'owner','WANDB_PROJECT':'project'}), \
                 patch('uav_network.experiments.tracking.time.monotonic', side_effect=lambda:clock[0]), \
                 patch('uav_network.experiments.checkpoint_store.upload_checkpoint_files', return_value=descriptor) as upload:
                tracker = OnlineTracker(config, directory)
                progress = {'episodes':1,'global_step':10}
                self.assertIsNotNone(tracker.checkpoint(policy, progress))
                clock[0] = 5
                self.assertIsNone(tracker.checkpoint(policy, progress))
                clock[0] = 300
                self.assertIsNotNone(tracker.checkpoint(policy, progress))
                clock[0] = 301
                self.assertIsNotNone(tracker.checkpoint(policy, {**progress,'resume_checkpoint':str(resume)}, final=True))
                self.assertEqual(upload.call_count, 3)
                self.assertIn('resume.pt', upload.call_args.args[1])
                tracker.close()

    def test_finished_session_reports_time_limit_separately_from_episode_budget(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=StoredArtifact)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            policy, resume = Path(directory) / "policy.pt", Path(directory) / "resume.pt"
            policy.write_bytes(b"policy")
            resume.write_bytes(b"full training")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                tracker.complete({"checkpoint": str(policy), "resume_checkpoint": str(resume),
                                  "episodes": 0, "global_step": 8, "updates": 0,
                                  "initial_global_step": 4, "episode_budget_complete": False,
                                  "stop_reason": "time_budget", "environment_state_saved": True})
                tracker.close()
            self.assertTrue(run.finished)
            self.assertFalse(run.summary["episode_budget_complete"])
            self.assertEqual(run.summary["session_status"], "session_time_budget_reached")
            self.assertEqual(run.summary["initial_global_step"], 4)
            self.assertEqual(run.summary["global_step"], 8)
            self.assertTrue(run.summary["environment_state_saved"])

    def test_missing_completion_metadata_fails_closed_in_tracking_summary(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=StoredArtifact)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            policy = Path(directory) / "policy.pt"
            policy.write_bytes(b"policy")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                tracker.complete({"checkpoint": str(policy), "episodes": 0,
                                  "global_step": 8, "updates": 0})
                tracker.close()
        self.assertFalse(run.summary["episode_budget_complete"])
        self.assertEqual(run.summary["stop_reason"], "unknown")
        self.assertEqual(run.summary["session_status"], "session_completion_unknown")

    def test_checkpoint_upload_interval_must_be_finite_and_nonnegative(self):
        for value in (-1, float('nan'), float('inf')):
            config = Config(); config.tracking.checkpoint_upload_interval_seconds = value
            with self.assertRaisesRegex(ValueError, 'checkpoint_upload_interval'):
                config.validate()

    def test_public_metadata_identifies_training_mode_and_continuation_segment(self):
        run, captured = OnlineRun(), {}
        def init(**kwargs):
            captured.update(kwargs["config"])
            return run
        sdk = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project",
                    "UAV_TRAINING_MODE": "BASE", "UAV_AUTO_CONTINUE": "true",
                    "UAV_ROOT_KAGGLE_KERNEL": "owner/root", "UAV_KAGGLE_KERNEL": "owner/segment-two",
                    "UAV_CONTINUATION_INDEX": "2", "UAV_EPISODE_BUDGET": "10000"}):
                tracker = OnlineTracker(c, directory)
                self.assertEqual(captured.get("training_mode"), "BASE")
                self.assertEqual(captured["pipeline"], {"auto_continue": True, "root_kernel_ref": "owner/root",
                    "continuation_index": 2, "target_episodes": 10000})
                for name, expected in (("training_mode", "BASE"), ("root_kernel_ref", "owner/root"),
                                       ("continuation_index", 2)):
                    self.assertEqual(run.summary[name], expected)
                tracker.close()

    def test_continuation_throughput_counts_only_new_transitions(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory, initial_progress={"global_step": 1000})
                tracker.started = 8
                obs = {"observation": np.zeros((1, 6, 95)), "state": np.zeros((1, 217))}
                with patch("uav_network.experiments.tracking.time.perf_counter", return_value=10):
                    tracker.progress(obs, 1010, 0, SimpleNamespace(updates=0), 10, force=True)
                self.assertEqual(run.logged[-1]["train/joint_transitions_per_s"], 5)
                tracker.close()

    def test_progress_logs_episode_step_distribution_and_percent(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
        with TemporaryDirectory() as directory:
            c = Config()
            c.world.max_steps = 2000
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                obs = {"observation": np.zeros((4, 6, 95)), "state": np.zeros((4, 217))}
                tracker.progress(obs, 400, 2, SimpleNamespace(updates=3), 100,
                                 episode_lengths=np.array([100, 200, 300, 400]), force=True)
                logged = run.logged[-1]
                self.assertEqual(logged["progress/episode_step_min"], 100)
                self.assertEqual(logged["progress/episode_step_mean"], 250)
                self.assertEqual(logged["progress/episode_step_max"], 400)
                self.assertEqual(logged["progress/episode_progress_percent_mean"], 12.5)
                tracker.close()

    def test_periodic_upload_is_committed_while_the_run_is_still_active(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=StoredArtifact)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            path = Path(directory) / "episode_000010.pt"
            path.write_bytes(b"finite-policy")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                ref = tracker.checkpoint(path, {"episodes": 10, "global_step": 8000})
                self.assertTrue(ref.endswith(":v0"))
                self.assertFalse(run.finished)
                tracker.close()

    def test_hf_checkpoint_artifact_contains_only_tiny_manifest(self):
        run = OnlineRun()
        added = []
        class ManifestArtifact:
            def __init__(self, name, type, metadata):
                self.name = name
            def add_file(self, path, name):
                added.append((name, Path(path).stat().st_size))
            def wait(self, timeout):
                pass
            @property
            def qualified_name(self):
                return f"owner/project/{self.name}:v0"
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs,
                              Artifact=ManifestArtifact)
        descriptor = {"schema": 1, "backend": "huggingface", "repo_id": "owner/private",
                      "revision": "a" * 40, "files": {"checkpoint.pt": {
                          "path": "run/checkpoint.pt", "size": 5_000_000_000, "sha256": "b" * 64}}}
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            policy = Path(directory) / "policy.pt"
            policy.write_bytes(b"policy")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
                    patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files", return_value=descriptor):
                tracker = OnlineTracker(c, directory)
                tracker.checkpoint(policy, {"episodes": 1, "global_step": 10})
                tracker.close()
        self.assertEqual([name for name, _ in added], ["checkpoint.json"])
        self.assertLess(added[0][1], 256 * 1024)

    def test_empty_soft_deadline_does_not_claim_final_episode_aggregates(self):
        run = OnlineRun()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=StoredArtifact)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent.txt")
            path = Path(directory) / "final.pt"
            path.write_bytes(b"finite-policy")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                tracker = OnlineTracker(c, directory)
                result = tracker.complete({"episodes": 0, "global_step": 2, "updates": 0,
                                           "checkpoint": str(path), "episode_budget_complete": False})
                self.assertFalse(any(key.startswith("final_") for key in result))
                self.assertFalse(run.summary["episode_budget_complete"])
                tracker.close()

    def test_tracker_rejects_wandb_destination_mismatch_and_finishes_run(self):
        run = OnlineRun()
        run.project = "wrong-project"
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent-key.txt")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                with self.assertRaisesRegex(RuntimeError, "destination"):
                    OnlineTracker(c, directory)
        self.assertTrue(run.finished)

    def test_tracker_setup_failure_finishes_the_started_online_run(self):
        class Run:
            offline = False
            id, entity, project, url = "run123", "owner", "project", "https://wandb.ai/owner/project/runs/run123"
            finished_failed = False

            summary = {}

            def save(self, *args, **kwargs):
                pass

            def define_metric(self, *args, **kwargs):
                raise OSError("metric registration failed")

            def finish(self, exit_code=0):
                self.finished_failed = exit_code != 0

        run = Run()
        sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs)
        with TemporaryDirectory() as directory:
            c = Config()
            c.tracking.checkpoint_store = "wandb"
            c.tracking.credentials_file = str(Path(directory) / "absent-key.txt")
            with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
                    "WANDB_API_KEY": "test-key", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}):
                with self.assertRaisesRegex(OSError, "registration"):
                    OnlineTracker(c, directory)
        self.assertTrue(run.finished_failed)

    def test_legacy_key_schema_preserves_accounts_without_secret_repr(self):
        data = {"kaggle": {"accounts": [{"name": "account_01", "token": "kaggle-secret", "enabled": True}],
                           "selection": {"min_gpu_remaining_hours": .25}},
                "wandb": {"api_key": "wandb-secret", "entity": "owner", "project": "project"},
                "huggingface": {"token": "unused-hf-secret", "repo_id": "unused"}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "key.txt"
            path.write_text(json.dumps(data))
            credentials = load_credentials(path)
        self.assertEqual(credentials.accounts[0].name, "account_01")
        self.assertEqual(credentials.project, "project")
        self.assertNotIn("kaggle-secret", repr(credentials))
        self.assertNotIn("wandb-secret", repr(credentials))

    def test_progress_reports_search_delivery_and_charge_before_episode_end(self):
        c = Config()
        c.world.num_uavs = 2
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.network.enabled = False
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options=dict(positions=[[500, 100], [600, 100]],
                                            targets=[[500, 100]], obstacles=[]))
            env.battery_j[:] = [c.energy.battery_capacity_j / 2, 0]
            env.reports.delivered[:] = True
            obs = {"observation": local_observations(env)[None], "state": env.state()[None]}
            metrics = progress_metrics(obs, c)
            self.assertEqual(metrics["progress/search_rate_percent"], 100)
            self.assertEqual(metrics["progress/delivery_rate_percent"], 100)
            self.assertEqual(metrics["progress/mean_battery_fraction"], .25)
            self.assertEqual(metrics["progress/mean_active_uavs"], 1)
        finally:
            env.close()

    def test_episode_rates_energy_and_final_aggregate_use_real_denominators(self):
        record = {"episode": 1, "global_step": 20, "return": -5, "episode_length": 10,
                  "search_rate_percent": 50, "delivery_rate_percent": 30, "success": False,
                  "blocked_moves": 2, "boundary_clips": 1, "expired_reports": 1, "dropped_reports": 2,
                  "energy": {"propulsion_energy_j": 500, "battery_fraction": [.5, .25], "active_uavs": 2},
                  "network": {"enabled": True, "uav_communication_energy_j": 3,
                              "gcs_communication_energy_j": 1,
                              "uav_tx_energy_j": 2, "uav_rx_energy_j": 1,
                              "gcs_tx_energy_j": .4, "gcs_rx_energy_j": .6,
                              "packet_pdr_percent": 80,
                              "packet_e2e_delay_ms": 12, "payload_throughput_kbps": 4,
                              "phy_success_percent": 60, "collisions": 2,
                              "report_e2e_delay_s": 1.2, "report_average_hops": 2,
                              "energy_blocked_transmissions": 1}, "learner": {"critic_loss": 1}}
        metrics = episode_metrics(record, num_uavs=2)
        self.assertEqual(metrics["episode/return"], -5)
        self.assertEqual(metrics["safety/blocked_motion_rate"], .1)
        self.assertEqual(metrics["energy/total_uav_energy_j"], 503)
        self.assertEqual(metrics["energy/uav_tx_radio_energy_j"], 2)
        self.assertEqual(metrics["energy/uav_rx_radio_energy_j"], 1)
        self.assertEqual(metrics["energy/gcs_tx_radio_energy_j"], .4)
        self.assertEqual(metrics["energy/gcs_rx_radio_energy_j"], .6)
        self.assertEqual(metrics["network/pdr_percent"], 80)
        later = {**record, "return": 10, "search_rate_percent": 100,
                 "delivery_rate_percent": 100, "success": True}
        final = final_metrics([record, later])
        self.assertEqual(final["final_mean_return"], 2.5)
        self.assertEqual(final["final_search_rate_percent"], 75)
        self.assertEqual(final["final_delivery_rate_percent"], 65)
        self.assertEqual(final["final_success_rate"], .5)


if __name__ == "__main__":
    unittest.main()


def test_v2_reward_and_shield_metrics_are_scalar_episode_fields():
    record = {"episode":1,"global_step":1,"return":0.,"episode_length":2,
              "search_rate_percent":0.,"delivery_rate_percent":0.,"success":False,
              "blocked_moves":0,"boundary_clips":0,"expired_reports":0,"dropped_reports":0,
              "energy":{"propulsion_energy_j":1.,"battery_fraction":[1.],"battery_j":[1.],
                        "active_uavs":1,"per_uav_propulsion_energy_j":[1.],
                        "per_uav_rf_energy_j":[0.],"per_uav_total_energy_j":[1.]},
              "network":{"enabled":False},
              "motion_counts":{"shield_interventions":1,"shield_infeasible":0,
                               "commanded_stalls":0,"shield_correction":.125},
              "reward_components":{"communication_efficiency":-.25,"shield_correction":-.0125}}
    metrics = episode_metrics(record,1)
    assert metrics["motion/shield_correction"] == .125
    assert metrics["reward/communication_efficiency"] == -.25
    assert metrics["reward/shield_correction"] == -.0125
