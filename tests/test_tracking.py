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

    def add_file(self, path, name):
        self.data = Path(path).read_bytes()

    def wait(self, timeout):
        self.committed = True

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
                              "gcs_communication_energy_j": 1, "packet_pdr_percent": 80,
                              "packet_e2e_delay_ms": 12, "payload_throughput_kbps": 4,
                              "phy_success_percent": 60, "collisions": 2,
                              "report_e2e_delay_s": 1.2, "report_average_hops": 2,
                              "energy_blocked_transmissions": 1}, "learner": {"critic_loss": 1}}
        metrics = episode_metrics(record, num_uavs=2)
        self.assertEqual(metrics["episode/return"], -5)
        self.assertEqual(metrics["safety/blocked_motion_rate"], .1)
        self.assertEqual(metrics["energy/total_uav_energy_j"], 503)
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
