"""Training-only export: one validation-best policy on HF; no CPU/GitHub handoff."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import os
import json

import pytest

from uav_network.configuration import Config
from uav_network.cloud.kernel import kernel_script
from uav_network.cloud.cli import main
from uav_network.cloud.remote import finalize_runs
from uav_network.experiments.recovery_checkpoints import PeriodicRecovery
from uav_network.experiments.tracking import OnlineTracker


def test_training_only_cli_and_bootstrap_env():
    with patch("uav_network.cloud.cli.load_credentials", return_value=object()), \
         patch("uav_network.cloud.cli.launch", side_effect=lambda args, credentials: args):
        args = main(["launch", "--credential-dataset", "owner/private", "--training-only"])
    assert args.training_only is True
    script = kernel_script({"training_only": True})
    assert 'os.environ["UAV_TRAINING_ONLY"] = "true" if SPEC.get("training_only") else "false"' in script


def test_training_only_skips_recovery_and_final_cloud_evaluation(monkeypatch):
    monkeypatch.setenv("UAV_TRAINING_ONLY", "true")
    config = Config()
    config.tracking.checkpoint_store = "huggingface"
    with TemporaryDirectory() as temp:
        assert PeriodicRecovery(config, temp, started=0).enabled is False
        output = Path(temp)
        location = output / "train" / "matd3"
        location.mkdir(parents=True)
        (location / "summary.json").write_text(json.dumps({"episodes": 20000, "global_step": 1,
                                                             "updates": 1, "wandb_url": "test"}))
        with patch("uav_network.cloud.remote.subprocess.run") as command:
            result = finalize_runs({"runs": {"matd3": {}}, "stage": "train",
                                    "training_only": True}, [SimpleNamespace(returncode=0)], output)
        assert result["matd3"]["episodes"] == 20000
        assert not (output / "matd3-evaluation.log").exists()
        command.assert_not_called()


def test_training_only_uploads_only_validation_best(monkeypatch):
    monkeypatch.setenv("UAV_TRAINING_ONLY", "true")
    class Artifact:
        def __init__(self, name, type, metadata):
            self.aliases = []
        def add_file(self, path, name):
            assert name == "checkpoint.json"
        def wait(self, timeout):
            pass
        @property
        def qualified_name(self):
            return "owner/project/model:v1"
    class Run:
        id = "abcd1234"
        summary = {}
        def log_artifact(self, artifact, aliases):
            artifact.aliases = aliases
            assert aliases == ["latest", "final", "best"]
            return artifact
    with TemporaryDirectory() as temp:
        root = Path(temp)
        best = root / "best.pt"
        best.write_bytes(b"checkpoint")
        tracker = OnlineTracker.__new__(OnlineTracker)
        tracker.config = Config()
        tracker.config.tracking.log_checkpoint = True
        tracker.config.tracking.checkpoint_store = "huggingface"
        tracker.config.tracking.artifact_timeout_seconds = 30
        tracker.wandb = SimpleNamespace(Artifact=Artifact)
        tracker.output = root
        tracker.run = Run()
        tracker.last_checkpoint_upload = None
        tracker.last_recovery_confirmed_at = None
        valid = {"episode_budget_complete": True, "episodes": 20000,
                 "global_step": 1234, "checkpoint_selection": {"kind": "validation_best"},
                 "resume_checkpoint": str(root / "missing-resume.pt")}
        descriptor = {"schema": 1, "backend": "huggingface", "repo_id": "owner/model",
                      "revision": "a" * 40,
                      "files": {"checkpoint.pt": {"path": "test/checkpoint.pt", "size": 10,
                                                "sha256": "b" * 64}}}
        with patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files",
                   return_value=descriptor) as upload:
            assert tracker._checkpoint(best, valid, final=False) is None
            assert upload.call_count == 0
            with pytest.raises(ValueError, match="validated best"):
                tracker._checkpoint(best, {**valid, "checkpoint_selection": {}}, final=True)
            assert tracker._checkpoint(best, {**valid, "episode_budget_complete": False}, final=True) == "owner/project/model:v1"
            assert upload.call_count == 1
            assert upload.call_args.args[1] == {"checkpoint.pt": best}
            assert not (root / "resume.pt").exists()


def test_each_promoted_best_updates_one_stable_hf_location(monkeypatch):
    import hashlib
    monkeypatch.setenv("UAV_TRAINING_ONLY", "true")
    with TemporaryDirectory() as directory:
        actor = Path(directory) / "best.pt"
        actor.write_bytes(b"best-actor")
        digest = hashlib.sha256(actor.read_bytes()).hexdigest()
        tracker = OnlineTracker.__new__(OnlineTracker)
        tracker.config = Config()
        tracker.run = SimpleNamespace(id="abcd1234", summary={})
        descriptor = {"repo_id": "owner/model", "revision": "f" * 40,
                      "files": {"checkpoint.pt": {"sha256": digest}}}
        with patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files",
                   return_value=descriptor) as upload:
            outcome = tracker.export_best_checkpoint(actor,
                {"checkpoint_sha256": digest, "episode": 4096, "global_step": 4915200})
        assert outcome == descriptor
        assert upload.call_args.args[1] == {"checkpoint.pt": actor}
        assert upload.call_args.args[2] == "uav-network/abcd1234/best"
        assert tracker.run.summary["best_checkpoint_storage"]["episode"] == 4096
