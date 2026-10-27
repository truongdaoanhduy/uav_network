"""Verified full-state recovery without finalizing a live training segment."""
from copy import deepcopy
from importlib.util import find_spec
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_tracking import OnlineRun, StoredArtifact
from exact_resume_helpers import exact_world_config
from uav_network.configuration import Config, load_config
from uav_network.experiments.tracking import OnlineTracker
from uav_network.experiments.training import train


def recovery_class():
    assert find_spec("uav_network.experiments.recovery_checkpoints"), "Missing bounded wall-clock recovery"
    from uav_network.experiments.recovery_checkpoints import PeriodicRecovery
    return PeriodicRecovery


class Artifact(StoredArtifact):
    def wait(self, timeout):
        if getattr(self, "fail_confirmation", False):
            raise TimeoutError("confirmation unavailable")
        super().wait(timeout)


def upload_descriptor(config, files, prefix):
    return {"schema": 1, "backend": "huggingface", "repo_id": "owner/checkpoints",
            "revision": "a" * 40, "files": {
                name: {"path": prefix + "/" + name, "size": Path(path).stat().st_size,
                       "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for name, path in files.items()}}


def tracker_context(config, directory, run):
    sdk = SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: kwargs, Artifact=Artifact)
    return patch.dict(sys.modules, {"wandb": sdk})


def test_resumable_publication_bypasses_light_throttle_and_keeps_separate_pointer(tmp_path):
    c = Config()
    c.tracking.credentials_file = str(tmp_path / "absent.json")
    run = OnlineRun()
    artifacts = []
    def log(artifact, aliases):
        artifacts.append((artifact, aliases))
        return artifact
    run.log_artifact = log
    p, r = tmp_path / "policy.pt", tmp_path / "resume.pt"
    p.write_bytes(b"policy"); r.write_bytes(b"exact replay optimizer world RNG")
    with tracker_context(c, tmp_path, run), patch.dict(os.environ, {
            "WANDB_API_KEY": "test", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
            patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files", side_effect=upload_descriptor):
        tracker = OnlineTracker(c, tmp_path)
        tracker.checkpoint(p, {"episodes": 0, "global_step": 4})
        ref = tracker.checkpoint(p, {"episodes": 0, "global_step": 8, "resume_checkpoint": str(r), "environment_state_saved": True},
                                 resumable=True)
        artifact, aliases = artifacts[-1]
        manifest = json.loads(artifact.data)
        assert set(manifest["files"]) == {"checkpoint.pt", "resume.pt"}
        assert "resumable" not in aliases and "final" not in aliases
        assert "resumable" in artifact.aliases
        assert artifact.metadata["resumable"] is True
        assert artifact.metadata["environment_state_saved"] is True
        assert artifact.metadata["episode_budget_complete"] is False
        assert run.summary["recovery/checkpoint_artifact"] == ref
        assert run.summary["recovery/last_verified_global_step"] == 8
        tracker.last_checkpoint_upload = None
        tracker.checkpoint(p, {"episodes": 0, "global_step": 12})
        assert run.summary["recovery/last_verified_global_step"] == 8
        assert "resumable" not in artifacts[-1][1]
        prior = dict(run.summary)
        def fail(artifact, aliases):
            artifact.fail_confirmation = True
            return artifact
        run.log_artifact = fail
        with pytest.raises(TimeoutError):
            tracker.checkpoint(p, {"episodes": 0, "global_step": 16, "resume_checkpoint": str(r)},
                               resumable=True)
        assert run.summary["recovery/checkpoint_artifact"] == prior["recovery/checkpoint_artifact"]
        assert run.summary["recovery/last_verified_global_step"] == 8
        assert run.summary.get("episode_budget_complete") is not True
        tracker.close()


def test_resumable_publication_rejects_missing_full_state_before_external_upload(tmp_path):
    c = Config(); c.tracking.credentials_file = str(tmp_path / "absent.json")
    p = tmp_path / "policy.pt"; p.write_bytes(b"policy")
    run = OnlineRun()
    with tracker_context(c, tmp_path, run), patch.dict(os.environ, {
            "WANDB_API_KEY": "test", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
            patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files",
                  side_effect=AssertionError("must validate before network")):
        tracker = OnlineTracker(c, tmp_path)
        with pytest.raises(ValueError, match="resume"):
            tracker.checkpoint(p, {"episodes": 0, "global_step": 4}, resumable=True)
        tracker.close()


def recovery_config(tmp_path):
    c = Config()
    c.tracking.enabled = True
    c.tracking.resumable_checkpoint_interval_seconds = 1.
    c.tracking.resumable_checkpoint_max_uploads = 2
    c.runtime.max_train_seconds = 10.
    return c


def test_wall_clock_spreads_bounded_attempts_before_any_episode_and_releases_only_staging(tmp_path):
    cls = recovery_class()
    c = recovery_config(tmp_path)
    clock = [0.]
    preserved = tmp_path / "resume_latest.pt"; preserved.write_bytes(b"final-owned")
    snapshots = []
    class Tracker:
        def checkpoint(self, policy, progress, *, resumable=False):
            assert resumable is True and preserved.read_bytes() == b"final-owned"
            assert Path(progress["resume_checkpoint"]).read_bytes() == b"full state"
            snapshots.append(Path(progress["resume_checkpoint"]).parent)
            return "owner/project/recovery:v0"
        def diagnostic(self, values, step): pass
    def save(policy, *, resume_path):
        Path(policy).write_bytes(b"policy"); Path(resume_path).write_bytes(b"full state")
        return {"episodes": 0, "global_step": 4}
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter",
               side_effect=lambda: clock[0]):
        recovery = cls(c, tmp_path, started=0.)
        clock[0] = 4.9; recovery.maybe_save(save, Tracker())
        assert not snapshots
        clock[0] = 5.; recovery.maybe_save(save, Tracker())
        assert len(snapshots) == 1 and not snapshots[0].exists()
        assert preserved.read_bytes() == b"final-owned"
        clock[0] = 9.9; recovery.maybe_save(save, Tracker())
        assert len(snapshots) == 1
        clock[0] = 10.; recovery.maybe_save(save, Tracker())
        assert len(snapshots) == 2 and all(not p.exists() for p in snapshots)
        clock[0] = 20.; recovery.maybe_save(save, Tracker())
        assert len(snapshots) == 2


def test_failed_upload_is_visible_bounded_and_does_not_destroy_final_checkpoint(tmp_path):
    cls = recovery_class(); c = recovery_config(tmp_path)
    c.runtime.max_train_seconds = 0
    c.tracking.resumable_checkpoint_max_uploads = 1
    preserved = tmp_path / "resume_latest.pt"; preserved.write_bytes(b"previous final")
    events = []
    class Tracker:
        def checkpoint(self, *args, **kwargs): raise OSError("temporary external failure")
        def diagnostic(self, values, step): events.append((values, step))
    def save(policy, *, resume_path):
        Path(policy).write_bytes(b"policy"); Path(resume_path).write_bytes(b"resume")
        return {"episodes": 0, "global_step": 4}
    clock = [2.]
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter",side_effect=lambda:clock[0]):
        recovery = cls(c, tmp_path, started=0.)
        recovery.maybe_save(save, Tracker())
        assert events[-1][0]["recovery/status"] == "failed"
        assert events[-1][0]["recovery/error_type"] == "OSError"
        assert preserved.read_bytes() == b"previous final"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["resume_latest.pt"]
        count = len(events)
        clock[0] = 100.; recovery.maybe_save(save, Tracker())
        assert len(events) == count
        assert [v["recovery/status"] for v, step in events] == ["snapshot_started","upload_started","failed"]


def test_recovery_settings_load_validate_and_canonical_profile_enables_protection():
    c = load_config(overrides=["tracking.resumable_checkpoint_interval_seconds=10",
                               "tracking.resumable_checkpoint_max_uploads=2"])
    assert c.tracking.resumable_checkpoint_interval_seconds == 10
    for setting in ("tracking.resumable_checkpoint_interval_seconds=-1",
                    "tracking.resumable_checkpoint_max_uploads=0"):
        with pytest.raises(ValueError, match="resumable_checkpoint"):
            load_config(overrides=[setting])
    canonical = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    assert canonical.tracking.resumable_checkpoint_interval_seconds > 0
    assert canonical.tracking.resumable_checkpoint_max_uploads <= 4


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("pql", [False, True])
def test_periodic_full_state_before_first_episode_exactly_resumes_and_leaves_no_staging(tmp_path, algorithm, pql):
    captured = tmp_path / "captured.pt"
    c = exact_world_config(tmp_path / "first", algorithm)
    c.world.max_steps = 4
    c.tracking.enabled = True
    c.tracking.evaluation_interval = c.tracking.evaluation_interval_steps = 0
    c.tracking.resumable_checkpoint_interval_seconds = 1e-12
    c.tracking.resumable_checkpoint_max_uploads = 1
    c.pql.enabled, c.pql.mode = pql, "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps = 1
    class Tracker:
        def identity(self): return {"wandb_run_id": "test-run"}
        def learner(self,*args,**kwargs): pass
        def progress(self,*args,**kwargs): pass
        def episode(self,*args,**kwargs): pass
        def diagnostic(self,*args,**kwargs): pass
        def failure(self,*args,**kwargs): pass
        def complete(self,*args,**kwargs): return {}
        def close(self,*args,**kwargs): pass
        def checkpoint(self, policy, progress, final=False, resumable=False):
            if resumable:
                payload = torch.load(progress["resume_checkpoint"],weights_only=True)
                assert payload["progress"]["episodes"] == 0
                assert progress["environment_state_saved"] is True
                assert payload["progress"]["environment_state_saved"] is True
                assert payload["training"]["environment"]["world"]["arrays"]
                assert payload["training"]["replay"]["size"] > 0
                torch.save(payload,captured)
            return "owner/project/recovery:v0"
    module = "uav_network.experiments.pql_training" if pql else "uav_network.experiments.training"
    with patch(module+".start_tracking",return_value=Tracker()):
        first = train(c)
    assert captured.is_file(), "Periodic recovery must publish saved-world evidence before the first episode"
    assert not list(Path(c.runtime.output_dir).glob(".recovery-*"))
    saved = torch.load(first["resume_checkpoint"],weights_only=True)
    c.runtime.output_dir = str(tmp_path / "resumed")
    c.runtime.warm_start_checkpoint = str(captured)
    c.tracking.enabled = False
    c.tracking.resumable_checkpoint_interval_seconds = 0.
    second = train(c)
    assert second["state_restore"] == "training_state_exact_environment"
    resumed = torch.load(second["resume_checkpoint"],weights_only=True)
    assert resumed["updates"] == saved["updates"]
    assert resumed["actor_updates"] == saved["actor_updates"]
    for name,value in saved["model"].items():
        torch.testing.assert_close(resumed["model"][name],value,rtol=0,atol=0)


def test_final_policy_without_full_state_is_not_advertised_as_resumable(tmp_path):
    c = Config(); c.tracking.credentials_file = str(tmp_path / "absent.json")
    run = OnlineRun(); published = []
    def log(artifact, aliases):
        published.append((artifact, aliases)); return artifact
    run.log_artifact = log
    p = tmp_path / "policy.pt"; p.write_bytes(b"policy")
    with tracker_context(c, tmp_path, run), patch.dict(os.environ, {
            "WANDB_API_KEY": "test", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
            patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files", side_effect=upload_descriptor):
        tracker = OnlineTracker(c, tmp_path)
        tracker.checkpoint(p, {"episodes": 1, "global_step": 4}, final=True)
        assert published[-1][0].metadata["resumable"] is False
        assert "resumable" not in published[-1][1]
        assert "recovery/checkpoint_artifact" not in run.summary
        tracker.close()


def test_backup_status_logging_failure_does_not_become_a_new_training_failure(tmp_path, capsys):
    cls = recovery_class(); c = recovery_config(tmp_path); c.runtime.max_train_seconds = 0
    class Tracker:
        def checkpoint(self, *args, **kwargs): raise OSError("offline")
        def diagnostic(self, *args, **kwargs): raise RuntimeError("logging unavailable")
    def save(policy, *, resume_path):
        Path(policy).write_bytes(b"policy"); Path(resume_path).write_bytes(b"resume")
        return {"episodes": 0, "global_step": 4}
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter", return_value=2.):
        cls(c, tmp_path, started=0.).maybe_save(save, Tracker())
    assert "Recovery" in capsys.readouterr().out


def test_explicit_legacy_backend_does_not_enable_large_periodic_HF_backups(tmp_path):
    cls = recovery_class()
    c = recovery_config(tmp_path)
    c.tracking.checkpoint_store = "wandb"
    c.validate()
    captured = []
    def save(policy, *, resume_path):
        captured.append(policy)
        Path(policy).write_bytes(b"policy"); Path(resume_path).write_bytes(b"resume")
        return {"episodes": 0, "global_step": 4}
    class Tracker:
        def checkpoint(self, *args, **kwargs): return "owner/project/policy:v0"
        def diagnostic(self, *args, **kwargs): pass
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter", return_value=20.):
        cls(c, tmp_path, started=0.).maybe_save(save, Tracker())
    assert not captured


@pytest.mark.parametrize("final", [False, True])
def test_failed_confirmation_keeps_remote_resumable_alias_on_previous_version(tmp_path, final):
    server = {"resumable": "previous:v1"}
    class AliasArtifact(StoredArtifact):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.aliases = []
        def wait(self, timeout):
            raise TimeoutError("confirmation unavailable")
        def save(self):
            assert self.committed
            server.update({alias: "unconfirmed:v2" for alias in self.aliases})
    run = OnlineRun()
    run.summary["recovery/checkpoint_artifact"] = "previous:v1"
    def log(artifact, aliases):
        # A remote write may complete even when its local wait response is lost.
        server.update({alias: "unconfirmed:v2" for alias in aliases})
        artifact.aliases = list(aliases)
        return artifact
    run.log_artifact = log
    c = Config(); c.tracking.credentials_file = str(tmp_path / "absent.json")
    policy, resume = tmp_path / "policy.pt", tmp_path / "resume.pt"
    policy.write_bytes(b"policy"); resume.write_bytes(b"full state")
    sdk = SimpleNamespace(init=lambda **kwargs:run, Settings=lambda **kwargs:kwargs, Artifact=AliasArtifact)
    with patch.dict(sys.modules, {"wandb": sdk}), patch.dict(os.environ, {
            "WANDB_API_KEY": "test", "WANDB_ENTITY": "owner", "WANDB_PROJECT": "project"}), \
            patch("uav_network.experiments.checkpoint_store.upload_checkpoint_files", side_effect=upload_descriptor):
        tracker = OnlineTracker(c,tmp_path)
        with pytest.raises(TimeoutError):
            tracker.checkpoint(policy, {"episodes": 0, "global_step": 4, "resume_checkpoint":str(resume)},
                               final=final, resumable=not final)
        assert server["resumable"] == "previous:v1"
        assert run.summary["recovery/checkpoint_artifact"] == "previous:v1"
        tracker.close()


def test_backup_reports_snapshot_and_upload_phases_before_slow_operations(tmp_path):
    c=recovery_config(tmp_path);c.runtime.max_train_seconds=0
    phases=[]
    class Tracker:
        def diagnostic(self,values,step):phases.append(values["recovery/status"])
        def checkpoint(self,path,progress,*,resumable=False):
            assert phases[-1]=="upload_started"
            return "owner/project/recovery:v1"
    def save(policy,*,resume_path):
        assert phases[-1]=="snapshot_started"
        Path(policy).write_bytes(b"policy");Path(resume_path).write_bytes(b"resume")
        return {"episodes":0,"global_step":4}
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter",return_value=2.):
        recovery_class()(c,tmp_path,started=0.).maybe_save(save,Tracker())
    assert phases==["snapshot_started","upload_started","uploaded"]


def test_backup_logging_and_console_failures_do_not_abort_the_snapshot(tmp_path):
    c=recovery_config(tmp_path);c.runtime.max_train_seconds=0
    saves=[]
    class Tracker:
        def diagnostic(self,*args):raise OSError("logging unavailable")
        def checkpoint(self,*args,**kwargs):return "owner/project/recovery:v1"
    def save(policy,*,resume_path):
        saves.append(True);Path(policy).write_bytes(b"p");Path(resume_path).write_bytes(b"r")
        return {"episodes":0,"global_step":4}
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter",return_value=2.), \
         patch("builtins.print",side_effect=BrokenPipeError()):
        recovery_class()(c,tmp_path,started=0.).maybe_save(save,Tracker())
    assert saves==[True]


def test_closed_stdout_does_not_prevent_backup_when_status_logging_fails(tmp_path):
    from io import StringIO
    c = recovery_config(tmp_path); c.runtime.max_train_seconds = 0
    saves = []
    class Tracker:
        def diagnostic(self, *args, **kwargs): raise OSError("telemetry unavailable")
        def checkpoint(self, *args, **kwargs): return "owner/project/recovery:v1"
    def save(policy, *, resume_path):
        saves.append(True); Path(policy).write_bytes(b"p"); Path(resume_path).write_bytes(b"r")
        return {"episodes": 0, "global_step": 4}
    closed = StringIO(); closed.close()
    with patch("uav_network.experiments.recovery_checkpoints.time.perf_counter", return_value=2.), patch("sys.stdout", closed):
        recovery_class()(c, tmp_path, started=0.).maybe_save(save, Tracker())
    assert saves == [True]
