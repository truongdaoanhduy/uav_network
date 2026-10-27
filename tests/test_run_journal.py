"""Structured run evidence is useful even when optional logging I/O fails."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np


def journal_type():
    from importlib.util import find_spec
    assert find_spec("uav_network.experiments.run_journal"), "Structured run journal is missing"
    from uav_network.experiments.run_journal import RunJournal
    return RunJournal


def test_journal_retains_context_and_redacts_secrets_without_serializing_objects(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv("WANDB_API_KEY","synthetic-private-token")
    class Unsafe:
        def __repr__(self): raise AssertionError("must not dump tensors/objects")
    j=journal_type()(tmp_path,{"run_id":"run123","algorithm":"masac","seed":44,"api_key":"synthetic-private-token"})
    j.record("progress",{"global_step":4096,"completed_episodes":0,"phase":"learning",
                         "td_error":float("nan"),"note":"synthetic-private-token","payload":Unsafe()})
    j.record("failure",{"error_type":"RuntimeError"})
    status=json.loads((tmp_path/"training-status.json").read_text())
    rows=[json.loads(line) for line in (tmp_path/"training-events.jsonl").read_text().splitlines()]
    assert status["event"]=="failure" and status["context"]["global_step"]==4096
    assert status["context"]["completed_episodes"]==0 and status["context"]["td_error"]=="NaN"
    assert len(rows)==2 and rows[0]["identity"]["seed"]==44
    assert "synthetic-private-token" not in json.dumps(rows)+capsys.readouterr().out
    assert "api_key" not in status["identity"]


def test_journal_console_failure_does_not_prevent_disk_evidence(tmp_path):
    j=journal_type()(tmp_path,{"run_id":"r"})
    with patch("builtins.print",side_effect=BrokenPipeError()):
        j.record("progress",{"global_step":10})
    assert json.loads((tmp_path/"training-status.json").read_text())["context"]["global_step"]==10
    assert (tmp_path/"training-events.jsonl").read_text()


def test_journal_status_failure_does_not_prevent_append_or_escape(tmp_path):
    j=journal_type()(tmp_path,{"run_id":"r"})
    with patch.object(Path,"replace",side_effect=OSError("disk full")):
        j.record("progress",{"global_step":11})
    assert json.loads((tmp_path/"training-events.jsonl").read_text().splitlines()[-1])["context"]["global_step"]==11


def tracker_fixture(tmp_path,stack):
    from test_tracking import OnlineRun,StoredArtifact
    from uav_network.configuration import Config
    from uav_network.experiments.tracking import OnlineTracker
    c=Config();c.tracking.credentials_file=str(tmp_path/"missing.txt");c.tracking.checkpoint_store="wandb"
    c.runtime.episodes=20
    run=OnlineRun();saved=[]
    run.save=lambda path,**kw:saved.append((Path(path).name,kw["policy"]))
    sdk=SimpleNamespace(init=lambda **kw:run,Settings=lambda **kw:kw,Artifact=StoredArtifact)
    stack.enter_context(patch.dict("sys.modules",{"wandb":sdk}))
    stack.enter_context(patch.dict("os.environ",{"WANDB_API_KEY":"synthetic-wandb","WANDB_ENTITY":"owner","WANDB_PROJECT":"project"}))
    stack.enter_context(patch("uav_network.experiments.dashboard.ensure_project_workspace",return_value="workspace"))
    tracker=OnlineTracker(c,tmp_path)
    return c,run,saved,tracker


def observations(c):
    from uav_network.environment.observations import observation_dim,state_dim
    return {"observation":np.zeros((1,c.world.num_uavs,observation_dim(c))),
            "state":np.zeros((1,state_dim(c)))}


def test_tracker_journals_actual_progress_and_registers_only_small_live_file(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        t.learner({"critic_loss":2.5,"q_abs_max":4.0},4096)
        t.progress(observations(c),4096,0,SimpleNamespace(updates=7),131072,force=True)
        status=json.loads((tmp_path/"training-status.json").read_text())
        assert status["context"]["global_step"]==4096
        assert status["context"]["completed_episodes"]==0
        assert status["context"]["phase"]=="learning"
        assert status["context"]["learner/critic_loss"]==2.5
        assert ("training-status.json","live") in saved
        assert ("training-events.jsonl","end") in saved
        assert ("metrics-guide.md","now") in saved
        assert not any(name=="episodes.jsonl" for name,policy in saved)
        assert run.logged[-1]["run/episode_budget_percent"]==0.0
        t.close()


def test_full_upload_clock_is_not_reset_by_light_policy_uploads(tmp_path):
    from contextlib import ExitStack
    clock=[0.]
    with ExitStack() as stack:
        stack.enter_context(patch("uav_network.experiments.tracking.time.perf_counter",side_effect=lambda:clock[0]))
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        policy=tmp_path/"policy.pt";policy.write_bytes(b"policy")
        resume=tmp_path/"resume.pt";resume.write_bytes(b"full state")
        clock[0]=10.
        t.checkpoint(policy,{"episodes":1,"global_step":20,"resume_checkpoint":str(resume)},resumable=True)
        clock[0]=20.;t.checkpoint(policy,{"episodes":2,"global_step":30})
        clock[0]=30.;t.progress(observations(c),40,2,SimpleNamespace(updates=8),20,force=True)
        assert run.logged[-1]["recovery/seconds_since_confirmed_upload"]==20.
        assert run.logged[-1]["recovery/full_checkpoint_available"]==1
        run.save=lambda *a,**k:(_ for _ in ()).throw(OSError("optional upload unavailable"))
        t.close();assert run.finished


def test_cpu_evaluation_journal_does_not_invent_training_budget_progress(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.dict("os.environ",{"UAV_STAGE":"visualize"}))
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        t.diagnostic({"evaluation/episodes_completed":2,"evaluation/delivery_rate_percent":70.,
                      "evaluation/success":0,"evaluation/checkpoint_global_step":12345},2)
        status=json.loads((tmp_path/"training-status.json").read_text())
        assert status["event"]=="evaluation"
        assert status["context"]["evaluation/episodes_completed"]==2
        assert status["context"]["evaluation/checkpoint_global_step"]==12345
        assert "run/episode_budget_percent" not in status["context"]
        t.close()


def test_recovery_finished_phase_returns_to_actual_learning_state(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        t.progress(observations(c),4096,0,SimpleNamespace(updates=7),100,force=True)
        t.diagnostic({"recovery/status":"snapshot_started"},4096)
        t.diagnostic({"recovery/status":"upload_started"},4096)
        t.diagnostic({"recovery/status":"uploaded"},4096)
        status=json.loads((tmp_path/"training-status.json").read_text())
        assert status["context"]["phase"]=="learning"
        t.close()


def test_fixed_policy_aggregate_results_are_journaled_even_without_cpu_episode_fields(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        t.diagnostic({"eval/mean_return":10.,"eval/mean_delivery_rate_percent":80.,
                      "eval/success_rate":.5,"evaluation/checkpoint_global_step":4096},8192)
        status=json.loads((tmp_path/"training-status.json").read_text())
        assert status["event"]=="evaluation_result"
        assert status["context"]["eval/mean_delivery_rate_percent"]==80.
        assert status["context"]["evaluation/checkpoint_global_step"]==4096
        t.close()


def test_journal_closed_stdout_keeps_append_and_status(tmp_path):
    from io import StringIO
    closed = StringIO(); closed.close()
    j = journal_type()(tmp_path, {"run_id": "r"})
    with patch("sys.stdout", closed):
        j.record("progress", {"global_step": 12})
    assert json.loads((tmp_path / "training-status.json").read_text())["context"]["global_step"] == 12
    assert json.loads((tmp_path / "training-events.jsonl").read_text())["event"] == "progress"


def test_journal_io_warning_with_closed_stdout_is_optional(tmp_path):
    from io import StringIO
    closed = StringIO(); closed.close()
    j = journal_type()(tmp_path, {"run_id": "r"})
    with patch("sys.stdout", closed):
        j._warn(OSError("disk unavailable"))
    assert j.warned


def test_successful_recovery_does_not_retain_previous_attempt_error(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        t.diagnostic({"recovery/status":"failed","recovery/error_type":"OSError"},4)
        t.diagnostic({"recovery/status":"snapshot_started"},8)
        t.diagnostic({"recovery/status":"uploaded"},8)
        status=json.loads((tmp_path/"training-status.json").read_text())
        assert status["context"]["recovery/status"]=="uploaded"
        assert status["context"]["recovery/error_type"] is None
        t.close()


def test_failure_traceback_file_survives_closed_stdout(tmp_path):
    from io import StringIO
    from uav_network.experiments.diagnostics import failure_report
    closed=StringIO();closed.close()
    with patch("sys.stdout",closed):
        path=failure_report(None,tmp_path,RuntimeError("original failure"),{"global_step":4})
    report=json.loads(path.read_text())
    assert report["error_type"]=="RuntimeError"
    assert "original failure" in report["traceback"]


def test_atomic_status_has_bounded_explicit_refresh_and_forced_transition_upload(tmp_path):
    from contextlib import ExitStack
    clock=[0.]
    with ExitStack() as stack:
        stack.enter_context(patch("uav_network.experiments.tracking.time.monotonic",side_effect=lambda:clock[0]))
        c,run,saved,t=tracker_fixture(tmp_path,stack)
        uploads=lambda:sum(item==("training-status.json","now") for item in saved)
        initial=uploads()
        for step in (1,2):
            clock[0]=float(step)
            t.progress(observations(c),step,0,SimpleNamespace(updates=1),100,force=True)
        assert uploads()==initial
        clock[0]=61.
        t.progress(observations(c),3,0,SimpleNamespace(updates=1),100,force=True)
        assert uploads()==initial+1
        clock[0]=62.
        t.diagnostic({"recovery/status":"snapshot_started"},3)
        assert uploads()==initial+2
        assert json.loads((tmp_path/"training-status.json").read_text())["context"]["phase"]=="checkpoint_snapshot"
        t.close()
