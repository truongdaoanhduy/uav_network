"""Optional evidence must remain useful without creating or replacing a failure."""
import errno
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from uav_network.configuration import Config
from uav_network.experiments.diagnostics import TrainingDiagnostics, failure_report


def test_warning_console_failure_keeps_disk_event(tmp_path):
    diagnostics=TrainingDiagnostics(Config(),tmp_path)
    closed=StringIO();closed.close()
    with patch("sys.stdout",closed):
        event=diagnostics.event("delivery_gap",4,{"delivery":0})
    assert json.loads((tmp_path/"diagnostics.jsonl").read_text())["code"]=="delivery_gap"
    assert event["root_cause_confirmed"] is False


def test_optional_diagnostic_disk_failure_does_not_abort_or_repeat_cohort(tmp_path,capsys):
    config=Config();config.world.max_steps=2;config.runtime.num_envs=1
    diagnostics=TrainingDiagnostics(config,tmp_path)
    record={"seed":44,"return":2.,"search_rate_percent":80.,"delivery_rate_percent":20.,"success":False}
    with patch.object(Path,"open",side_effect=OSError(errno.ENOSPC,"disk full")):
        diagnostics.event("delivery_gap",1,{"delivery":0})
        first=diagnostics.batch([record],2)
    second=diagnostics.batch([record],4)
    assert first["cohort/episodes"]==1 and second["cohort/episodes"]==1
    assert second["cohort/source_start_step"]==2
    assert 'delivery_gap' in capsys.readouterr().out


def test_failure_report_preserves_nested_nonfinite_context_and_redacts_all_channels(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv("GH_TOKEN","synthetic-github-private-value")
    monkeypatch.setenv("WANDB_API_KEY","synthetic-wandb-private-value")
    class Unsafe:
        def __repr__(self):raise AssertionError("must not repr tensor-like objects")
    path=failure_report(None,tmp_path,RuntimeError("primary synthetic-github-private-value"),{
        "last_learner_metrics":{"q":np.float32(float("nan")),"count":np.int64(4)},
        "request":{"Authorization":"Bearer unknown-private-value","password":"do-not-copy"},
        "payload":Unsafe(),"note":"synthetic-wandb-private-value"})
    report=json.loads(path.read_text());combined=path.read_text()+capsys.readouterr().out
    assert report["error_type"]=="RuntimeError" and report["root_cause_confirmed"] is False
    assert report["context"]["last_learner_metrics"]=={"q":"NaN","count":4}
    assert report["context"]["payload"]=="<Unsafe>"
    for private in ("synthetic-github-private-value","synthetic-wandb-private-value","unknown-private-value","do-not-copy"):
        assert private not in combined


def test_failure_chain_and_resources_keep_wrapper_separate_from_cause(tmp_path):
    try:
        try:raise MemoryError("allocation refused")
        except MemoryError as cause:raise RuntimeError("learner wrapper") from cause
    except RuntimeError as error:
        report=json.loads(failure_report(None,tmp_path,error,{"global_step":8}).read_text())
    assert report["error_type"]=="RuntimeError"
    assert [x["error_type"] for x in report["error_chain"]]==["RuntimeError","MemoryError"]
    assert report["error_category"]=="memory_allocation"
    assert report["root_cause_confirmed"] is False
    assert report["resources"]["disk_free_bytes"]>=0
    assert report["resources"]["process_peak_rss_bytes"]>0


def test_failure_disk_write_cannot_hide_original_console_evidence(tmp_path,capsys):
    with patch.object(Path,"open",side_effect=OSError(errno.ENOSPC,"disk full")):
        path=failure_report(None,tmp_path,ValueError("primary failure"),{"global_step":8})
    assert path==tmp_path/"failure.json"
    assert 'primary failure' in capsys.readouterr().out


def test_async_status_file_and_upload_are_optional_for_actual_evaluation_state(tmp_path):
    from uav_network.experiments.async_evaluation import AsyncCheckpointEvaluator
    diagnostics=TrainingDiagnostics(Config(),tmp_path);logged=[]
    tracker=SimpleNamespace(run=SimpleNamespace(save=lambda *a,**k:(_ for _ in ()).throw(OSError("upload unavailable"))),diagnostic=lambda values,step:logged.append(values))
    evaluator=AsyncCheckpointEvaluator(Config(),tmp_path,tracker,diagnostics)
    with patch.object(Path,"open",side_effect=OSError(errno.ENOSPC,"disk full")):
        evaluator._event("submitted",4,{"global_step":4})
    assert evaluator.counts["submitted"]==1
    assert logged[-1]["evaluation_status/submitted"]==1


@pytest.mark.parametrize("pql",[False,True])
def test_reporting_console_failure_preserves_primary_training_exception(tmp_path,pql):
    from test_continuation import small_config
    from uav_network.experiments.training import train
    config=small_config(tmp_path/"run");config.pql.enabled=pql
    closed=StringIO();closed.close()
    module="uav_network.experiments.pql_training" if pql else "uav_network.experiments.training"
    startup="PQLSystem" if pql else "restore_learner"
    with patch(module+"."+startup,side_effect=RuntimeError("primary learner startup")),patch(module+".failure_report",side_effect=OSError("optional report unavailable")),patch("sys.stdout",closed):
        with pytest.raises(RuntimeError,match="primary learner startup"):
            train(config)


def test_unknown_mapping_keys_never_invoke_arbitrary_repr(tmp_path):
    class UnsafeKey:
        def __repr__(self):raise AssertionError("must not repr unknown keys")
    report=json.loads(failure_report(None,tmp_path,RuntimeError("primary"),{UnsafeKey():4}).read_text())
    assert report["context"]=={"<UnsafeKey>":4}


def test_torch_allocation_category_does_not_infer_a_cuda_backend(tmp_path):
    import torch
    report=json.loads(failure_report(None,tmp_path,torch.OutOfMemoryError("allocation failed"),{}).read_text())
    assert report["error_category"]=="torch_allocation"
    assert report["root_cause_confirmed"] is False


def test_failed_cuda_sampling_does_not_replace_primary_failure(tmp_path,monkeypatch):
    import sys
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:(_ for _ in ()).throw(RuntimeError("CUDA unavailable"))))
    monkeypatch.setitem(sys.modules,"torch",fake)
    report=json.loads(failure_report(None,tmp_path,ValueError("primary failure"),{}).read_text())
    assert report["error_type"]=="ValueError"
    assert report["resources"]["cuda_sampling_error"]=="RuntimeError"


def test_cpu_failure_identity_uses_its_actual_run_and_stage_not_restored_env(tmp_path,monkeypatch):
    monkeypatch.setenv("UAV_STAGE","train");monkeypatch.setenv("WANDB_RUN_ID","parent-gpu")
    (tmp_path/"wandb_run.json").write_text(json.dumps({"wandb_run_id":"cpu-actual"}))
    report=json.loads(failure_report(None,tmp_path,ValueError("primary"),{"stage":"visualize","source_commit":"b"*40}).read_text())
    assert report["identity"]["stage"]=="visualize"
    assert report["identity"]["run_id"]=="cpu-actual"
    assert report["identity"]["source_commit"]=="b"*40
