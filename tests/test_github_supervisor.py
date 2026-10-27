"""The GitHub reconciler owns transitions and must be idempotent."""
from unittest.mock import patch

from test_cloud_continuation import parent_spec, learner_result
from uav_network.cloud.credentials import Account, Credentials


def fixtures():
    from uav_network.cloud import github_supervisor as supervisor
    account = Account("selected", "synthetic-token")
    credentials = Credentials([account], "synthetic-wandb", "owner", "project")
    parent = parent_spec()
    parent["orchestration"] = "github_actions"
    parent["external_supervisor_required"] = True
    return supervisor, account, credentials, parent


def controls(parent, results):
    return {"job.json": parent, "job-result.json": results,
            "continuation.json": {"status": "external_pending"}}


def test_active_kernel_is_observed_without_reading_outputs_or_submitting():
    s, account, credentials, parent = fixtures()
    with patch.object(s, "kernel_status", return_value="running"), \
         patch.object(s, "read_controls", side_effect=AssertionError("active output")), \
         patch.object(s, "submit_continuation", side_effect=AssertionError("submit")):
        result = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert result == {"ref": parent["kernel_ref"], "status": "running", "action": "active"}


def test_completed_incomplete_segment_submits_one_gpu_child_and_then_adopts_it():
    s, account, credentials, parent = fixtures()
    result = learner_result(complete=False)
    child_ref = s.kernel_ref_for_segment(parent["root_kernel_ref"], 1)
    with patch.object(s, "kernel_status", return_value="complete"), \
         patch.object(s, "read_controls", return_value=controls(parent, result)), \
         patch.object(s, "_owned_kernel_exists", side_effect=[False, True]), \
         patch.object(s, "submit_continuation", return_value={"kernel_ref": child_ref, "status": "submitted"}) as submit:
        first = s.reconcile_ref(parent["kernel_ref"], account, credentials)
        second = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert first["action"] == "submitted_gpu" and first["child_ref"] == child_ref
    assert second["action"] == "existing_gpu" and second["child_ref"] == child_ref
    assert submit.call_count == 1


def test_completed_episode_budget_submits_cpu_visualization_once():
    s, account, credentials, parent = fixtures()
    result = learner_result(complete=True)
    cpu_ref = s.kernel_ref_for_segment(parent["root_kernel_ref"], 0, cpu=True)
    with patch.object(s, "kernel_status", return_value="complete"), \
         patch.object(s, "read_controls", return_value=controls(parent, result)), \
         patch.object(s, "_owned_kernel_exists", return_value=False), \
         patch.object(s, "submit_visualization", return_value={"kernel_ref": cpu_ref}) as submit:
        observed = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert observed["action"] == "submitted_cpu"
    assert observed["child_ref"] == cpu_ref
    assert submit.call_count == 1


def test_dry_run_reports_transition_without_push():
    s, account, credentials, parent = fixtures()
    result = learner_result(complete=False)
    with patch.object(s, "kernel_status", return_value="complete"), \
         patch.object(s, "read_controls", return_value=controls(parent, result)), \
         patch.object(s, "_owned_kernel_exists", return_value=False), \
         patch.object(s, "submit_continuation", side_effect=AssertionError("dry-run push")):
        observed = s.reconcile_ref(parent["kernel_ref"], account, credentials, dry_run=True)
    assert observed["action"] == "would_submit_gpu"


def test_unrelated_legacy_job_is_ignored():
    s, account, credentials, parent = fixtures()
    parent.pop("orchestration")
    with patch.object(s, "kernel_status", return_value="complete"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}):
        observed = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert observed["action"] == "ignored"


def test_terminal_failed_job_is_blocked_until_verified_recovery():
    s, account, credentials, parent = fixtures()
    with patch.object(s, "kernel_status", return_value="error"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}), \
         patch.object(s, "verified_recovery_result", side_effect=s.RecoveryUnavailable("missing")):
        observed = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert observed["action"] == "recovery_blocked"


def test_partial_cpu_evaluation_retries_from_saved_progress_once():
    s, account, credentials, parent = fixtures()
    cpu_ref = s.kernel_ref_for_segment(parent["root_kernel_ref"], 0, cpu=True)
    cpu = {
        "stage": "visualize", "orchestration": "github_actions", "account": account.name,
        "kernel_ref": cpu_ref, "root_kernel_ref": parent["root_kernel_ref"],
        "parent_kernel_ref": parent["kernel_ref"], "commit": parent["commit"],
        "entity": parent["entity"], "project": parent["project"],
        "credential_dataset": parent["cpu_credential_dataset"],
        "pipeline_credential_dataset": parent["pipeline_credential_dataset"],
        "storage_credential_dataset": parent["storage_credential_dataset"],
        "checkpoint_repo": parent["checkpoint_repo"],
        "artifacts": {"masac": "owner/project/policy:v5"},
        "cpu_eval": parent["cpu_eval"], "runs": {"masac": {"id": "cpu", "name": "eval"}},
        "training_mode": "base", "timeout_seconds": 39600,
    }
    progress = {"artifacts": cpu["artifacts"], "cpu_eval": cpu["cpu_eval"],
                "records": {"masac": [{"episode": 1, "seed": cpu["cpu_eval"]["seed"]}]}}
    retry_ref = s.cpu_retry_ref(parent["root_kernel_ref"], 1)
    calls = []

    def submit(spec, output, selected, creds):
        calls.append(spec)
        return {**spec, "kernel_ref": spec["kernel_ref"]}

    with patch.object(s, "kernel_status", return_value="complete"), \
         patch.object(s, "read_controls", return_value={"job.json": cpu, "evaluation-progress.json": progress}), \
         patch.object(s, "_owned_kernel_exists", side_effect=[False, True]), \
         patch.object(s, "submit_visualization_spec", side_effect=submit):
        first = s.reconcile_ref(cpu_ref, account, credentials)
        second = s.reconcile_ref(cpu_ref, account, credentials)
    assert first == {"ref": cpu_ref, "status": "complete", "action": "submitted_cpu_retry", "child_ref": retry_ref}
    assert second["action"] == "existing_cpu_retry"
    assert calls[0]["completed_records"] == progress["records"]
    assert calls[0]["cpu_attempt"] == 1
    assert len(calls) == 1


class FakeArtifact:
    def __init__(self, metadata, name="owner/project/uav-network-masac-deadbeef:v7"):
        self.metadata = metadata
        self.qualified_name = name
        self.manifest = type("Manifest", (), {"entries": {"checkpoint.json": object()}})()


class FakeRun:
    def __init__(self, summary, state="crashed"):
        self.summary = summary
        self.state = state


class FakeWandbApi:
    def __init__(self, run, artifact):
        self._run, self._artifact = run, artifact
    def run(self, path):
        return self._run
    def artifact(self, ref, type=None):
        assert ref == self._artifact.qualified_name and type == "model"
        return self._artifact


def good_recovery_api(parent, *, resumable=True, source_run=None, source_commit=None):
    run_id = parent["runs"]["masac"]["id"]
    ref = "owner/project/uav-network-masac-deadbeef:v7"
    summary = {"recovery/checkpoint_artifact": ref,
               "recovery/last_verified_episodes": 640,
               "recovery/last_verified_global_step": 123456}
    metadata = {"resumable": resumable, "completed_episodes": 640, "global_step": 123456,
                "episode_budget_complete": False,
                "source_run_id": source_run or run_id,
                "source_git_commit": source_commit or parent["commit"]}
    return FakeWandbApi(FakeRun(summary), FakeArtifact(metadata, ref))


def test_verified_recovery_requires_immutable_resumable_matching_provenance():
    s, account, credentials, parent = fixtures()
    result = s.verified_recovery_result(parent, credentials, api=good_recovery_api(parent))
    recovered = result["masac"]
    assert recovered["episodes"] == 640
    assert recovered["global_step"] == 123456
    assert recovered["checkpoint_artifact"].endswith(":v7")
    assert recovered["recovery_source_state"] == "crashed"

    for api in (
        good_recovery_api(parent, resumable=False),
        good_recovery_api(parent, source_run="wrong-run"),
        good_recovery_api(parent, source_commit="f" * 40),
    ):
        try:
            s.verified_recovery_result(parent, credentials, api=api)
        except s.RecoveryUnavailable:
            pass
        else:
            raise AssertionError("invalid recovery provenance was accepted")


def test_crashed_job_with_verified_recovery_submits_exact_continuation_once():
    s, account, credentials, parent = fixtures()
    recovered = s.verified_recovery_result(parent, credentials, api=good_recovery_api(parent))
    child_ref = s.kernel_ref_for_segment(parent["root_kernel_ref"], 1)
    with patch.object(s, "kernel_status", return_value="error"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}), \
         patch.object(s, "verified_recovery_result", return_value=recovered), \
         patch.object(s, "_owned_kernel_exists", side_effect=[False, True]), \
         patch.object(s, "submit_continuation", return_value={"status": "submitted", "kernel_ref": child_ref}) as submit:
        first = s.reconcile_ref(parent["kernel_ref"], account, credentials)
        second = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert first["action"] == "submitted_recovery_gpu"
    assert second["action"] == "existing_recovery_gpu"
    assert submit.call_count == 1
    assert submit.call_args.args[1]["masac"]["episodes"] == 640


def test_crashed_job_without_exact_recovery_blocks_without_push():
    s, account, credentials, parent = fixtures()
    with patch.object(s, "kernel_status", return_value="error"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}), \
         patch.object(s, "verified_recovery_result", side_effect=s.RecoveryUnavailable("no full state")), \
         patch.object(s, "submit_continuation", side_effect=AssertionError("warm restart")):
        observed = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert observed["action"] == "recovery_blocked"
    assert observed["error_type"] == "RecoveryUnavailable"


def test_environment_credentials_use_explicit_enabled_account_allowlist():
    from uav_network.cloud import github_supervisor as s
    env = {
        "KAGGLE_ENABLED_ACCOUNTS": "account_01,account_03",
        "KAGGLE_API_TOKEN_ACCOUNT_01": "token-one",
        "KAGGLE_API_TOKEN_ACCOUNT_02": "stale-disabled-token",
        "KAGGLE_API_TOKEN_ACCOUNT_03": "token-three",
        "WANDB_API_KEY": "wandb",
        "WANDB_ENTITY": "owner",
        "WANDB_PROJECT": "project",
        "HF_TOKEN": "hf",
        "HF_REPO_ID": "owner/repo",
    }
    credentials = s.credentials_from_environment(env)
    assert [account.name for account in credentials.accounts] == ["account_01", "account_03"]
    assert all(account.name != "account_02" for account in credentials.accounts)


def test_environment_credentials_fail_closed_when_allowlisted_token_is_missing():
    from uav_network.cloud import github_supervisor as s
    env = {
        "KAGGLE_ENABLED_ACCOUNTS": "account_01,account_03",
        "KAGGLE_API_TOKEN_ACCOUNT_01": "token-one",
        "WANDB_API_KEY": "wandb",
        "WANDB_ENTITY": "owner",
        "WANDB_PROJECT": "project",
        "HF_TOKEN": "hf",
        "HF_REPO_ID": "owner/repo",
    }
    import pytest
    with pytest.raises(RuntimeError, match="enabled account_03"):
        s.credentials_from_environment(env)


def test_owned_refs_filters_legacy_kernels_before_reconciliation():
    s, account, credentials, parent = fixtures()
    rows = [
        {"ref": "owner/old-training"},
        {"ref": "owner/gh-tra-g3-masac-base-1027-ab1234"},
        {"ref": "owner/gh-tra-g3-masac-base-1027-ab1234-c01"},
        {"ref": "owner/gh-tra-g3-masac-base-1027-ab1234-cpu"},
        {"ref": "owner/notebook123"},
    ]
    with patch.object(s, "run_kaggle", return_value=__import__("json").dumps(rows)):
        refs = s._owned_refs(account, 100)
    assert refs == [row["ref"] for row in rows if "/gh-" in row["ref"]]


def test_reconcile_account_filters_unmanaged_kaggle_refs_before_reconciliation():
    s, account, credentials, _parent = fixtures()
    refs = [
        "owner/notebookbf403d32a3",
        "owner/uavnet-train-20261016-081807-d07bdf",
        "owner/uav-memory-1m-20261020-122024",
        "owner/gh-tra-g3-masac-base-h75-10271003-abc123",
        "owner/gh-val-g3-matd3-pql-fullgpu-10271004-def456-c01",
        "owner/uav-media-0123456789abcd-a01",
    ]
    seen = []

    def reconcile(ref, selected, creds, *, dry_run=False):
        seen.append(ref)
        return {"ref": ref, "action": "checked"}

    with patch.object(s, "_owned_refs", return_value=refs), \
         patch.object(s, "reconcile_ref", side_effect=reconcile):
        outcomes = s.reconcile_account(account, credentials, dry_run=True)

    assert seen == refs[3:]
    assert [row["ref"] for row in outcomes] == refs[3:]


def test_managed_ref_matches_reserved_github_lineages_and_cpu_retries_only():
    from uav_network.cloud import github_supervisor as s
    assert s._managed_ref("owner/gh-tra-g3-masac-base-h75-10271003-abc123")
    assert s._managed_ref("owner/gh-val-g3-matd3-pql-fullgpu-10271004-def456-c01")
    assert s._managed_ref("owner/uav-media-0123456789abcd-a01")
    assert not s._managed_ref("owner/tra-g3-masac-base-h75-10271003-abc123")
    assert not s._managed_ref("owner/uavnet-train-20261016-081807-d07bdf")


def test_owned_refs_keeps_reserved_cpu_retry_namespace():
    from uav_network.cloud import github_supervisor as s
    account = Account("selected", "synthetic-token")
    rows = [
        {"ref": "owner/gh-tra-g3-masac-base-h75-10271003-abc123"},
        {"ref": "owner/uav-media-0123456789abcd-a01"},
        {"ref": "owner/notebook123"},
    ]
    with patch.object(s, "run_kaggle", return_value=__import__("json").dumps(rows)):
        assert s._owned_refs(account, 100) == [rows[0]["ref"], rows[1]["ref"]]


def test_service_probe_authenticates_reads_and_never_reconciles(monkeypatch, capsys):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    import wandb
    import huggingface_hub
    s, account, credentials, parent = fixtures()
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-secret")
    monkeypatch.setenv("HF_REPO_ID", "owner/checkpoints")
    project_reads = []
    def project_rows():
        project_reads.append("fetched")
        yield object()
    wb = Mock(viewer=object())
    wb.runs.return_value = project_rows()
    hf = Mock()
    hf.whoami.return_value = {"name": "owner"}
    hf.repo_info.return_value = SimpleNamespace(id="owner/checkpoints")
    with patch.object(s, "credentials_from_environment", return_value=credentials), \
         patch.object(s, "reconcile_account", side_effect=AssertionError("probe submitted")), \
         patch.object(s, "resolve_username", return_value="owner") as username, \
         patch.object(wandb, "Api", return_value=wb) as wb_constructor, \
         patch.object(huggingface_hub, "HfApi", return_value=hf) as hf_constructor:
        code = s.main(["--check-services"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report == {"wandb": {"status": "ok"}, "huggingface": {"status": "ok"},
                      "kaggle/selected": {"status": "ok", "username": "owner"}}
    username.assert_called_once_with(account)
    wb_constructor.assert_called_once_with(api_key="synthetic-wandb", timeout=30)
    wb.runs.assert_called_once_with("owner/project", per_page=1)
    assert project_reads == ["fetched"]
    hf_constructor.assert_called_once_with(token="synthetic-hf-secret")
    hf.whoami.assert_called_once_with()
    hf.repo_info.assert_called_once_with("owner/checkpoints", repo_type="model", timeout=30)


def test_service_probe_failure_is_nonzero_and_redacts_remote_error(monkeypatch, capsys):
    import json
    from unittest.mock import Mock
    import wandb
    import huggingface_hub
    s, account, credentials, parent = fixtures()
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf-secret")
    monkeypatch.setenv("HF_REPO_ID", "owner/checkpoints")
    def denied_project():
        raise RuntimeError("synthetic-wandb synthetic-hf-secret private-details")
        yield
    for failed_service in ("wandb", "huggingface", "kaggle/selected"):
        wb = Mock(viewer=object())
        wb.runs.return_value = denied_project() if failed_service == "wandb" else iter([])
        hf = Mock()
        hf.whoami.return_value = {"name": "owner"}
        if failed_service == "huggingface":
            hf.repo_info.side_effect = RuntimeError("synthetic-hf-secret private-details")
        username_error = RuntimeError("synthetic-hf-secret private-details") if failed_service == "kaggle/selected" else None
        with patch.object(s, "credentials_from_environment", return_value=credentials), \
             patch.object(s, "reconcile_account", side_effect=AssertionError("probe submitted")), \
             patch.object(s, "resolve_username", return_value="owner", side_effect=username_error), \
             patch.object(wandb, "Api", return_value=wb), \
             patch.object(huggingface_hub, "HfApi", return_value=hf):
            code = s.main(["--check-services"])
        output = capsys.readouterr()
        report = json.loads(output.out)
        assert code == 1
        assert report[failed_service] == {"status": "error", "error_type": "RuntimeError"}
        assert all(result["status"] == "ok" for service, result in report.items() if service != failed_service)
        assert "synthetic-wandb" not in output.out + output.err
        assert "synthetic-hf-secret" not in output.out + output.err
        assert "private-details" not in output.out + output.err


def test_one_unavailable_account_does_not_starve_other_accounts(capsys):
    import json
    s, account, credentials, _ = fixtures()
    credentials.accounts.append(Account("healthy", "another-synthetic-token"))
    checked = []
    def reconcile(selected, creds, **kwargs):
        checked.append(selected.name)
        if selected.name == "selected":
            raise RuntimeError("synthetic-token private-error-details")
        return [{"ref": "owner/gh-tra-g3-healthy", "action": "active"}]
    with patch.object(s, "credentials_from_environment", return_value=credentials), \
         patch.object(s, "reconcile_account", side_effect=reconcile):
        code = s.main([])
    output = capsys.readouterr().out
    report = json.loads(output)
    assert code == 1 and checked == ["selected", "healthy"]
    assert report["selected"][0]["action"] == "account_unavailable"
    assert report["healthy"][0]["action"] == "active"
    assert "synthetic-token" not in output and "private-error-details" not in output


def test_unresolved_recovery_and_unknown_service_are_not_reported_green(capsys):
    s, account, credentials, _ = fixtures()
    for action in ("recovery_blocked", "recovery_required", "service_unavailable", "reconcile_error"):
        with patch.object(s, "credentials_from_environment", return_value=credentials), \
             patch.object(s, "reconcile_account", return_value=[{"action": action}]):
            assert s.main([]) == 1, action
        capsys.readouterr()


def test_sanitized_report_survives_account_failure(tmp_path, capsys):
    import json
    s, account, credentials, _ = fixtures()
    path = tmp_path / "reports" / "reconcile.json"
    with patch.object(s, "credentials_from_environment", return_value=credentials), \
         patch.object(s, "reconcile_account", side_effect=TimeoutError("synthetic-wandb secret-error")):
        assert s.main(["--report", str(path)]) == 1
    data = path.read_text()
    assert json.loads(data)["selected"][0]["error_type"] == "TimeoutError"
    assert "synthetic-wandb" not in data and "secret-error" not in data
    assert json.loads(data) == json.loads(capsys.readouterr().out)


def test_verified_recovery_of_continuation_keeps_its_initial_global_step():
    s, account, credentials, parent = fixtures()
    parent.update(continuation_index=1, continuation_expected_episodes={"masac": 300},
                  continuation_expected_global_step={"masac": 1000})
    result = s.verified_recovery_result(parent, credentials, api=good_recovery_api(parent))
    assert s.next_stage(parent, result) == "gpu"
    assert result["masac"]["initial_global_step"] == 1000
    from uav_network.cloud.continuation import build_continuation_spec
    child = build_continuation_spec(parent, result)
    assert child["continuation_expected_global_step"] == {"masac": 123456}


def test_zero_episode_recovery_needs_verified_environment_state():
    import pytest
    s, account, credentials, parent = fixtures()
    for marker in (True, False):
        api = good_recovery_api(parent)
        api._run.summary["recovery/last_verified_episodes"] = 0
        api._artifact.metadata.update(completed_episodes=0, environment_state_saved=marker)
        if not marker:
            with pytest.raises(s.RecoveryUnavailable):
                s.verified_recovery_result(parent, credentials, api=api)
            continue
        result = s.verified_recovery_result(parent, credentials, api=api)
        assert s.next_stage(parent, result) == "gpu"
        assert result["masac"]["initial_global_step"] == 0
        assert result["masac"]["environment_state_saved"] is True


def test_cancelled_gpu_and_cpu_are_not_automatically_restarted():
    s, account, credentials, parent = fixtures()
    for stage in ("train", "visualize"):
        parent["stage"] = stage
        with patch.object(s, "kernel_status", return_value="cancelled"), \
             patch.object(s, "read_controls", side_effect=AssertionError("cancelled job was adopted")):
            result = s.reconcile_ref(parent["kernel_ref"], account, credentials)
        assert result["action"] == "cancelled"


def test_recovery_waiting_for_quota_is_not_a_submitted_child():
    s, account, credentials, parent = fixtures()
    recovered = s.verified_recovery_result(parent, credentials, api=good_recovery_api(parent))
    with patch.object(s, "kernel_status", return_value="error"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}), \
         patch.object(s, "verified_recovery_result", return_value=recovered), \
         patch.object(s, "_owned_kernel_exists", return_value=False), \
         patch.object(s, "submit_continuation", return_value={"status": "waiting_for_quota", "wait_reason": "insufficient_quota"}):
        result = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert result["action"] == "waiting_for_quota"
    assert result["wait_reason"] == "insufficient_quota"


def test_consecutive_crash_recovery_is_bounded_and_clean_segment_resets_it():
    import pytest
    from uav_network.cloud.continuation import build_continuation_spec
    s, account, credentials, parent = fixtures()
    parent["consecutive_recoveries"] = 2
    results = learner_result()
    results["masac"]["exact_recovery"] = True
    assert build_continuation_spec(parent, results)["consecutive_recoveries"] == 3
    parent["consecutive_recoveries"] = 3
    with pytest.raises(ValueError, match="recovery budget"):
        build_continuation_spec(parent, results)
    del results["masac"]["exact_recovery"]
    assert build_continuation_spec(parent, results)["consecutive_recoveries"] == 0


def test_exhausted_recovery_budget_blocks_before_network_or_submission():
    s, account, credentials, parent = fixtures()
    parent["consecutive_recoveries"] = 3
    with patch.object(s, "kernel_status", return_value="error"), \
         patch.object(s, "read_controls", return_value={"job.json": parent}), \
         patch.object(s, "verified_recovery_result", side_effect=AssertionError("unbounded recovery")):
        result = s.reconcile_ref(parent["kernel_ref"], account, credentials)
    assert result["action"] == "recovery_blocked"
    assert result["failure_reason"] == "recovery_budget_exhausted"
