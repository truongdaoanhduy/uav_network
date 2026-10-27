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
