"""GitHub must be the external supervisor without exposing credentials."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/uav_external_supervisor.yml"
SYNC = ROOT / "scripts/sync_github_supervisor_secrets.py"


def test_external_supervisor_workflow_is_scheduled_manual_and_nonoverlapping():
    assert WORKFLOW.is_file()
    source = WORKFLOW.read_text()
    data = yaml.safe_load(source)
    events = data.get("on", data.get(True))
    assert events["schedule"] == [{"cron": "*/15 * * * *"}]
    assert "workflow_dispatch" in events
    assert data["concurrency"] == {
        "group": "uav-external-supervisor",
        "cancel-in-progress": False,
    }
    assert set(data["jobs"]) == {"supervise"}
    job = data["jobs"]["supervise"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] <= 30
    assert "uav_network.cloud.github_supervisor" in source
    assert "continuation-watch" not in source


def test_external_supervisor_workflow_references_masked_secrets_not_values():
    source = WORKFLOW.read_text()
    for index in range(1, 25):
        assert f"secrets.KAGGLE_API_TOKEN_ACCOUNT_{index:02d}" in source
    for name in ("WANDB_API_KEY", "HF_TOKEN"):
        assert f"secrets.{name}" in source
    for name in ("WANDB_ENTITY", "WANDB_PROJECT", "HF_REPO_ID", "KAGGLE_ENABLED_ACCOUNTS"):
        assert f"vars.{name}" in source
    # Real secret prefixes/tokens must never be copied into the workflow.
    assert "KGAT_" not in source


def test_secret_sync_uses_stdin_and_skips_disabled_accounts(tmp_path):
    assert SYNC.is_file()
    from scripts.sync_github_supervisor_secrets import sync_credentials

    secret_values = {
        "enabled-token",
        "disabled-token",
        "wandb-secret",
        "hf-secret",
    }
    key = tmp_path / "key.txt"
    key.write_text(json.dumps({
        "kaggle": {"accounts": [
            {"name": "account_01", "token": "enabled-token", "enabled": True},
            {"name": "account_02", "token": "disabled-token", "enabled": False},
        ]},
        "wandb": {"api_key": "wandb-secret", "entity": "owner", "project": "project"},
        "huggingface": {"token": "hf-secret", "repo_id": "owner/checkpoints"},
    }))
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        class Result:
            stdout = ""
        return Result()

    result = sync_credentials(key, repo="owner/repo", run=run)
    secret_calls = [(command, kwargs) for command, kwargs in calls if command[1:3] == ["secret", "set"]]
    names = {command[3] for command, _ in secret_calls}
    assert names == {"KAGGLE_API_TOKEN_ACCOUNT_01", "WANDB_API_KEY", "HF_TOKEN"}
    assert "KAGGLE_API_TOKEN_ACCOUNT_02" not in names
    for command, kwargs in secret_calls:
        assert kwargs["input"] in secret_values
        assert kwargs["text"] is True and kwargs["check"] is True
        assert not any(value in " ".join(command) for value in secret_values)
    variable_names = {command[3] for command, _ in calls if command[1:3] == ["variable", "set"]}
    assert variable_names == {"WANDB_ENTITY", "WANDB_PROJECT", "HF_REPO_ID", "KAGGLE_ENABLED_ACCOUNTS"}
    enabled_call = next(kwargs for command, kwargs in calls
                        if command[1:4] == ["variable", "set", "KAGGLE_ENABLED_ACCOUNTS"])
    assert enabled_call["input"] == "account_01"
    assert result["secrets"] == sorted(names)
    assert "KAGGLE_ENABLED_ACCOUNTS" in result["variables"]


def test_manual_dry_run_probes_cloud_credentials_before_reconciling():
    data = yaml.safe_load(WORKFLOW.read_text())
    steps = data["jobs"]["supervise"]["steps"]
    probe_index = next((index for index, step in enumerate(steps)
                        if "--check-services" in step.get("run", "")), None)
    assert probe_index is not None, "Cloud credential readiness has not been checked"
    assert steps[probe_index]["if"] == "github.event_name == 'workflow_dispatch' && inputs.dry_run"
    reconcile_index = next(index for index, step in enumerate(steps)
                           if step.get("name") == "Reconcile UAV lineages")
    assert probe_index < reconcile_index


def test_workflow_retains_sanitized_reconcile_report_even_on_failure():
    data = yaml.safe_load(WORKFLOW.read_text())
    steps = data["jobs"]["supervise"]["steps"]
    reconcile = next(step for step in steps if step.get("name") == "Reconcile UAV lineages")
    assert "--report outputs/supervisor-report.json" in reconcile["run"]
    upload = next((step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")), None)
    assert upload is not None
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "outputs/supervisor-report.json"
