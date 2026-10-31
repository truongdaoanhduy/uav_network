from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/uav_external_supervisor.yml"
PINNED_COMMIT = "0ac776bd20f1abd472c0ed016bb9e3d95f7a95b4"


def _workflow_data():
    source = WORKFLOW.read_text()
    return source, yaml.safe_load(source)


def test_supervisor_workflow_has_manual_masac_pql_replicate_mode():
    source, data = _workflow_data()
    events = data.get("on", data.get(True))
    dispatch = events["workflow_dispatch"]
    assert dispatch["inputs"]["launch_masac_pql_replicate"]["type"] == "boolean"
    assert dispatch["inputs"]["launch_masac_pql_replicate"]["default"] is False
    assert set(data["jobs"]) == {"supervise", "replicate"}
    replicate = data["jobs"]["replicate"]
    assert "inputs.launch_masac_pql_replicate" in replicate["if"]
    assert f"ref: {PINNED_COMMIT}" in source
    assert "KAGGLE_API_TOKEN_ACCOUNT_06" in source
    assert "--account', 'account_06'" in source
    assert "--algorithm', 'masac'" in source
    assert "pql.enabled=true" in source
    assert "runtime.seed=44" in source
    assert "runtime.episodes=20000" in source
    assert "runtime.compile_models=false" in source
    assert "tracking.group=uav-xyz-cbf1500-0ac776b-masac-pql-s44" in source
    assert "--gpus-per-run', '2'" in source
    assert "--auto-continue" in source
    assert "warm-start" not in source.lower()
    assert "KGAT_" not in source


def test_supervisor_replicate_mode_does_not_change_scheduled_supervision():
    source, data = _workflow_data()
    supervise = data["jobs"]["supervise"]
    assert "launch_masac_pql_replicate" in supervise["if"]
    assert data.get("on", data.get(True))["schedule"] == [{"cron": "*/15 * * * *"}]
    assert "/tmp/uav-replicate-key.json" in source
    assert "unlink(missing_ok=True)" in source
    assert "outputs/masac-pql-replicate-launch.json" in source
    assert "actions/upload-artifact@v4" in source
