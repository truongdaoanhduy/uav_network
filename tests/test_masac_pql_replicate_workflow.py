from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/uav_masac_pql_replicate.yml"
PINNED_COMMIT = "0ac776bd20f1abd472c0ed016bb9e3d95f7a95b4"


def test_replicate_workflow_is_manual_only_and_pins_scientific_identity():
    assert WORKFLOW.is_file()
    source = WORKFLOW.read_text()
    data = yaml.safe_load(source)
    events = data.get("on", data.get(True))
    assert set(events) == {"workflow_dispatch"}
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


def test_replicate_workflow_keeps_ephemeral_credentials_outside_repo():
    source = WORKFLOW.read_text()
    assert "/tmp/uav-replicate-key.json" in source
    assert "unlink(missing_ok=True)" in source
    assert "outputs/masac-pql-replicate-launch.json" in source
    assert "actions/upload-artifact@v4" in source
