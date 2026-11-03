"""An owner can submit one matched case without exposing other Kaggle accounts."""
import json

import pytest

from scripts.run_owned_kaggle_study_case import build_case, single_owner_credentials
from scripts.prepare_matched_architecture_study import prepare, scientific_settings
from uav_network.configuration import load_config


def test_each_remote_case_matches_prepared_reference(tmp_path):
    rows = prepare(tmp_path / "cases")["cases"]
    for number, row in enumerate(rows, 1):
        selected = build_case(number)
        actual = load_config(selected["config"], selected["overrides"])
        expected = load_config(row["config"])
        assert selected["algorithm"] == row["algorithm"]
        assert selected["name"] == row["name"]
        assert scientific_settings(actual) == scientific_settings(expected)
    assert build_case(4)["overrides"] != build_case(10)["overrides"]  # output only
    four = load_config(build_case(4)["config"], build_case(4)["overrides"])
    ten = load_config(build_case(10)["config"], build_case(10)["overrides"])
    assert scientific_settings(four) == scientific_settings(ten)


def test_owner_credentials_do_not_include_other_accounts():
    data = {"kaggle": {"accounts": [
        {"name": "account_01", "token": "other-secret"},
        {"name": "account_02", "token": "another-secret"}]},
        "wandb": {"api_key": "team-wandb", "entity": "team", "project": "project"},
        "huggingface": {"token": "team-hf"}}
    result = single_owner_credentials(data, "OWNER_API_TOKEN")
    assert len(result["kaggle"]["accounts"]) == 1
    assert result["kaggle"]["accounts"][0]["token"] == "OWNER_API_TOKEN"
    assert "other-secret" not in json.dumps(result)
    assert "another-secret" not in json.dumps(result)
    assert result["wandb"] == data["wandb"]
    assert result["huggingface"] == data["huggingface"]


def test_invalid_case_rejected():
    with pytest.raises(ValueError, match="1..10"):
        build_case(0)
    with pytest.raises(ValueError, match="1..10"):
        build_case(11)


def test_submit_uses_only_owner_token_and_keeps_receipt_secret_free(tmp_path, monkeypatch):
    import scripts.run_owned_kaggle_study_case as launcher
    import uav_network.cloud.cli as cloud_cli
    import uav_network.cloud.client as client

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "verify_published", lambda: "a" * 40)
    monkeypatch.setattr(client, "resolve_username", lambda account: "genuine-owner")
    team_key = tmp_path / "team.json"
    team_key.write_text(json.dumps({
        "kaggle": {"accounts": [{"name": "other", "token": "other-secret"}]},
        "wandb": {"api_key": "team-key", "entity": "team", "project": "project"},
        "huggingface": {"token": "hf-secret"},
    }))
    submitted = []

    def fake_cloud_main(args):
        assert args[args.index("--account") + 1] == "owner"
        assert args[args.index("--credential-dataset") + 1] == "genuine-owner/uav-cpu-wandb-credential"
        private = __import__("pathlib").Path(args[args.index("--key-file") + 1])
        assert private.stat().st_mode & 0o777 == 0o600
        creds = json.loads(private.read_text())
        assert len(creds["kaggle"]["accounts"]) == 1
        assert creds["kaggle"]["accounts"][0]["token"] == "owner-secret"
        assert "other-secret" not in json.dumps(creds)
        submitted.append(args)
        return {"kernel_url": "https://www.kaggle.com/code/genuine-owner/run-4",
                "wandb_urls": {"masac": "https://wandb.ai/team/project/runs/example"},
                "manifest": "outputs/kaggle/test/launch.json"}

    monkeypatch.setattr(cloud_cli, "main", fake_cloud_main)
    case = launcher.build_case(4)
    result = launcher.submit_case(case, team_key, "owner-secret")
    assert len(submitted) == 1
    receipt = tmp_path / "outputs" / launcher.STUDY / "kaggle-submissions" / "case-04.json"
    assert json.loads(receipt.read_text()) == result
    assert "owner-secret" not in receipt.read_text()
    with pytest.raises(FileExistsError):
        launcher.submit_case(case, team_key, "owner-secret")
    assert len(submitted) == 1
