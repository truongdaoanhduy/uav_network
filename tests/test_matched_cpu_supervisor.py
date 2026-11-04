"""Success-only automation never launches GPU or uses incomplete checkpoints."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.run_owned_kaggle_study_case import build_case
from uav_network.configuration import load_config
from uav_network.cloud.matched_cpu_supervisor import (
    DEFAULT_MANIFEST, completed_result, cpu_parent, final_artifact,
    reconcile_case, validate_identity,
)
from uav_network.cloud.visualize import build_visualization_spec

STUDY = json.loads(DEFAULT_MANIFEST.read_text())
ROW = STUDY["cases"][0]


def sample_run(*, state="finished", completed=True, artifact=True, case_row=ROW):
    case = build_case(case_row["case"])
    config = load_config(Path(case["config"]), case["overrides"]).to_dict()
    config.update(source_git_commit=STUDY["source_commit"], kaggle_kernel=case_row["kernel_ref"],
                  training_mode="pql" if "pql" in case["name"] else "base")
    summary = {"episodes_completed": 20000 if completed else 4096,
               "global_step": 81000000, "episode_budget_complete": completed,
               "stop_reason": "episode_budget" if completed else "time_budget"}
    art = SimpleNamespace(
        type="model",
        name=f"uav-network-{case['algorithm']}-{case_row['run_id']}:v2",
        qualified_name=f"{STUDY['entity']}/{STUDY['project']}/uav-network-{case['algorithm']}-{case_row['run_id']}:v2",
        metadata={
            "source_run_id": case_row["run_id"], "source_git_commit": STUDY["source_commit"],
            "completed_episodes": 20000 if artifact else 4096,
            "episode_budget_complete": artifact,
        },
        manifest=SimpleNamespace(entries={"checkpoint.json": True}),
    )
    return SimpleNamespace(id=case_row["run_id"], state=state, group=STUDY["group"],
                           name="TRAIN-MASAC-BASE", config=config,
                           summary=summary, logged_artifacts=lambda: [art])


class FakeAPI:
    def __init__(self, run):
        self.run_object = run

    def run(self, ref):
        assert ref.endswith(self.run_object.id)
        return self.run_object


def test_finished_with_complete_artifact_would_only_cpu_visualize():
    run = sample_run()
    output = reconcile_case(FakeAPI(run), ROW, STUDY, dry_run=True, env={})
    assert output["action"] == "would_submit_cpu"
    assert output["cpu_ref"].endswith("-cpu")
    assert output["checkpoint_artifact"].endswith(":v2")


@pytest.mark.parametrize("state", ["crashed", "failed", "running"])
def test_non_finished_never_launches(state):
    with patch("uav_network.cloud.matched_cpu_supervisor.submit_visualization") as launch:
        output = reconcile_case(FakeAPI(sample_run(state=state)), ROW, STUDY,
                                dry_run=False, env={})
    assert output["action"] in {"blocked_failed", "not_finished"}
    launch.assert_not_called()


@pytest.mark.parametrize("completed,artifact", [(False, True), (True, False)])
def test_no_cpu_without_full_completed_checkpoint(completed, artifact):
    output = reconcile_case(FakeAPI(sample_run(completed=completed, artifact=artifact)),
                            ROW, STUDY, dry_run=False, env={})
    assert output["action"] == "completion_or_final_checkpoint_unverified"


def test_rejects_changed_git_commit():
    run = sample_run()
    run.config["source_git_commit"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_identity(run, ROW, STUDY)


def test_when_cpu_exists_do_not_submit_again():
    run = sample_run()
    env = {"KAGGLE_API_TOKEN_ACCOUNT_01": "dummy", "WANDB_API_KEY": "dummy",
           "HF_TOKEN": "dummy"}
    with patch("uav_network.cloud.matched_cpu_supervisor._owned_kernel_exists", return_value=True), \
         patch("uav_network.cloud.matched_cpu_supervisor.submit_visualization") as launch:
        output = reconcile_case(FakeAPI(run), ROW, STUDY, dry_run=False, env=env)
    assert output["action"] == "cpu_kernel_already_exists"
    launch.assert_not_called()


def test_verified_finished_run_launches_cpu_exactly_once():
    run = sample_run()
    parent = cpu_parent(ROW, STUDY, run, build_case(1))
    spec = build_visualization_spec(parent, {"masac": completed_result(run, STUDY)})
    env = {"KAGGLE_API_TOKEN_ACCOUNT_01": "dummy", "WANDB_API_KEY": "dummy",
           "HF_TOKEN": "dummy"}
    mock_result = {"kernel_ref": spec["kernel_ref"],
                   "kernel_url": "https://www.kaggle.com/code/" + spec["kernel_ref"]}
    with patch("uav_network.cloud.matched_cpu_supervisor._owned_kernel_exists", return_value=False), \
         patch("uav_network.cloud.matched_cpu_supervisor.submit_visualization", return_value=mock_result) as launch:
        output = reconcile_case(FakeAPI(run), ROW, STUDY, dry_run=False, env=env)
    assert output["action"] == "submitted_cpu"
    args = launch.call_args.args
    assert args[0]["requested_episodes"] == 20000
    assert args[0]["cpu_eval"]["episodes"] == 30
    assert args[0]["cpu_eval"]["seed"] == 2000000044
    assert args[1]["masac"]["checkpoint_artifact"].endswith(":v2")
    assert args[0]["orchestration"] == "github_actions"
    assert args[0].get("auto_continue") is None
    assert launch.call_count == 1


def test_cpu_parent_uses_exact_r3_study_group():
    study = json.loads((Path(__file__).resolve().parents[1] / "configs/matched-seed44-cpu-automation-r3.json").read_text())
    case = build_case(1)
    run = sample_run(case_row=study["cases"][0])
    parent = cpu_parent(study["cases"][0], study, run, case)
    overrides = parent["overrides"]
    assert f"tracking.group={study['group']}" in overrides
    assert f"runtime.output_dir=outputs/{study['group']}/training/{case['name']}" in overrides
    assert f"tracking.group={STUDY['group']}" not in overrides
