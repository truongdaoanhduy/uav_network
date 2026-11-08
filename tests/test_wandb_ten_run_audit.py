"""Offline regression checks for a real, pinned 10-case W&B experiment snapshot."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_latest_ten_wandb_baselines import (
    matched_pair, run_case, stable_hash, training_slice,
)
from scripts.render_wandb_ten_case_registry import registry

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs/methods/evidence/wandb_latest_ten_2026-11-08.json"


def test_training_cutoff_excludes_future_episodes():
    rows = np.array([
        [i, 10.0 * i, 4 * i, 2 * i, i * 4096, 30] for i in range(1, 8)
    ], dtype=float)
    selected = training_slice(rows, 3, tail=2)
    assert selected["episode"] == 3
    assert selected["global_step_at_episode"] == 3 * 4096
    assert selected["return"] == 25.0
    assert selected["cumulative_return"] == 20.0


def test_pairwise_uses_smaller_completed_episode_and_latest_shared_eval():
    a_rows = np.array([[i, 2 * i, i, 2 * i, i * 8, 10] for i in range(1, 7)], float)
    b_rows = np.array([[i, i, i, i, i * 8, 10] for i in range(1, 5)], float)
    a = {"rows": a_rows, "eval_history": [
        {"checkpoint_global_step": 10, "delivery": 11, "return": -2},
        {"checkpoint_global_step": 20, "delivery": 20, "return": 3},
        {"checkpoint_global_step": 30, "delivery": 99, "return": 99},
    ]}
    b = {"rows": b_rows, "eval_history": [
        {"checkpoint_global_step": 10, "delivery": 5, "return": -3},
        {"checkpoint_global_step": 20, "delivery": 15, "return": 2},
    ]}
    pair = matched_pair(a, b)
    assert pair["episode_cutoff"] == 4
    assert pair["left"]["episode"] == pair["right"]["episode"] == 4
    assert pair["left"]["delivery"] == 5.0
    assert pair["right"]["delivery"] == 2.5
    assert pair["validation_at_latest_shared_checkpoint"]["checkpoint_global_step"] == 20
    assert pair["validation_at_latest_shared_checkpoint"]["delta_delivery_left_minus_right"] == 5


@pytest.mark.parametrize("case,settings", [
    (1, (True, 512, 8, True)), (2, (True, 512, 16, True)),
    (3, (False, 512, 16, True)), (4, (False, 512, 16, False)),
    (5, (True, 256, 8, True)),
])
def test_versioned_architecture_mapping(case, settings):
    shared, width, updates, layernorm = settings
    c = {"runtime": {"algorithm": "matd3"}, "matd3": {
        "share_actor_parameters": shared, "share_critic_parameters": shared,
        "hidden_sizes": [width, width], "updates_per_step": updates,
        "actor_layer_norm": layernorm,
    }}
    assert run_case(c) == ("matd3", case)


def test_full_source_provenance_and_case_registry_survive_new_chat():
    snapshot = json.loads(SNAPSHOT.read_text())
    assert len(snapshot["runs"]) == 10
    assert {r["source_git_commit"] for r in snapshot["runs"]} == {
        "a6e07a8f4b199028298b6ffd9a47f2cecd654fc9"
    }
    assert len({r["scenario_sha256"] for r in snapshot["runs"]}) == 1
    assert len({r["runtime_protocol_sha256"] for r in snapshot["runs"]}) == 1
    assert len({r["case_fingerprint_sha256"] for r in snapshot["runs"]}) == 10
    assert all(r["validation_at_common_checkpoint"]["checkpoint_global_step"] ==
               7_372_800 for r in snapshot["runs"])
    result = registry(snapshot)
    assert result["full_20000"] == 5 and result["partial"] == 5
    assert result["do_not_repeat_identical_training"] is True
    assert result["primary_result"] == {"algorithm": "matd3", "case": 1,
                                       "reason": result["primary_result"]["reason"]}


def test_pinned_history_does_not_compare_unmatched_best_steps_as_equal_budget():
    s = json.loads(SNAPSHOT.read_text())
    assert s["protocol"]["all_run_common_episode_cutoff"] == 4096
    assert s["protocol"]["all_run_shared_eval_checkpoint_global_step"] == 7372800
    assert s["protocol"]["best_checkpoint_not_used_as_fair_same_budget_ranking"]
    runs = {(r["algorithm"], r["case"]): r for r in s["runs"]}
    assert runs[("matd3", 1)]["observed_episodes"] == 20000
    assert runs[("masac", 1)]["observed_episodes"] == 11675
    assert runs[("masac", 2)]["observed_episodes"] == 4096
    assert runs[("matd3", 3)]["observed_episodes"] == 16384
    assert s["pairwise"]["cross_algorithm_case_01"]["episode_cutoff"] == 11675
    assert s["pairwise"]["cross_algorithm_case_05"]["episode_cutoff"] == 20000


def test_scenario_fingerprint_reacts_to_scientific_changes():
    original = {"world": {"num_targets": 35}, "reward": {"delivery": 50}}
    changed = {"world": {"num_targets": 36}, "reward": {"delivery": 50}}
    assert stable_hash(original) != stable_hash(changed)
