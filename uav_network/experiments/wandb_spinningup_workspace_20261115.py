"""Configure W&B Runs columns + Training/Eval/Test panels, preserving existing panels.

This module NEVER handles API keys. It uses the signed-in W&B SDK context
already provided by Kaggle Secrets (or the user's own wandb login).
By default it performs OFFLINE DRY RUN; --apply is an explicit online write.

The learning curve uses TRUE GPU checkpoint evaluation (16 fixed scenarios),
x = cumulative environment interactions. A single training seed has no
cross-training-seed std band; never invent one.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ENTITY = "truongdaoanhduy-uit"
PROJECT = "uav_search_target"
EXISTING_WORKSPACE_URL = (
    f"https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}/workspace"
    "?nw=nwusertruongdaoanhduy"
)
VIEW_NAME = "UAV Spinning Up Training Eval Test"
RUNS_COLUMNS = (
    "run:displayName",
    "run:state",
    "summary:episodes_completed",
    "summary:gpu_final/discovery_rate_percent",
    "summary:gpu_final/gcs_delivery_rate_percent",
    "summary:gpu_final/discovery_auc",
    "summary:gpu_final/delivery_auc",
    "summary:gpu_final/energy_per_delivered_target_j",
    "summary:gpu_final/gcs_delivery_time_capped_mean_s",
    "summary:gpu_final/map_coverage_percent",
    "summary:gpu_final/mission_success_rate_percent",
    "summary:gpu_final/episodes",
    "summary:pipeline/test_finished",
)

VALIDATION_PLOTS = (
    ("Discovery rate (%) — deterministic validation", "discovery_rate_percent", "Higher is better"),
    ("GCS delivery rate (%) — deterministic validation", "gcs_delivery_rate_percent", "Higher is better"),
    ("Discovery AUC — earlier finds score higher", "discovery_auc", "Higher is better"),
    ("Delivery AUC — earlier GCS reports score higher", "delivery_auc", "Higher is better"),
    ("Energy per delivered target (J)", "energy_per_delivered_target_j", "Lower is better"),
    ("Capped mean GCS delivery time (s)", "gcs_delivery_time_capped_mean_s", "Lower is better"),
    ("Area coverage (%)", "map_coverage_percent", "Higher is better"),
    ("Coverage AUC — faster map coverage", "coverage_auc", "Higher is better"),
    ("Mission success (%)", "mission_success_rate_percent", "Higher is better"),
)

FINAL_TEST_METRICS = (
    ("Final Test — target discovery (%)", "discovery_rate_percent"),
    ("Final Test — GCS delivery (%)", "gcs_delivery_rate_percent"),
    ("Final Test — discovery AUC", "discovery_auc"),
    ("Final Test — delivery AUC", "delivery_auc"),
    ("Final Test — J per delivered target", "energy_per_delivered_target_j"),
    ("Final Test — capped delivery time (s)", "gcs_delivery_time_capped_mean_s"),
    ("Final Test — coverage (%)", "map_coverage_percent"),
    ("Final Test — coverage AUC", "coverage_auc"),
    ("Final Test — complete mission success (%)", "mission_success_rate_percent"),
)

TRAINING_DIAGNOSTICS = (
    # All mission outcomes are observed in actual TRAINING episodes, not TEST.
    # Use global_step for the SAME x coordinate as checkpoint Validation.
    ("TRAIN • Search rate (%)", "training/discovery_rate_percent", "global_step"),
    ("TRAIN • GCS delivery rate (%)", "training/gcs_delivery_rate_percent", "global_step"),
    ("TRAIN • Discovery AUC", "training/discovery_auc", "global_step"),
    ("TRAIN • Delivery AUC", "training/delivery_auc", "global_step"),
    ("TRAIN • Map coverage (%)", "training/map_coverage_percent", "global_step"),
    ("TRAIN • Coverage AUC", "training/coverage_auc", "global_step"),
    ("TRAIN • Energy / delivered target (J)", "training/energy_per_delivered_target_j", "global_step"),
    ("TRAIN • GCS delivery time capped (s)", "training/gcs_delivery_time_capped_mean_s", "global_step"),
    ("TRAIN • Mission success rate (%)", "training/mission_success_rate_percent", "global_step"),
    ("TRAIN • Reward (exploration episodes)", "episode/return", "global_step"),
    ("TRAIN • Actor loss", "learner/actor_loss", "global_step"),
    ("TRAIN • Critic loss", "learner/critic_loss", "global_step"),
)


def manifest():
    return {
        "entity": ENTITY, "project": PROJECT, "existing_workspace_url": EXISTING_WORKSPACE_URL,
        "name_if_new": VIEW_NAME, "runs_columns": list(RUNS_COLUMNS),
        "sections": {
            "EVAL | Deterministic checkpoints — OpenAI SpinningUp-style": [
                {"title": title, "x": "gpu_eval/checkpoint_global_step",
                 "y": f"gpu_eval/{key}", "direction": direction}
                for title, key, direction in VALIDATION_PLOTS
            ],
            "TEST | 100 held-out episodes (single final estimate)": [
                {"title": title, "metric": f"gpu_final/{key}"}
                for title, key in FINAL_TEST_METRICS
            ],
            "TRAINING | Auxiliary diagnostics (not the benchmark curve)": [
                {"title": title, "x": x, "y": y}
                for title, y, x in TRAINING_DIAGNOSTICS
            ],
        },
        "scientific_disclosure": (
            "Checkpoint validation runs 16 fixed evaluation episodes; old/new U8 "
            "notebooks produce GPU Eval after training using saved policies. "
            "Spinning Up original benchmarks use 10 trajectories per evaluation "
            "and average/std across 10 TRAINING seeds. This project currently "
            "has one training seed per algorithm, so NO SD band should appear."
        )
    }


def construct_sections(ws, wr, existing=None):
    validation = ws.Section(
        name="EVAL | Deterministic checkpoint learning curves (Spinning Up style)",
        is_open=True,
        panels=[
            wr.LinePlot(
                title=title, x="gpu_eval/checkpoint_global_step",
                y=[f"gpu_eval/{metric}"], title_x="Environment transitions",
                smoothing_type="average", smoothing_factor=0.35,
                smoothing_show_original=False,
                line_widths={f"gpu_eval/{metric}": 2.0},
                legend_position="south",
            )
            for title, metric, _ in VALIDATION_PLOTS
        ] + [wr.MediaBrowser(
            title="Paper-style validation curves (11-checkpoint moving mean)",
            media_keys=["paper/validation_learning_curves"],
            mode="gallery", num_columns=1,
        )],
    )
    test = ws.Section(
        name="TEST | Held-out final policy (100 episodes)",
        is_open=True,
        panels=[
            wr.ScalarChart(title=title, metric=f"gpu_final/{metric}")
            for title, metric in FINAL_TEST_METRICS
        ],
    )
    training_new = ws.Section(
        name="TRAINING | Mission improvement through learning",
        is_open=True,
        panels=[
            wr.LinePlot(title=title, x=x, y=[metric],
                        title_x=("Episodes" if x == "episode/index" else "Environment transitions"),
                        smoothing_type="average", smoothing_factor=0.25)
            for title, metric, x in TRAINING_DIAGNOSTICS
        ],
    )
    if not existing:
        return [training_new, validation, test]
    leftovers = []
    for section in existing:
        name = section.name or ""
        if name.startswith(("EVAL | Deterministic checkpoint", "TEST | Held-out final policy",
                            "TRAINING | Mission improvement")):
            continue  # replace only our previously generated sections
        if name == "Panel Section" and not section.panels:
            continue  # the empty panel section shown in screenshot
        section.is_open = False
        leftovers.append(section)  # KEEP all old custom panels
    return [training_new, validation, test, *leftovers]


def prepare_workspace():
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    workspace = ws.Workspace(
        entity=ENTITY, project=PROJECT, name=VIEW_NAME,
        sections=construct_sections(ws, wr),
        runset_settings=ws.RunsetSettings(pinned_columns=list(RUNS_COLUMNS)),
        auto_generate_panels=False,
    )
    return workspace


def apply():
    if not os.environ.get("WANDB_API_KEY"):
        # A saved W&B login is also valid, e.g. a user-run wandb login.
        # Let the SDK resolve its own settings rather than reading credentials.
        print("W&B API key not found in environment; trying existing wandb login")
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    loaded_existing = False
    try:
        workspace = ws.Workspace.from_url(EXISTING_WORKSPACE_URL)
        if workspace.entity != ENTITY or workspace.project != PROJECT:
            raise RuntimeError("Existing workspace refers to a different project")
        loaded_existing = True
    except Exception as exc:
        print("EXISTING_WORKSPACE_LOAD_FAILED", type(exc).__name__, str(exc)[:180],
              "Creating a separate named view without modifying the personal view.",
              flush=True)
        workspace = prepare_workspace()
    if loaded_existing:
        existing = workspace.sections
        old_count = sum(len(s.panels) for s in existing)
        workspace.sections = construct_sections(ws, wr, existing=existing)
        # Maintain existing name and all old nonempty sections.
        workspace.runset_settings.pinned_columns = list(RUNS_COLUMNS)
        print("LOADED_EXISTING_WORKSPACE", workspace.name,
              "preserved_old_panel_count", old_count, flush=True)
    url = workspace.save()
    if not url:
        url = workspace.url
    print("WANDB_DASHBOARD_SAVED", str(url), "existing_view_updated", loaded_existing, flush=True)
    return {"workspace_url": str(url), "updated_existing_view": loaded_existing,
            "sections": [s.name for s in workspace.sections],
            "pins": list(RUNS_COLUMNS)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write to W&B online using existing, authorized SDK login")
    parser.add_argument("--manifest-out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = manifest()
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(result, indent=2))
    if args.apply:
        result["online"] = apply()
    else:
        model = prepare_workspace()
        assert len(model.sections) == 3
        assert len(model.sections[1].panels) == len(VALIDATION_PLOTS) + 1
        assert len(model.sections[2].panels) == len(FINAL_TEST_METRICS)
        print("OFFLINE_DASHBOARD_PLAN_OK", json.dumps({
            "sections": [s.name for s in model.sections],
            "eval_line_plots": len(model.sections[1].panels),
            "final_test_scalars": len(model.sections[2].panels),
            "runs_pinned_columns": len(model.runset_settings.pinned_columns),
            "no_network_writes": True,
        }))
    return result


if __name__ == "__main__":
    main()
