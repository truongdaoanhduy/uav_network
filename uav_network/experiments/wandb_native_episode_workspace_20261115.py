"""Native interactive W&B panels: numeric series, never image uploads.

The workspace has one metric per W&B LinePlot and is designed for a single-run
view (one algorithm). Native gaussian smoothing and the original faint trace
are controlled by W&B so zoom/tooltip/export remain interactive.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ENTITY = "truongdaoanhduy-uit"
PROJECT = "uav_search_target"
URL = "https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=m89iort84op"
NATIVE_SECTION = "TRAINING | Native W&B interactive (Episode, Gaussian smoothing)"
EVAL_SECTION = "EVAL | GPU checkpoint validation (native)"
TEST_SECTION = "TEST | Held-out final policy (native scalars)"

# All names are existing NUMERIC W&B metrics, not images.
TRAINING = (
    ("Reward • Training", "episode/return"),
    ("Search / discovery (%)", "training/discovery_rate_percent"),
    ("GCS delivery (%)", "training/gcs_delivery_rate_percent"),
    ("Discovery AUC", "training/discovery_auc"),
    ("Delivery AUC", "training/delivery_auc"),
    ("Map coverage (%)", "training/map_coverage_percent"),
    ("Coverage AUC", "training/coverage_auc"),
    ("Energy per delivered target (J)", "training/energy_per_delivered_target_j"),
    ("Capped mean GCS delivery time (s)", "training/gcs_delivery_time_capped_mean_s"),
    ("Mission success (%)", "training/mission_success_rate_percent"),
    ("Confirmed targets", "training/targets_found"),
    ("Reported targets at GCS", "training/targets_delivered"),
    ("Actor loss", "learner/actor_loss"),
    ("Critic loss", "learner/critic_loss"),
)

EVAL = (
    ("Validation discovery (%)", "gpu_eval/discovery_rate_percent"),
    ("Validation GCS delivery (%)", "gpu_eval/gcs_delivery_rate_percent"),
    ("Validation discovery AUC", "gpu_eval/discovery_auc"),
    ("Validation delivery AUC", "gpu_eval/delivery_auc"),
    ("Validation map coverage (%)", "gpu_eval/map_coverage_percent"),
    ("Validation coverage AUC", "gpu_eval/coverage_auc"),
    ("Validation energy / delivered target (J)", "gpu_eval/energy_per_delivered_target_j"),
    ("Validation capped delivery time (s)", "gpu_eval/gcs_delivery_time_capped_mean_s"),
    ("Validation mission success (%)", "gpu_eval/mission_success_rate_percent"),
)
TEST = tuple((title.replace("Validation", "Final Test"), metric.replace("gpu_eval/", "gpu_final/"))
             for title, metric in EVAL)

# This view must be reusable for any future algorithm. It does NOT store
# names of algorithms or combine separate algorithms into the same image.
def native_training_section(ws, wr):
    return ws.Section(
        name=NATIVE_SECTION, is_open=True,
        panels=[
            wr.LinePlot(
                title=title,
                x="episode/index", y=[metric],
                title_x="Training Episode", title_y=title,
                smoothing_type="gaussian",
                smoothing_factor=0.5,
                smoothing_show_original=True,
                line_widths={metric: 2.8},
                max_runs_to_show=1,
                legend_position="south",
                # No false across-seed SD: only one selected W&B training run.
                groupby_rangefunc="none",
            )
            for title, metric in TRAINING
        ]
    )


def build_sections(ws, wr, existing=None):
    native = native_training_section(ws, wr)
    eval_sec = ws.Section(
        name=EVAL_SECTION, is_open=False,
        panels=[
            wr.LinePlot(
                title=title, x="gpu_eval/checkpoint_global_step", y=[key],
                title_x="Environment interactions",
                smoothing_type="gaussian", smoothing_factor=0.25,
                smoothing_show_original=True, max_runs_to_show=1,
                line_widths={key: 2.5},
            )
            for title, key in EVAL
        ]
    )
    test_sec = ws.Section(
        name=TEST_SECTION, is_open=False,
        panels=[wr.ScalarChart(title=title, metric=key)
                for title, key in TEST],
    )
    preserve = []
    for old in existing or []:
        label = (old.name or "").lower()
        # Remove only previous assistant-generated image/duplicated line-panel
        # sections; leave unrelated user-authored sections untouched.
        generated = (
            label.startswith("paper style |")
            or label.startswith("paper | openai")
            or label.startswith("openai spinning up |")
            or label.startswith("training | mission improvement")
            or label.startswith("eval | deterministic checkpoint")
            or label.startswith("test | held-out final")
            or label.startswith(NATIVE_SECTION.lower())
            or label.startswith(EVAL_SECTION.lower())
            or label.startswith(TEST_SECTION.lower())
        )
        if generated:
            continue
        old.is_open = False
        preserve.append(old)
    return [native, eval_sec, test_sec, *preserve]


def prepare_workspace(ws, wr, existing=None):
    import wandb_workspaces.workspaces as real_ws
    if existing is not None:
        existing.sections = build_sections(ws, wr, existing.sections)
        existing._auto_generate_panels = False
        return existing
    return real_ws.Workspace(
        entity=ENTITY, project=PROJECT, name="UAV Native Episode Charts",
        sections=build_sections(ws, wr),
        auto_generate_panels=False,
    )


def apply(*, url=URL):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    target = ws.Workspace.from_url(url)
    if target.entity != ENTITY or target.project != PROJECT:
        raise RuntimeError("Refusing to edit an unrelated W&B project")
    model = prepare_workspace(ws, wr, target)
    try:
        model.save()
    except Exception as exc:
        print("W&B returned after save:", type(exc).__name__)
    fresh = ws.Workspace.from_url(url)
    first = fresh.sections[0]
    assert first.name == NATIVE_SECTION
    assert len(first.panels) == len(TRAINING)
    assert all(isinstance(p, wr.LinePlot)
               and getattr(p.x, "name", p.x) == "episode/index" for p in first.panels)
    assert not any(isinstance(p, wr.MediaBrowser)
                   for section in fresh.sections for p in section.panels)
    print("VERIFIED_NATIVE_WANDB_VIEW", url,
          "training_panels", len(first.panels),
          "eval_panels", len(fresh.sections[1].panels),
          "test_panels", len(fresh.sections[2].panels),
          "algorithm_specific_panels", 0,
          "image_panels", 0)
    return fresh


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args(argv)
    if args.apply:
        apply(url=args.url)
    else:
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        sections = build_sections(ws, wr)
        print("DRY_RUN", [(s.name, len(s.panels)) for s in sections])


if __name__ == "__main__":
    main()
