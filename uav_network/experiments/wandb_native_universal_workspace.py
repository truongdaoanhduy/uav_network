"""Apply one native W&B chart style to ALL real learner runs in the project.

Never upload PNGs/Vega tables or change training metrics. Run-selector shows
all runs with real episode progress, exactly one visible at a time. The native
Episode X uses episode/index (present on 57 of 74 existing Training runs);
older 17 runs with no episode/index cannot be drawn on this axis without
changing or reconstructing their training history, which this tool never does.
Smoothing: Training Running Average 21 / GPU Validation Average 5.
One line/metric, Show Original=False, per-run widths 1.6.
"""
from __future__ import annotations

import argparse
import os
import re

from . import wandb_native_paper_workspace as base


def _learner_ids(api, entity, project):
    records=list(api.runs(f"{entity}/{project}",order="-created_at",per_page=100))
    selected=[]
    for run in records:
        summary=run.summary  # Hydrate W&B's lazy Run objects.
        completed=summary.get("train/episodes_completed")
        if isinstance(completed,(int,float)) and 1<=completed<1e12:
            if re.fullmatch(r"[A-Za-z0-9]{8}",run.id):
                selected.append(run.id)
    if not selected:
        raise RuntimeError("No real learner runs found; leave W&B unchanged")
    return selected,len(records)


def build_universal_sections(ws,wr,run_ids):
    if not run_ids:
        raise ValueError("Must have at least one real Training run ID")
    sections=base.build_sections(ws,wr,run_ids[0])
    for panel in sections[0].panels:
        panel.x="episode/index"
        panel.title_x="Training Episode"
        panel.line_widths={f"{rid}:{panel.y[0]}":1.6 for rid in run_ids}
        panel.max_runs_to_show=1
    for panel in sections[1].panels:
        # GPU Evaluation uses its own actual checkpoint episodes. Missing
        # historical GPU metrics remain blank; never invent them.
        panel.x="gpu_eval_periodic/checkpoint_global_step"
        panel.title_x="Environment transitions"
        panel.line_widths={f"{rid}:{panel.y[0]}":1.6 for rid in run_ids}
        panel.max_runs_to_show=1
    return sections


def apply():
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Please grant WANDB_API_KEY to use the connected W&B account")
    api=wandb.Api(timeout=70)
    run_ids,all_run_count=_learner_ids(api,base.ENTITY,base.PROJECT)

    current=ws.Workspace.from_url(base.URL)
    if (current.entity,current.project)!=(base.ENTITY,base.PROJECT):
        raise RuntimeError("Refuse update of unrelated W&B workspace")

    current.sections=build_universal_sections(ws,wr,run_ids)
    current._auto_generate_panels=False
    current.runset_settings.query=""
    current.runset_settings.regex_query=False
    current.runset_settings.filters=expr.And(
        expr.Summary("train/episodes_completed")>=1)
    current.runset_settings.order=[
        expr.Ordering(expr.Metric("CreatedTimestamp"),ascending=False)]
    current.runset_settings.pinned_runs=[]
    # A maximum of one curve PER PANEL is not enough: W&B could otherwise
    # display Training from a new run and Eval from an older run. Make the
    # selected run consistent across every section, while keeping all
    # 74 eligible runs in the selector for manual switching.
    previously_visible=[
        rid for rid,settings in current.runset_settings.run_settings.items()
        if rid in run_ids and not settings.disabled
    ]
    selected=(previously_visible[0] if len(previously_visible)==1
              else 'f57a5e0f' if 'f57a5e0f' in run_ids else run_ids[0])
    current.runset_settings.run_settings={
        rid:ws.RunSettings(disabled=(rid!=selected)) for rid in run_ids
    }
    current.settings.max_runs=1
    current.settings.x_axis="episode/index"
    current.settings.tooltip_number_of_runs="single"
    # Preserve the user-curated 11-column locked Runs table when a future
    # Kaggle job refreshes the common W&B workspace.
    from .wandb_compact_runs_columns import save_compact_workspace
    save_compact_workspace(current)

    saved=ws.Workspace.from_url(base.URL)
    expected=[14,9,9]
    assert [len(s.panels) for s in saved.sections]==expected
    assert all(isinstance(p,wr.LinePlot) for s in saved.sections[:2] for p in s.panels)
    assert all(isinstance(p,wr.ScalarChart) for p in saved.sections[2].panels)
    assert saved.settings.max_runs==1
    assert "train/episodes_completed" in str(saved.runset_settings.filters)
    assert not (saved.runset_settings.pinned_runs)
    assert len(saved.sections[0].panels[0].line_widths)==len(run_ids)
    enabled=[rid for rid,setting in saved.runset_settings.run_settings.items()
             if not setting.disabled]
    assert enabled==[selected],f'Cross-run chart mixing risk: {enabled}'
    for section in saved.sections[:2]:
        for panel in section.panels:
            assert panel.smoothing_type=="average"
            assert panel.smoothing_show_original is False
            assert panel.smoothing_factor == (21 if section is saved.sections[0] else 5)
            assert panel.max_runs_to_show==1
            assert len(panel.y)==1
            assert len(panel.line_widths)==len(run_ids)
            assert set(panel.line_widths.values())=={1.6}
    print("VERIFIED_UNIVERSAL_NATIVE_WORKSPACE",
          "total_project_runs",all_run_count,
          "actual_training_runs",len(run_ids),
          "training_panels",expected[0],
          "gpu_evaluation_panels",expected[1],
          "final_test_panels",expected[2],
          "max_visible_runs",saved.settings.max_runs,
          "run_ids_kept_in_chart_styles",len(run_ids),
          "single_visible_run",selected,
          "x_training","episode/index",
          "line_width",1.6,"show_original",False)
    print("WAND_B_URL",base.URL)
    return saved


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args(argv)
    if args.apply:
        apply()
    else:
        print("Use --apply to update the common project W&B view for all real learner runs")


if __name__=="__main__":
    main()
