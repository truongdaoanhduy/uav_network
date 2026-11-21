"""Restore ordinary W&B plots, with optional paper-like native smoothing.

No Vega / CustomChart / PNG / W&B Tables. One real run, one numeric metric per
native W&B LinePlot. W&B itself does tooltip, Episode x-axis, smoothing,
faint original trace, zoom, and export.

Use: python -m uav_network.experiments.wandb_native_paper_workspace --apply --run-id f57a5e0f
"""
from __future__ import annotations

import argparse
import os
import re

ENTITY="truongdaoanhduy-uit"
PROJECT="uav_search_target"
URL="https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=m89iort84op"

TRAIN_SECTION="TRAINING | W&B Line Plots • paper-style smoothing"
EVAL_SECTION="EVAL | Periodic validation by environment transitions"
TEST_SECTION="TEST | Final held-out metrics"

# Grounded original scalars, no recomputation and no invented band.
TRAIN=(
    ("Reward / team return","episode/return"),
    ("Target discovery (%)","training/discovery_rate_percent"),
    ("GCS delivery (%)","training/gcs_delivery_rate_percent"),
    ("Discovery AUC","training/discovery_auc"),
    ("Delivery AUC","training/delivery_auc"),
    ("Map coverage (%)","training/map_coverage_percent"),
    ("Coverage AUC","training/coverage_auc"),
    ("Energy per delivered target (J)","training/energy_per_delivered_target_j"),
    ("Capped GCS delivery time (s)","training/gcs_delivery_time_capped_mean_s"),
    ("Mission success (%)","training/mission_success_rate_percent"),
    ("Confirmed targets","training/targets_found"),
    ("Reported targets to GCS","training/targets_delivered"),
    ("Actor loss","learner/actor_loss"),
    ("Critic loss","learner/critic_loss"),
)

EVAL=(
    ("Validation target discovery (%)","gpu_eval_periodic/discovery_rate_percent"),
    ("Validation GCS delivery (%)","gpu_eval_periodic/gcs_delivery_rate_percent"),
    ("Validation discovery AUC","gpu_eval_periodic/discovery_auc"),
    ("Validation delivery AUC","gpu_eval_periodic/delivery_auc"),
    ("Validation map coverage (%)","gpu_eval_periodic/map_coverage_percent"),
    ("Validation coverage AUC","gpu_eval_periodic/coverage_auc"),
    ("Validation energy per delivered target (J)","gpu_eval_periodic/energy_per_delivered_target_j"),
    ("Validation capped delivery time (s)","gpu_eval_periodic/gcs_delivery_time_capped_mean_s"),
    ("Validation mission success (%)","gpu_eval_periodic/mission_success_rate_percent"),
)

TEST=tuple((title.replace("Validation ","Final Test "), key.replace("gpu_eval_periodic/","gpu_final/"))
           for title,key in EVAL)


def make_line(wr, title, key, run_id, phase):
    return wr.LinePlot(
        title=title,
        x="episode/index" if phase=="train" else "gpu_eval_periodic/checkpoint_global_step",
        y=[key],
        title_x="Training Episode" if phase=="train" else "Environment transitions",
        title_y=title,
        smoothing_type="average",
        smoothing_factor=21 if phase=="train" else 5,
        smoothing_show_original=False,
        groupby_rangefunc="none",
        max_runs_to_show=1,
        line_widths={f"{run_id}:{key}":1.6},
    )


def build_sections(ws,wr,run_id):
    return [
        ws.Section(name=TRAIN_SECTION,is_open=True,
                   panels=[make_line(wr,title,key,run_id,"train") for title,key in TRAIN]),
        ws.Section(name=EVAL_SECTION,is_open=True,
                   panels=[make_line(wr,title,key,run_id,"eval") for title,key in EVAL]),
        ws.Section(name=TEST_SECTION,is_open=True,
                   panels=[wr.ScalarChart(title=title,metric=key) for title,key in TEST]),
    ]


def apply(*,run_id):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    if not re.fullmatch(r"[A-Za-z0-9]{8}",str(run_id)):
        raise ValueError("A real 8-character W&B run ID is required")
    view=ws.Workspace.from_url(URL)
    if (view.entity,view.project)!=(ENTITY,PROJECT):
        raise RuntimeError("Unexpected W&B project; refusing update")
    view.sections=build_sections(ws,wr,run_id)
    view.runset_settings.pinned_runs=[]
    view.runset_settings.filters=expr.And(expr.Metric("ID")==run_id)
    view.settings.max_runs=1
    view.settings.x_axis="episode/index"
    view._auto_generate_panels=False
    view.save()

    saved=ws.Workspace.from_url(URL)
    assert [(s.name,len(s.panels)) for s in saved.sections]==[
        (TRAIN_SECTION,len(TRAIN)),(EVAL_SECTION,len(EVAL)),(TEST_SECTION,len(TEST))]
    assert saved.settings.max_runs==1
    assert run_id in str(saved.runset_settings.filters)
    assert all(isinstance(p,wr.LinePlot) for sec in saved.sections[:2] for p in sec.panels)
    assert all(isinstance(p,wr.ScalarChart) for p in saved.sections[2].panels)
    assert all(p.smoothing_type=="average" and p.smoothing_show_original is False
               for sec in saved.sections[:2] for p in sec.panels)
    assert all(p.x.name=="episode/index" if hasattr(p.x,"name") else p.x=="episode/index"
               for p in saved.sections[0].panels)
    assert all(p.x.name=="gpu_eval_periodic/checkpoint_global_step" if hasattr(p.x,"name")
               else p.x=="gpu_eval_periodic/checkpoint_global_step"
               for p in saved.sections[1].panels)
    print("VERIFIED_NATIVE_WANDB",URL,"run",run_id,
          "sections",[(s.name,len(s.panels)) for s in saved.sections],
          "native_lineplots",len(TRAIN)+len(EVAL),"custom_charts",0,"png_panels",0)
    return saved


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--apply",action="store_true")
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args(argv)
    if args.apply:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Grant WANDB_API_KEY as a secret; no key on command line")
        apply(run_id=args.run_id)
    else:
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        print([(s.name,len(s.panels)) for s in build_sections(ws,wr,args.run_id)])


if __name__=="__main__":
    main()
