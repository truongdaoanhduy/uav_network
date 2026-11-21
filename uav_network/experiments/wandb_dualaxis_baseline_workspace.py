"""Create a NEW per-run W&B native dashboard with both Episode and Env-Step axes.

Does not mutate any existing personal workspace or backup. Panels are standard
W&B interactive LinePlots/ScalarCharts and reference logged scalar metrics;
there are no synthetic series, image uploads, virtual episode indices or
mistaken optimizer-loss-to-episode mappings.

Usage:
  python -m uav_network.experiments.wandb_dualaxis_baseline_workspace --run-id 1234abcd
  python -m ... --run-id 1234abcd --apply --wait-seconds 600 --output /kaggle/working/view.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

ENTITY = 'truongdaoanhduy-uit'
PROJECT = 'uav_search_target'

# Paper-like original Training episode outcomes, NO optimizer loss sampled on
# episode/index when the optimizer record has no episode number.
EPISODE = (
    ('Reward / team return', 'episode/return'),
    ('Target discovery (%)', 'training/discovery_rate_percent'),
    ('GCS delivery (%)', 'training/gcs_delivery_rate_percent'),
    ('Discovery AUC', 'training/discovery_auc'),
    ('Delivery AUC', 'training/delivery_auc'),
    ('Map coverage (%)', 'training/map_coverage_percent'),
    ('Coverage AUC', 'training/coverage_auc'),
    ('Energy per delivered target (J)', 'training/energy_per_delivered_target_j'),
    ('Capped GCS delivery time (s)', 'training/gcs_delivery_time_capped_mean_s'),
    ('Mission success (%)', 'training/mission_success_rate_percent'),
    ('Confirmed targets', 'training/targets_found'),
    ('Reports to GCS', 'training/targets_delivered'),
    ('Episode length (steps)', 'episode/length'),
    ('Unsafe action fraction', 'safety/raw_unsafe_action_rate'),
)
VECTOR = (
    ('Instant team reward | all 4096 worlds', 'train/step_reward_mean'),
    ('Reward population SD | all worlds', 'train/step_reward_std'),
    ('Instant target discovery (%) | episode resets visible', 'train_progress/discovery_rate_percent'),
    ('Instant GCS delivery (%) | episode resets visible', 'train_progress/gcs_delivery_rate_percent'),
    ('Instant map coverage (%) | episode resets visible', 'train_progress/map_coverage_percent'),
    ('Mean battery fraction', 'train_progress/mean_battery_fraction'),
    ('Mean active UAVs', 'train_progress/mean_active_uavs'),
)
LEARNER = (
    ('Actor loss | actual delayed optimizer updates', 'train_loss/actor_loss'),
    ('Critic loss | actual optimizer updates', 'train_loss/critic_loss'),
    ('Actor gradient norm', 'train_loss/actor_grad_norm'),
    ('Critic gradient norm', 'train_loss/critic_grad_norm'),
    ('Absolute TD error', 'train_loss/td_error_abs_mean'),
    ('Actor updates in vector step', 'train_loss/actor_updates'),
    ('Critic updates in vector step', 'train_loss/critic_updates'),
)
# Same nine mission metrics in the user's restored W&B screenshot, plus episodic return.
from .wandb_native_paper_workspace import EVAL as PAPER_EVAL, TEST as PAPER_TEST
EVAL = (('Validation mean episodic return', 'gpu_eval_periodic/mean_return'),
        ('Validation episodic return SD (100 seeds)', 'gpu_eval_periodic/return_std'),
        *PAPER_EVAL)
TEST = PAPER_TEST


def axis(panel):
    return getattr(panel.x, 'name', panel.x)


def validate_run_id(run_id):
    if not re.fullmatch('[A-Za-z0-9]{8}', str(run_id)):
        raise ValueError('Run ID must be exactly eight letters/numbers')
    return str(run_id)


def _line(wr, run_id, title, metric, *, x, title_x, smoothing):
    return wr.LinePlot(
        title=title, x=x, y=[metric], title_x=title_x, title_y=title,
        smoothing_type='average', smoothing_factor=smoothing,
        smoothing_show_original=False, groupby_rangefunc='none',
        max_runs_to_show=1,
        line_widths={f'{run_id}:{metric}':1.6},
    )


def build_sections(ws, wr, run_id):
    run_id=validate_run_id(run_id)
    def section(name, entries, x, label, smoothing):
        return ws.Section(name=name, is_open=True,
            panels=[_line(wr,run_id,title,key,x=x,title_x=label,smoothing=smoothing)
                    for title,key in entries])
    return [
        section('01 | TRAINING — Episode (MA21, raw hidden)',
                EPISODE,'episode/index','Training Episode',21),
        section('02 | TRAINING — Global Step / 1 vector step',
                VECTOR,'global_step','Total Environment Steps (4096 transitions/vector step)',21),
        section('03 | TRAINING — Optimizer / Global Step',
                LEARNER,'global_step','Total Environment Steps',21),
        section('04 | EVAL — Fixed 100 seeds / checkpoint (deterministic)',
                EVAL,'gpu_eval_periodic/checkpoint_global_step','Total Environment Steps',5),
        ws.Section(name='05 | TEST — Best checkpoint / 4000 scenarios',is_open=True,
            panels=[wr.ScalarChart(title=title,metric=key) for title,key in TEST]),
    ]


def new_workspace(run_id, *, entity=ENTITY, project=PROJECT, variant='MATD3'):
    run_id=validate_run_id(run_id)
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    # Create NEW saved view. In particular, don't call Workspace.from_url()
    # to edit the user's existing personal Episode dashboard or backup.
    return ws.Workspace(
        entity=entity, project=project,
        name=f'{variant} Episode and Global Step - {run_id}',
        sections=build_sections(ws,wr,run_id),
        settings=ws.WorkspaceSettings(x_axis='global_step',max_runs=1,
            tooltip_number_of_runs='single'),
        runset_settings=ws.RunsetSettings(
            filters=expr.And(expr.Metric('ID') == run_id),
            run_settings={run_id:ws.RunSettings(disabled=False)}),
        auto_generate_panels=False,
    )


def create_view(run_id, *, entity=ENTITY, project=PROJECT, variant='MATD3'):
    view = new_workspace(run_id, entity=entity, project=project, variant=variant)
    view.save()
    return view.url


def apply_wait(run_id, *, output=None, wait_seconds=600, variant='MATD3'):
    validate_run_id(run_id)
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('WANDB_API_KEY is required, do not embed it in source')
    destination=Path(output) if output else None
    if destination is not None and destination.exists():
        old=json.loads(destination.read_text())
        if old.get('run_id')==run_id and old.get('url'):
            print('EXISTING_DUAL_AXIS_VIEW',old['url'],flush=True)
            return old['url']
    # The W&B run is created by the Training subprocess, not in the Kaggle
    # notebook bootstrap. Wait for it to exist before creating the saved view.
    import wandb
    deadline=time.monotonic()+max(0,int(wait_seconds))
    while True:
        try:
            found=wandb.Api(timeout=25).run(f'{ENTITY}/{PROJECT}/{run_id}')
            if found.id!=run_id:
                raise RuntimeError('Unexpected W&B run id')
            break
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Run {run_id} not visible in W&B before deadline: {type(exc).__name__}') from exc
            time.sleep(min(15,max(0,deadline-time.monotonic())))
    url=create_view(run_id,variant=variant)
    if destination is not None:
        destination.parent.mkdir(exist_ok=True,parents=True)
        destination.write_text(json.dumps({'run_id':run_id,'url':url,'type':'native_dual_axis'},indent=2))
    print('CREATED_MATD3_EPISODE_GLOBAL_STEP_VIEW',url,flush=True)
    return url


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--variant',default='MATD3')
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--output',default=None)
    parser.add_argument('--wait-seconds',type=int,default=600)
    args=parser.parse_args(argv)
    if args.apply:
        apply_wait(args.run_id,output=args.output,wait_seconds=args.wait_seconds,variant=args.variant)
    else:
        model=new_workspace(args.run_id,variant=args.variant)
        print('DUAL_AXES_DRY_RUN',model.name,[(s.name,len(s.panels)) for s in model.sections])


if __name__=='__main__':
    main()
