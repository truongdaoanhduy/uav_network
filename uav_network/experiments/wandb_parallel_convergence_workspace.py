"""New-run W&B dashboard for vectorized UAV RL, without editing legacy views.

No network calls on import or --dry-run. Create a separate view ONLY when
explicitly invoked with --apply after a real W&B run has been created.
"""
from __future__ import annotations

import argparse
import re

# Label and X clock are deliberately independent of W&B global defaults.
SECTIONS = (
    ("TRAINING | SAME 1-vector-step reward & live mission progress", "global_step", (
        ("Mean team reward | 4096 worlds", "train/step_reward_mean"),
        ("Reward population SD | 4096 worlds", "train/step_reward_std"),
        ("Target discovery (%) | same 4096 worlds", "train_progress/discovery_rate_percent"),
        ("GCS delivery (%) | same 4096 worlds", "train_progress/gcs_delivery_rate_percent"),
        ("Map coverage (%) | same 4096 worlds", "train_progress/map_coverage_percent"),
        ("Mean battery fraction | same 4096 worlds", "train_progress/mean_battery_fraction"),
        ("Mean active UAVs | same 4096 worlds", "train_progress/mean_active_uavs"),
    )),
    ("TRAINING | SAME 1-vector-step optimizer updates (starts after warmup)", "global_step", (
        ("Critic loss mean | all true updates", "train_loss/critic_loss"),
        ("Critic loss SD | all true updates", "train_loss/critic_loss_std"),
        ("Actor loss mean | true Actor updates only", "train_loss/actor_loss"),
        ("Actor loss SD | true Actor updates only", "train_loss/actor_loss_std"),
        ("Absolute TD error mean | all critic updates", "train_loss/td_error_abs_mean"),
        ("Critic gradient norm | all critic updates", "train_loss/critic_grad_norm"),
        ("Actor gradient norm | true actor updates", "train_loss/actor_grad_norm"),
        ("MASAC temperature objective | true updates", "train_loss/alpha_loss"),
        ("Critic updates counted in vector step", "train_loss/critic_updates"),
        ("Actor updates counted in vector step", "train_loss/actor_updates"),
    )),
    ("TRAINING | True completed episodes at each available vector step", "global_step", (
        ("Completed episodes count in this vector step", "train_episode_step/episode_count"),
        ("Episodic return | completed this vector step", "train_episode_step/episode_return_mean"),
        ("Final discovery (%) | completed this step", "train_episode_step/discovery_rate_percent_mean"),
        ("Final GCS delivery (%) | completed this step", "train_episode_step/gcs_delivery_rate_percent_mean"),
        ("Discovery AUC | completed this vector step", "train_episode_step/discovery_auc_mean"),
        ("Delivery AUC | completed this vector step", "train_episode_step/delivery_auc_mean"),
        ("Final map coverage (%) | completed this step", "train_episode_step/map_coverage_percent_mean"),
        ("Mission success (%) | completed this step", "train_episode_step/mission_success_rate_percent_mean"),
        ("Energy per delivery | valid completed episodes", "train_episode_step/energy_per_delivered_target_j_mean"),
        ("Return | 64-completion stability reference", "train/episode_return_mean"),
        ("Discovery (%) | 64-completion reference", "train_cohort/discovery_rate_percent_mean"),
        ("GCS Delivery (%) | 64-completion reference", "train_cohort/gcs_delivery_rate_percent_mean"),
        ("Map Coverage (%) | 64-completion reference", "train_cohort/map_coverage_percent_mean"),
    )),
    ("OPTIMIZER | Sparse advanced diagnostics (real 65536-step cadence)", "global_step", (
        ("Q-value absolute maximum | diagnostic sample", "learner/q_abs_max"),
        ("TD-error diagnostic sample", "learner/td_error_abs_mean"),
        ("SAC continuous entropy temperature", "learner/alpha_continuous"),
        ("SAC discrete entropy temperature", "learner/alpha_discrete"),
    )),
    ("EVAL | Real fixed-seed validation at checkpoint global_step", "global_step", (
        ("Validation mean episodic return", "gpu_eval_periodic/mean_return"),
        ("Validation episodic return SD", "gpu_eval_periodic/return_std"),
        ("Validation target discovery (%)", "gpu_eval_periodic/discovery_rate_percent"),
        ("Validation GCS delivery (%)", "gpu_eval_periodic/gcs_delivery_rate_percent"),
        ("Validation discovery AUC", "gpu_eval_periodic/discovery_auc"),
        ("Validation delivery AUC", "gpu_eval_periodic/delivery_auc"),
        ("Validation map coverage (%)", "gpu_eval_periodic/map_coverage_percent"),
    )),
)

FINAL = (
    ("Final Test discovery (%)", "gpu_final/discovery_rate_percent"),
    ("Final Test GCS delivery (%)", "gpu_final/gcs_delivery_rate_percent"),
    ("Final Test mean return", "gpu_final/mean_return"),
    ("Final Test return SD", "gpu_final/mean_return_std"),
    ("Final Test discovery AUC", "gpu_final/discovery_auc"),
    ("Final Test delivery AUC", "gpu_final/delivery_auc"),
)


def build_sections(ws, wr, run_id):
    sections = []
    for section_name, x, items in SECTIONS:
        panels = []
        for title, metric in items:
            panels.append(wr.LinePlot(
                title=title, x=x, y=[metric],
                title_x="Total environment transitions (all worlds)",
                title_y=title, smoothing_type="none", smoothing_factor=0,
                smoothing_show_original=False, groupby_rangefunc="none",
                max_runs_to_show=1, line_widths={f"{run_id}:{metric}": 1.6},
            ))
        sections.append(ws.Section(name=section_name, panels=panels, is_open=True))
    # Test has only ONE held-out best-checkpoint observation: one point on
    # the same environment transition X, NOT a fabricated time-series.
    sections.append(ws.Section(name="TEST | One held-out best-checkpoint point",
        panels=[wr.LinePlot(title=title + " (single best-checkpoint result)",
            x="global_step", y=[key],
            title_x="Total environment transitions (all worlds)",
            title_y=title, smoothing_type="none", smoothing_factor=0,
            smoothing_show_original=False, groupby_rangefunc="none",
            max_runs_to_show=1) for title,key in FINAL], is_open=True))
    return sections


def create_separate_view(run_id, entity, project):
    if not re.fullmatch(r"[a-zA-Z0-9]{8}", str(run_id)):
        raise ValueError("Supply a real eight-character W&B run id")
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    run = wandb.Api(timeout=45).run(f"{entity}/{project}/{run_id}")
    if run.id != run_id:
        raise RuntimeError("W&B resolved an unexpected run ID")
    new_view = ws.Workspace(
        entity=entity, project=project,
        name=f"Parallel RL convergence - {run_id}",
        sections=build_sections(ws, wr, run_id),
        settings=ws.WorkspaceSettings(x_axis="global_step", max_runs=1,
            smoothing_type="none", smoothing_weight=0),
        runset_settings=ws.RunsetSettings(
            filters=expr.And(expr.Metric("ID") == run_id),
            run_settings={run_id:ws.RunSettings(disabled=False)}),
        auto_generate_panels=False,
    )
    new_view.save()
    return new_view.url


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--entity', default='truongdaoanhduy-uit')
    parser.add_argument('--project', default='uav_search_target')
    parser.add_argument('--apply',action='store_true',help='Create NEW W&B view; never update original workspace')
    args=parser.parse_args(argv)
    if not args.apply:
        print('DRY_RUN: separate view, 5 sections, ALL time-series X = global_step')
        print('Would create for',args.entity,args.project,args.run_id)
    else:
        print('CREATED_NEW_VIEW',create_separate_view(args.run_id,args.entity,args.project))

if __name__=='__main__':
    main()
