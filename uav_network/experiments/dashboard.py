"""A saved online W&B project view using this environment's metric names."""
from pathlib import Path
import json

from .tracking import _authentication


PANELS = {
    "Mission outcomes": [
        ("Team return", "episode/return"),
        ("Found targets (%)", "outcome/search_rate_percent"),
        ("Delivered targets (%)", "outcome/delivery_rate_percent"),
        ("Mission success", "outcome/success"),
    ],
    "Report communication": [
        ("Packet delivery (%)", "network/pdr_percent"),
        ("Report delay (s)", "network/report_e2e_delay_s"),
        ("Payload throughput (kbps)", "network/payload_throughput_kbps"),
        ("Report hops", "network/report_average_hops"),
    ],
    "Battery": [
        ("Total UAV energy (J)", "energy/total_uav_energy_j"),
        ("Battery fraction", "energy/mean_battery_fraction"),
    ],
    "Safety and report loss": [
        ("Blocked motion rate", "safety/blocked_motion_rate"),
        ("Expired reports", "reports/expired"),
        ("Dropped reports", "reports/dropped"),
    ],
    "Learner": [("Actor loss", "learner/actor_loss"), ("Critic loss", "learner/critic_loss")],
    "Fixed-seed evaluation": [
        ("Evaluation return", "eval/mean_return"),
        ("Evaluation delivery (%)", "eval/mean_delivery_rate_percent"),
    ],
    "Diagnostic alerts": [("Warning count", "diagnostics/warning_count")],
}

DIAGNOSTIC_PANELS = {
    "Numerical learning": [
        ("TD error", "learner/td_error_abs_mean"), ("Q magnitude", "learner/q_abs_max"),
        ("Critic gradient", "learner/critic_grad_norm"), ("Actor gradient", "learner/actor_grad_norm"),
        ("Continuous entropy", "learner/continuous_entropy"), ("Next-hop entropy", "learner/discrete_entropy"),
        ("Continuous alpha", "learner/alpha_continuous"), ("Discrete alpha", "learner/alpha_discrete"),
    ],
    "Routing and reward": [
        ("Silent fraction", "action/silent_fraction"), ("Peer fraction", "action/peer_fraction"),
        ("GCS fraction", "action/gcs_fraction"), ("Discovery reward", "reward/discovery"),
        ("Delivery reward", "reward/delivery"), ("Obstacle penalty", "reward/blocked_motion"),
        ("Expiry penalty", "reward/expired_report"), ("Drop penalty", "reward/dropped_report"),
    ],
    "Warnings and failures": [
        ("Search-delivery gap", "diagnostics/delivery_gap_warning"),
        ("Return drop", "diagnostics/return_drop_warning"),
        ("Reward accounting", "diagnostics/reward_accounting_warning"),
        ("Fatal error", "diagnostics/fatal_error_count"),
    ],
}


def run_filter(group, run_ids=()):
    expression = f"Config('tracking.group') == {group!r}"
    if run_ids:
        selected = " or ".join(f"Metric('ID') == {ident!r}" for ident in run_ids)
        expression += f" and ({selected})"
    return expression


def create_dashboard(config, run_ids=(), name="UAV Network 2D - MASAC and MATD3", workspace_url=None, profile="core"):
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    entity, project = _authentication(config)
    api = wandb.Api()
    for ident in run_ids:
        api.run(f"{entity}/{project}/{ident}")  # Only pin verified existing project runs.
    workspace = ws.Workspace.from_url(workspace_url) if workspace_url else ws.Workspace(entity=entity, project=project, name=name)
    if workspace.entity != entity or workspace.project != project:
        raise ValueError("Workspace must belong to the configured W&B entity/project")
    if profile not in ("core", "diagnostics"):
        raise ValueError("Dashboard profile must be core or diagnostics")
    panels = PANELS if profile == "core" else DIAGNOSTIC_PANELS
    workspace.sections = [ws.Section(name=title, is_open=title not in
                                     ("Fixed-seed evaluation", "Diagnostic alerts"), panels=[
        wr.LinePlot(title=label, x="global_step", y=[metric]) for label, metric in metrics
    ]) for title, metrics in panels.items()]
    workspace.settings = ws.WorkspaceSettings(x_axis="global_step", max_runs=len(run_ids) if run_ids else 2)
    workspace.runset_settings = ws.RunsetSettings(
        filters=run_filter(config.tracking.group, run_ids),
        pinned_runs=list(run_ids),
        run_settings={ident: ws.RunSettings(disabled=False) for ident in run_ids},
        pinned_columns=["run:state", "run:createdAt", "run:duration",
                        "config:runtime.algorithm", "config:kaggle_account",
                        "summary:train/episodes_completed",
                        "summary:outcome/search_rate_percent", "summary:outcome/delivery_rate_percent"],
    )
    workspace.save()
    result = {"workspace_url": workspace.url, "workspace_name": name,
              "entity": entity, "project": project, "pinned_runs": list(run_ids), "profile": profile,
              "panel_count": sum(len(v) for v in panels.values())}
    folder = Path("outputs/wandb")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "workspace.json").write_text(json.dumps(result, indent=2))
    return result
