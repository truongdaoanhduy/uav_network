"""A saved online W&B project view using this environment's metric names."""
from pathlib import Path
import json

from .tracking import _authentication


PANELS = {
    "Training and outcomes": [
        ("Team return", "episode/return"),
        ("Found targets (%)", "outcome/search_rate_percent"),
        ("Delivered targets (%)", "outcome/delivery_rate_percent"),
        ("Mission success", "outcome/success"),
        ("Episodes completed", "train/episodes_completed"),
        ("Episode length", "episode/length"),
    ],
    "Live progress": [
        ("Search progress (%)", "progress/search_rate_percent"),
        ("Delivery progress (%)", "progress/delivery_rate_percent"),
        ("Optimizer updates", "train/updates"),
        ("Joint transitions / second", "train/joint_transitions_per_s"),
    ],
    "Safety and report errors": [
        ("Blocked motion rate", "safety/blocked_motion_rate"),
        ("Boundary clip rate", "safety/boundary_clip_rate"),
        ("Expired reports", "reports/expired"),
        ("Dropped reports", "reports/dropped"),
    ],
    "Energy": [
        ("UAV total energy (J)", "energy/total_uav_energy_j"),
        ("Propulsion (J)", "energy/propulsion_energy_j"),
        ("UAV radio (J)", "energy/uav_rf_energy_j"),
        ("Battery fraction", "progress/mean_battery_fraction"),
    ],
    "Network": [
        ("Packet delivery (%)", "network/pdr_percent"),
        ("Packet delay (ms)", "network/e2e_delay_ms"),
        ("Payload throughput (kbps)", "network/payload_throughput_kbps"),
        ("Report hops", "network/report_average_hops"),
        ("PHY success (%)", "network/phy_success_percent"),
        ("Collisions", "network/collisions"),
    ],
    "Learner": [("Actor loss", "learner/actor_loss"), ("Critic loss", "learner/critic_loss")],
}


def create_dashboard(config, run_ids=(), name="UAV Network 2D - MASAC and MATD3", workspace_url=None):
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
    workspace.sections = [ws.Section(name=title, is_open=True, panels=[
        wr.LinePlot(title=label, x="global_step", y=[metric]) for label, metric in metrics
    ]) for title, metrics in PANELS.items()]
    workspace.settings = ws.WorkspaceSettings(x_axis="global_step", max_runs=8)
    workspace.runset_settings = ws.RunsetSettings(
        filters=f"Config('tracking.group') == {config.tracking.group!r}",
        pinned_runs=list(run_ids),
        run_settings={ident: ws.RunSettings(disabled=False) for ident in run_ids},
        pinned_columns=["run:state", "config:runtime.algorithm", "config:runtime.seed",
                        "config:runtime.num_envs", "summary:global_step", "summary:train/updates",
                        "summary:outcome/search_rate_percent", "summary:outcome/delivery_rate_percent"],
    )
    workspace.save()
    result = {"workspace_url": workspace.url, "workspace_name": name,
              "entity": entity, "project": project, "pinned_runs": list(run_ids)}
    folder = Path("outputs/wandb")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "workspace.json").write_text(json.dumps(result, indent=2))
    return result
