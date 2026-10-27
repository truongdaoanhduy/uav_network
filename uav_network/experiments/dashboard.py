"""Personal Workspace by default, with an explicit saved-view URL override."""
from pathlib import Path
import base64
import json
import os
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .tracking import CPU_EPISODE_METRICS, _authentication, default_workspace_url
from .dashboard_columns import VISIBLE_COLUMNS, save_with_visible_columns, reconcile_runset


PANELS = {
    "Live mission progress": [
        ("Episode progress (%)", "progress/episode_progress_percent_mean"),
        ("Episode step mean", "progress/episode_step_mean"),
        ("Episode step min", "progress/episode_step_min"),
        ("Episode step max", "progress/episode_step_max"),
        ("Live search (%)", "progress/search_rate_percent"),
        ("Live delivery (%)", "progress/delivery_rate_percent"),
        ("Live battery fraction", "progress/mean_battery_fraction"),
        ("Active UAVs", "progress/mean_active_uavs"),
    ],
    "Training throughput": [
        ("Joint transitions / s", "train/joint_transitions_per_s"),
        ("Replay size", "train/replay_size"),
        ("Learner updates", "train/updates"),
        ("Sample uses / new transition", "train/sample_uses_per_new_transition"),
        ("Elapsed hours", "run/elapsed_hours"),
    ],
    "Learning quality": [
        ("Actor loss", "learner/actor_loss"),
        ("Critic loss", "learner/critic_loss"),
        ("TD error |mean|", "learner/td_error_abs_mean"),
        ("Q |max|", "learner/q_abs_max"),
        ("Critic gradient", "learner/critic_grad_norm"),
        ("Actor gradient", "learner/actor_grad_norm"),
    ],
    "PQL execution": [
        ("PQL transitions / s", "pql/transitions_per_s"),
        ("PQL critic updates / s", "pql/critic_updates_per_s"),
        ("PQL actor updates / s", "pql/actor_updates_per_s"),
        ("Critic / collector ratio", "pql/critic_collector_ratio"),
        ("PQL sample uses / new transition", "pql/sample_uses_per_new_transition"),
        ("Queued collections", "pql/queued_collections"),
        ("Collector wait (s)", "pql/collector_wait_seconds"),
    ],
    "Completed outcomes per transition window": [
        ("Completed-outcome return", "cohort/return_mean"),
        ("Completed-outcome discovery (%)", "cohort/search_rate_percent_mean"),
        ("Completed-outcome delivery (%)", "cohort/delivery_rate_percent_mean"),
        ("Completed-outcome success fraction", "cohort/success_mean"),
    ],
    "Report communication": [
        ("Pooled packet delivery fraction", "cohort/network/packet_pdr_fraction"),
        ("Payload throughput (kbps)", "network/payload_throughput_kbps"),
        ("PHY success (%)", "network/phy_success_percent"),
        ("Delivered-report hops", "cohort/reports/delivery_hops_mean"),
        ("Delivered-report age (s)", "cohort/reports/delivered_report_age_mean_s"),
        ("Collisions", "network/collisions"),
    ],
    "Battery and energy": [
        ("Total UAV energy (J)", "energy/total_uav_energy_j"),
        ("Propulsion energy (J)", "energy/propulsion_energy_j"),
        ("Communication energy (J)", "energy/uav_rf_energy_j"),
        ("Battery fraction", "energy/mean_battery_fraction"),
    ],
    "Safety and report loss": [
        ("Blocked motion rate", "safety/blocked_motion_rate"),
        ("Boundary clip rate", "safety/boundary_clip_rate"),
        ("Expired reports", "reports/expired"),
        ("Dropped reports", "reports/dropped"),
    ],
    "Fixed-seed evaluation": [
        ("Validation return (fixed checkpoint)", "evaluation/mean_return"),
        ("Validation search (%)", "evaluation/mean_search_rate_percent"),
        ("Validation delivery (%)", "evaluation/mean_delivery_rate_percent"),
        ("Validation success", "evaluation/success_rate"),
    ],
    "Diagnostic alerts": [
        ("Warning count", "diagnostics/warning_count"),
        ("Fatal error count", "diagnostics/fatal_error_count"),
    ],
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


def panel_axis(metric):
    if metric in CPU_EPISODE_METRICS:
        return "evaluation/episodes_completed"
    return "evaluation/checkpoint_global_step" if metric.startswith(("evaluation/", "eval/")) else "global_step"


def run_filter(group, run_ids=()):
    expression = f"Config('tracking.group') == {group!r}"
    if run_ids:
        selected = " or ".join(f"Metric('ID') == {ident!r}" for ident in run_ids)
        expression += f" and ({selected})"
    return expression


def _graphql(query, variables=None):
    # Workspace CRUD does not need a learner/SDK service process.
    base = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    authorization = base64.b64encode(f"api:{os.environ['WANDB_API_KEY']}".encode()).decode()
    request = Request(base + "/graphql", method="POST",
                      data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                      headers={"Content-Type": "application/json", "Authorization": "Basic " + authorization})
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"W&B workspace API returned HTTP {error.code}") from None
    if result.get("errors"):
        raise RuntimeError("W&B workspace API: " + "; ".join(e["message"] for e in result["errors"]))
    return result["data"]


def _project_runs(entity, project):
    runs, cursor = [], None
    while True:
        response = _graphql("""
            query WorkspaceRunColumns($entityName: String, $projectName: String, $after: String) {
                project(entityName: $entityName, name: $projectName) {
                    runs(first: 100, after: $after) {
                        edges { node { name config summaryMetrics } }
                        pageInfo { hasNextPage endCursor }
                    }
                }
            }
        """, {"entityName": entity, "projectName": project, "after": cursor})
        if response.get("project") is None:
            raise ValueError("Configured W&B project not found")
        page = response["project"]["runs"]
        for edge in page["edges"]:
            node = edge["node"]
            raw_config = json.loads(node["config"] or "{}")
            config = {k: v["value"] if isinstance(v, dict) and "value" in v else v
                      for k, v in raw_config.items()}
            runs.append((node["name"], config, json.loads(node["summaryMetrics"] or "{}")))
        if not page["pageInfo"]["hasNextPage"]:
            return runs
        cursor = page["pageInfo"]["endCursor"]


def create_dashboard(config, run_ids=(), name="UAV Network 2D - MASAC and MATD3", workspace_url=None, profile="core"):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    if profile not in ("core", "diagnostics"):
        raise ValueError("Dashboard profile must be core or diagnostics")
    entity, project = _authentication(config)
    username = _graphql("query WorkspaceOwner { viewer { username } }")["viewer"]["username"]
    selected = ""
    if workspace_url:
        parsed = urlparse(workspace_url)
        parts = parsed.path.strip("/").split("/")
        if parts[0] == "wandb":
            parts = parts[1:]
        if parts[:2] != [entity, project]:
            raise ValueError("Workspace must belong to the configured W&B entity/project")
        if parts[2:] not in ([], ["workspace"]):
            raise ValueError("Workspace URL must point to the project or its Workspace page")
        selected = parse_qs(parsed.query).get("nw", [""])[0]
        if selected.startswith("nwuser") and selected != f"nwuser{username}":
            raise ValueError("Personal Workspace must belong to the authenticated W&B user")
    saved_view = bool(selected and not selected.startswith("nwuser"))
    view_name = f"nw-{selected}-v" if saved_view else f"nw-nwuser{username}-w"
    response = _graphql("""
        query WorkspaceTarget($entityName: String, $projectName: String, $viewName: String) {
            project(entityName: $entityName, name: $projectName) {
                allViews(viewType: "project-view", viewName: $viewName) {
                    edges { node { id displayName } }
                }
            }
        }
    """, {"entityName": entity, "projectName": project, "viewName": view_name})
    if response.get("project") is None:
        raise ValueError("Configured W&B project not found")
    edges = response["project"]["allViews"]["edges"]
    if saved_view and not edges:
        raise ValueError("Requested saved view not found")
    existing = edges[0]["node"] if edges else {}
    workspace = ws.Workspace(entity=entity, project=project,
                             name=existing.get("displayName") or (name if saved_view else f"{username}'s workspace"))
    # ponytail: the SDK's default save creates a random saved-view identity.
    # Reuse the app's personal identity and the existing model/upsert boundary.
    workspace._internal_name = view_name
    workspace._internal_id = existing.get("id", "")
    runs = _project_runs(entity, project)
    missing = set(run_ids) - {ident for ident, _, _ in runs}
    if missing:
        raise ValueError(f"Requested W&B runs not found: {sorted(missing)}")
    panels = PANELS if profile == "core" else DIAGNOSTIC_PANELS
    workspace.sections = [ws.Section(name=title, is_open=title not in
                                     ("Fixed-seed evaluation", "Diagnostic alerts"), panels=[
        wr.LinePlot(title=label, x=panel_axis(metric), y=[metric]) for label, metric in metrics
    ]) for title, metrics in panels.items()]
    workspace.settings = ws.WorkspaceSettings(x_axis="global_step", max_runs=1000)
    workspace.runset_settings = ws.RunsetSettings(
        filters="",
        pinned_runs=[],
        run_settings={},
        pinned_columns=list(VISIBLE_COLUMNS),
    )
    column_result = save_with_visible_columns(workspace, runs, _graphql)
    url = default_workspace_url(entity, project)
    if saved_view:
        url += f"?nw={selected}"
    result = {"workspace_url": url, "workspace_name": workspace.name,
              "workspace_type": "saved" if saved_view else "personal",
              "entity": entity, "project": project, "pinned_runs": [], "verified_run_ids": list(run_ids), "profile": profile,
              "panel_count": sum(len(v) for v in panels.values()), **column_result}
    folder = Path("outputs/wandb")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "workspace.json").write_text(json.dumps(result, indent=2))
    return result


def ensure_project_workspace(config):
    """Keep the default Personal Workspace open to present and future runs."""
    entity, project = _authentication(config)
    username = _graphql("query WorkspaceOwner { viewer { username } }")["viewer"]["username"]
    variables = {"entityName": entity, "projectName": project,
                 "viewName": f"nw-nwuser{username}-w"}
    response = _graphql("""query MainWorkspace($entityName:String!, $projectName:String!, $viewName:String!) {
        project(entityName:$entityName, name:$projectName) {
            allViews(viewType:"project-view", viewName:$viewName) {
                edges { node { id name displayName spec } }
            }
        }
    }""", variables)
    if response.get("project") is None:
        raise ValueError("Configured main W&B project not found")
    edges = response["project"]["allViews"]["edges"]
    if not edges:
        return create_dashboard(config)["workspace_url"]
    view = edges[0]["node"]
    spec = json.loads(view["spec"]) if isinstance(view["spec"], str) else view["spec"]
    original = json.dumps(spec, sort_keys=True)
    section = spec["section"]
    section.setdefault("settings", {})["maxRuns"] = 1000
    for runset in section.get("runSets", []):
        reconcile_runset(runset)
    if json.dumps(spec, sort_keys=True) != original:
        inputs = {"id": view["id"], "entityName": entity, "projectName": project,
                  "name": view["name"], "displayName": view["displayName"], "type": "project-view",
                  "spec": json.dumps(spec), "createdUsing": "WANDB_SDK"}
        _graphql("mutation MainWorkspace($input:UpsertViewInput!) { upsertView(input:$input) { view { id name } } }",
                 {"input": inputs})
    return default_workspace_url(entity, project)
