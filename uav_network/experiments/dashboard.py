"""Personal Workspace by default, with an explicit saved-view URL override."""
from pathlib import Path
import base64
import json
import os
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from .tracking import CPU_EPISODE_METRICS, _authentication, default_workspace_url
from .dashboard_columns import VISIBLE_COLUMNS, PINNED_COLUMNS, save_with_visible_columns, reconcile_runset

from .run_journal import READING_GUIDE


PANELS = {
    "Mission quality - completed outcomes": [
        ("Delivery to GCS (%) - higher", "cohort/delivery_rate_percent_mean"),
        ("Mission success fraction - higher", "cohort/success_mean"),
        ("Completed-episode return - inspect with delivery", "cohort/return_mean"),
        ("Discovered targets (%) - higher", "cohort/search_rate_percent_mean"),
        ("Completed episodes in outcome window - sample size", "cohort/episodes"),
    ],
    "Training convergence - paper and benchmark views": [
        ("UAV-paper training return / reward", "episode/return"),
        ("UAV-paper search rate (%)", "outcome/search_rate_percent"),
        ("UAV-paper delivery rate (%)", "outcome/delivery_rate_percent"),
        ("UAV-paper mission success fraction", "outcome/success"),
        ("Sample-efficiency training return / reward", "episode/return"),
        ("Compute-efficiency training return / reward", "episode/return"),
        ("Episode return / reward (raw audit)", "episode/return"),
        ("Rolling return standard deviation", "rolling/return_std"),
    ],
    "Training progress and speed": [
        ("Completed training episodes", "train/episodes_completed"),
        ("Joint transitions / s - speed, not policy quality", "train/joint_transitions_per_s"),
        ("Sample uses / new transition - BASE segment vs PQL counters",
         ("train/sample_uses_per_new_transition", "pql/sample_uses_per_new_transition")),
    ],
    "Periodic fixed-policy evaluation - only when enabled": [
        ("Fixed-seed evaluation delivery (%) - higher", "evaluation/mean_delivery_rate_percent"),
        ("Fixed-seed evaluation success fraction - higher", "evaluation/success_rate"),
    ],
    "Final CPU evaluation - each episode": [
        ("CPU evaluation delivery (%) - higher", "evaluation/delivery_rate_percent"),
        ("CPU evaluation success (0/1)", "evaluation/success"),
    ],
    "Why delivery is low - inspect together": [
        ("Packet delivery fraction - distinct from target delivery", "cohort/network/packet_pdr_fraction"),
        ("Silent with eligible receiver (fraction)", "cohort/network/diagnostics/action_silent_given_eligible_fraction"),
        ("Blocked motion fraction", "safety/blocked_motion_rate"),
        ("UAV joules / GCS delivery - compare equal success", "cohort/energy/j_per_gcs_delivery"),
    ],
    "Learner stability - loss and gradients": [
        ("Critic loss - diagnostic only", "learner/critic_loss"),
        ("Actor loss - diagnostic only", "learner/actor_loss"),
        ("TD error absolute mean", "learner/td_error_abs_mean"),
        ("Q absolute maximum", "learner/q_abs_max"),
        ("Critic gradient norm", "learner/critic_grad_norm"),
    ],
    "Resources - runtime for system; training steps for disk": [
        ("GPU 0 / 1 utilization (%)", ("system/gpu.0.gpu", "system/gpu.1.gpu")),
        ("GPU 0 / 1 allocated memory (%)", ("system/gpu.0.memoryAllocated", "system/gpu.1.memoryAllocated")),
        ("CPU utilization (SDK %)", "system/cpu"),
        ("Host RAM utilization (%)", "system/memory_percent"),
        ("Free output disk bytes - training-step samples", "storage/disk_free_bytes"),
    ],
    "Recovery and upload cost": [
        ("Seconds since confirmed FULL upload - not captured-state age", "recovery/seconds_since_confirmed_upload"),
        ("Full snapshot + upload elapsed seconds", "recovery/elapsed_seconds"),
    ],
}


DIAGNOSTIC_PANELS = {
    "Numerical learning": [
        ("Actor loss - diagnostic only", "learner/actor_loss"),
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


def panel_axis(metric, title=""):
    metric = metric[0] if isinstance(metric, tuple) else metric
    if title.startswith("Sample-efficiency"):
        return "global_step"
    if title.startswith("Compute-efficiency"):
        return "_runtime"
    if metric == "episode/return" or metric.startswith("outcome/"):
        return "episode/index"
    if metric.startswith("system/"):
        return "_runtime"
    if metric in CPU_EPISODE_METRICS:
        return "evaluation/episodes_completed"
    return "evaluation/checkpoint_global_step" if metric.startswith(("evaluation/", "eval/")) else "global_step"


def panel_options(title, metric):
    """Display-only transforms; the underlying W&B history stays unchanged."""
    metric = metric[0] if isinstance(metric, tuple) else metric
    clean_curve = {"smoothing_type": "exponential", "smoothing_factor": 0.99,
                   "smoothing_show_original": False, "point_visualization_method": "sampling"}
    if title.startswith("UAV-paper"):
        return {**clean_curve, "title_x": "Episode (×5×10³)",
                "xaxis_expression": "${episode/index} / 5000"}
    if title.startswith("Sample-efficiency"):
        return {**clean_curve, "title_x": "Environment Steps (M)",
                "xaxis_expression": "${global_step} / 1000000"}
    if title.startswith("Compute-efficiency"):
        return {**clean_curve, "title_x": "Wall Time (min)",
                "xaxis_expression": "${_runtime} / 60"}
    if metric in {"learner/critic_loss", "learner/actor_loss", "learner/td_error_abs_mean"}:
        return {"smoothing_type": "average", "smoothing_factor": 0.8,
                "smoothing_show_original": True}
    return {"smoothing_type": "none"}


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
    workspace.sections = [ws.Section(name=title, is_open=not title.startswith(
        ("Periodic", "Numerical", "Recovery")), panels=[
        wr.LinePlot(title=label, x=panel_axis(metric, label),
                    y=list(metric) if isinstance(metric, tuple) else [metric],
                    max_runs_to_show=12, aggregate=False, **panel_options(label, metric))
        for label, metric in metrics
    ]) for title, metrics in panels.items()]
    workspace.sections.insert(0, ws.Section(name="How to read this evidence", is_open=True,
                              panels=[wr.MarkdownPanel(markdown=READING_GUIDE)]))
    workspace.settings = ws.WorkspaceSettings(x_axis="global_step", max_runs=1000)
    workspace.runset_settings = ws.RunsetSettings(
        filters="",
        pinned_runs=[],
        run_settings={},
        pinned_columns=list(PINNED_COLUMNS),
    )
    column_result = save_with_visible_columns(workspace, runs, _graphql)
    url = default_workspace_url(entity, project)
    if saved_view:
        url += f"?nw={selected}"
    result = {"workspace_url": url, "workspace_name": workspace.name,
              "workspace_type": "saved" if saved_view else "personal",
              "entity": entity, "project": project, "pinned_runs": [], "verified_run_ids": list(run_ids), "profile": profile,
              "panel_count": sum(len(v) for v in panels.values()) + 1, **column_result}
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
