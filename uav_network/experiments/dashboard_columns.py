"""Curated W&B run-table columns for live and completed UAV comparisons."""
VISIBLE_COLUMNS = [
    "run:displayName", "run:state", "run:createdAt", "run:duration",
    "config:kaggle_account",
    # Live training progress: available before the first episode terminates.
    "summary:global_step", "summary:train/episodes_completed", "summary:train/updates",
    "summary:train/replay_size", "summary:train/joint_transitions_per_s",
    "summary:train/sample_uses_per_new_transition",
    "summary:progress/search_rate_percent", "summary:progress/delivery_rate_percent",
    "summary:progress/mean_battery_fraction", "summary:progress/mean_active_uavs",
    # Learner health.
    "summary:learner/actor_loss", "summary:learner/critic_loss",
    "summary:learner/td_error_abs_mean", "summary:learner/q_abs_max",
    # PQL execution health; blank for BASE runs by design.
    "summary:pql/transitions_per_s", "summary:pql/critic_updates_per_s",
    "summary:pql/actor_updates_per_s", "summary:pql/critic_collector_ratio",
    "summary:pql/sample_uses_per_new_transition", "summary:pql/queued_collections",
    # Latest completed episode.
    "summary:episode/return", "summary:outcome/search_rate_percent",
    "summary:outcome/delivery_rate_percent", "summary:outcome/success",
    "summary:network/pdr_percent", "summary:network/payload_throughput_kbps",
    "summary:network/report_average_hops", "summary:energy/total_uav_energy_j",
    "summary:energy/mean_battery_fraction", "summary:reports/expired", "summary:reports/dropped",
    # Windowed/fixed-seed outcome summaries.
    "summary:cohort/return_mean", "summary:cohort/episodes", "summary:cohort/partial",
    "summary:cohort/source_end_step", "summary:cohort/search_rate_percent_mean",
    "summary:cohort/delivery_rate_percent_mean", "summary:cohort/success_mean",
    "summary:evaluation/mean_delivery_rate_percent",
    "summary:evaluation/checkpoint_global_step", "summary:evaluation/episodes",
]


def flattened_keys(values, prefix=""):
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        yield key
        if isinstance(value, dict):
            yield from flattened_keys(value, key)


def save_with_visible_columns(workspace, runs, graphql):
    from wandb_workspaces.expr import FE_METRIC_NAME_MAP

    model = workspace._to_model()
    feed = model.spec.section.run_sets[0].run_feed
    known = set(feed.column_visible) | set(feed.column_pinned) | set(feed.column_order)
    known.update(f"run:{name}" for name in FE_METRIC_NAME_MAP.values())
    for _, config, summary in runs:
        known.update(f"config:{key}" for key in flattened_keys(config))
        known.update(f"summary:{key}" for key in flattened_keys(summary))
    known.update(VISIBLE_COLUMNS)
    # ponytail: SDK save marks only pinned columns true and leaves others visible.
    # Use its current model/upsert boundary until public hidden-columns support exists.
    feed.column_visible = {name: name in VISIBLE_COLUMNS for name in sorted(known)}
    feed.column_pinned = {name: name in VISIBLE_COLUMNS for name in sorted(known)}
    feed.column_order = list(VISIBLE_COLUMNS)
    inputs = {"entityName": workspace.entity, "projectName": workspace.project,
              "name": model.name, "displayName": model.display_name, "type": "project-view",
              "spec": model.spec.model_dump_json(by_alias=True, exclude_none=True),
              "createdUsing": "WANDB_SDK"}
    if model.id:
        inputs["id"] = model.id
    result = graphql("""
        mutation SaveWorkspace($input: UpsertViewInput!) {
            upsertView(input: $input) { view { id name } }
        }
    """, {"input": inputs})
    workspace._internal_id = result["upsertView"]["view"]["id"]
    return {"visible_columns": VISIBLE_COLUMNS, "hidden_column_count": len(known - set(VISIBLE_COLUMNS))}
