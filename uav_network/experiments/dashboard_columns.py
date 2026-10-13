"""Explicitly hide unneeded W&B columns, including old project configuration."""
VISIBLE_COLUMNS = [
    "run:displayName", "run:state", "run:createdAt", "run:duration",
    "config:kaggle_account", "summary:train/episodes_completed",
    "summary:episode/return", "summary:outcome/search_rate_percent",
    "summary:outcome/delivery_rate_percent",
]


def flattened_keys(values, prefix=""):
    for name, value in values.items():
        key = f"{prefix}.{name}" if prefix else name
        yield key
        if isinstance(value, dict):
            yield from flattened_keys(value, key)


def save_with_visible_columns(workspace, api):
    from wandb_workspaces.workspaces import internal
    from wandb_workspaces.expr import FE_METRIC_NAME_MAP

    model = workspace._to_model()
    feed = model.spec.section.run_sets[0].run_feed
    known = set(feed.column_visible) | set(feed.column_pinned) | set(feed.column_order)
    known.update(f"run:{name}" for name in FE_METRIC_NAME_MAP.values())
    for run in api.runs(f"{workspace.entity}/{workspace.project}", per_page=100):
        known.update(f"config:{key}" for key in flattened_keys(run.config))
        known.update(f"summary:{key}" for key in flattened_keys(dict(run.summary)))
    known.update(VISIBLE_COLUMNS)
    # ponytail: SDK save marks only pinned columns true and leaves others visible.
    # Use its current model/upsert boundary until public hidden-columns support exists.
    feed.column_visible = {name: name in VISIBLE_COLUMNS for name in sorted(known)}
    feed.column_pinned = {name: name in VISIBLE_COLUMNS for name in sorted(known)}
    feed.column_order = list(VISIBLE_COLUMNS)
    internal.upsert_view2(model)
    return {"visible_columns": VISIBLE_COLUMNS, "hidden_column_count": len(known - set(VISIBLE_COLUMNS))}
