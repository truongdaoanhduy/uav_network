"""Compact W&B Runs: latest TRAIN episode and mean held-out TEST metrics.

Training discovery, GCS delivery and coverage describe the latest logged
training episode for both running and finished runs. The three Final Test
metrics are means across independent held-out episodes. Training-history means
are separate records and are deliberately not mixed into this live overview.
"""
from __future__ import annotations

# Full-pipeline live view. The W&B system run:state becomes Finished when
# the learner closes its SDK session, *before* a separate Validation/Test
# subprocess finishes. The user prefers the original W&B run:state column.
# Both live MASAC learners write train/episodes_completed and Episodes,
# while the legacy episodes_completed key is populated only on completion.
MAIN_COLUMNS = (
    "run:displayName",
    "run:state",
    "run:createdAt",
    "summary:Runtime",
    "summary:train/episodes_completed",
    "summary:Episodes",
    "summary:progress/episode_progress_percent_mean",
    "config:runtime.value.algorithm",
    "config:runtime.value.num_envs",
    "summary:training/discovery_rate_percent",
    "summary:training/gcs_delivery_rate_percent",
    "summary:training/map_coverage_percent",
    "summary:gpu_final/discovery_rate_percent",
    "summary:gpu_final/gcs_delivery_rate_percent",
    "summary:gpu_final/map_coverage_percent",
    "config:kaggle_account.value",
    "config:kaggle_url.value",

)
PINNED_COLUMNS = ("run:displayName", "run:state")

# A held-out-only leaderboard filtered by gpu_final/episodes >= 1, rather than
# placing nine almost-empty Test columns on the entire 74-run overview.
FINAL_COLUMNS = (
    "run:displayName",
    "config:runtime.value.algorithm",
    "config:runtime.value.num_envs",
    "config:kaggle_account.value",
    "config:kaggle_url.value",
    "summary:gpu_final/episodes",
    "summary:gpu_final/discovery_rate_percent",
    "summary:gpu_final/gcs_delivery_rate_percent",
    "summary:gpu_final/map_coverage_percent",
    "summary:gpu_final/delivery_auc",
    "summary:gpu_final/energy_per_delivered_target_j",
    "summary:gpu_final/gcs_delivery_time_capped_mean_s",
)
FINAL_PINNED_COLUMNS = ("run:displayName",)


def set_columns_on_model(view_model, columns=MAIN_COLUMNS, pinned=PINNED_COLUMNS):
    columns = tuple(columns)
    pinned = tuple(pinned)
    if not columns or any(name not in columns for name in pinned):
        raise ValueError("Invalid Runs table column selection")
    runsets = view_model.spec.section.run_sets
    if not runsets:
        raise ValueError("Workspace has no runset")
    for runset in runsets:
        feed = runset.run_feed
        feed.column_visible = {column: True for column in columns}
        feed.column_order = list(columns)
        feed.column_pinned = {column: True for column in pinned}
        feed.lock_columns = True
    return view_model


def save_compact_workspace(view, columns=MAIN_COLUMNS, pinned=PINNED_COLUMNS):
    from wandb_workspaces.workspaces import internal
    model = set_columns_on_model(view._to_model(), columns, pinned)
    result = internal.upsert_view2(model)
    stored = result["upsertView"]["view"]
    view._internal_name = stored["name"]
    view._internal_id = stored["id"]
    return view


def check_online_columns(view, columns=MAIN_COLUMNS, pinned=PINNED_COLUMNS):
    from wandb_workspaces.workspaces import internal
    model = internal.View.from_name(view.entity, view.project, view._internal_name)
    feed = model.spec.section.run_sets[0].run_feed
    assert feed.lock_columns is True
    assert feed.column_visible == {column: True for column in columns}
    assert feed.column_order == list(columns)
    assert feed.column_pinned == {column: True for column in pinned}
    assert view.runset_settings.pinned_columns == list(pinned)
    return True
