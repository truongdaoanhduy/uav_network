# W&B visibility fix — MASAC/MATD3 BASE/PQL

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Root cause

The 26-Oct production runs still exist in `truongdaoanhduy-uit/uav_search_target`; they were not deleted. The stale saved view `UAV Network 2D - MASAC and MATD3` still filtered `config.tracking.group == "uav-network-2d"` and limited `maxRuns` to 2, while the production G3 runs use the newer `UAV-G3-...-20261026` groups. Opening that saved view therefore hid the production runs.

The run table also pinned `summary:training_mode` but did not pin the already-existing historical field `config.runtime.algorithm`, making MASAC/MATD3 identification unnecessarily dependent on parsing the display name.

## Fix

- Added `config:runtime.algorithm` to `VISIBLE_COLUMNS` immediately before `summary:training_mode`; existing historical runs already contain this field, so no W&B run backfill was needed.
- TDD RED: the focused column test failed at index 4 because the algorithm column was missing.
- GREEN: dashboard + diagnostics focused suite passed 10/10.
- Updated the live stale saved view in place without deleting it or changing run data: filters=[], search="", enabled=true, maxRuns=1000, Algorithm and Mode columns visible+pinned.
- Fresh GraphQL read-back verified those exact live settings.
- Personal Workspace was already unfiltered with maxRuns=1000 and already showed Algorithm+Mode; no destructive reset was required.

Source commit: `b3837fa` (`fix(wandb): show algorithm in runs table`).
