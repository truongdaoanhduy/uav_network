# W&B Runs table live Runtime / Episodes fix — November 22, 2026

Reference table: `summary:Runtime`, `summary:train/episodes_completed`, `summary:Episodes`, `summary:progress/episode_progress_percent_mean`, `summary:run/episode_budget_percent`.

## Verified original regression

On run `504d1b89`, before this change W&B already reported genuine
`global_step=5849088`, `train/updates=5572`, `train/episodes_completed=3051`,
`run/elapsed_seconds=2438.0063` and `run/episode_budget_percent=15.255`, but
both `summary:Runtime` and `summary:Episodes` were missing.
At an earlier stage `train/episodes_completed=0` was truthful: none of the
4096 parallel environments had completed their first episode. `progress/episode_progress_percent_mean`
was a *within-episode* progress gauge, not 20000-episode completion.

The historical run `18d756ef` wrote the custom string fields
`Runtime = 2h 27m 31s` and `Episodes = 20000 / 20000`; the v7 OnlineTracker
had not connected the existing LiveRuntimeHeartbeat module.

## Corrected OnlineTracker lifecycle

- On `wandb.init`, start `LiveRuntimeHeartbeat` with real `time.perf_counter()`
  and the monotonic Kaggle session origin `UAV_JOB_STARTED_AT` when supplied.
- Publish `Runtime` and `Episodes` in W&B **summary** immediately, even at
  0/20000, and again every 30 seconds; save a small local
  `runtime-heartbeat.json` for diagnosis. **No `run.log` calls**, no fake
  episode history and no W&B training step changes.
- Advance the episode counter only in `OnlineTracker.episode` and
  `OnlineTracker.progress` using the actual completed-episode integer, never
  by estimating from rollout/environment steps.
- Stop and join the heartbeat thread before closing the run, preserving its
  terminal `finished` or `failed` state.
- Existing `PostTrainingRuntimeHeartbeat` handles GPU validation/final Test
  after the training W&B SDK instance closes.

`tests/test_online_tracker_live_table_20261122.py` tests initial values,
actual episode advancement, a local evidence file, and proper thread shutdown
on success or failure. Regression suite:

```bash
python -m pytest -q \
  tests/test_online_tracker_live_table_20261122.py \
  tests/test_tracking.py tests/test_run_journal.py tests/test_runtime.py \
  tests/test_18d756ef_fullflow_integrity_20261122.py \
  tests/test_matd3_reference_style_kaggle_ui.py \
  tests/test_training_only_pipeline.py tests/test_kaggle_embedded_source.py
# 58 passed
```

The original frozen MATD3 and MASAC source hashes were unchanged.

## Existing live Kaggle runs

Changing GitHub source, notebook files or the local working tree **cannot
inject a newly added heartbeat thread into a Python process that Kaggle is
already executing**. Run `504d1b89` was observed as `running` during this
review. Its existing truthful W&B fields may already be displayed using
**Columns**: `run/elapsed_seconds`, `train/episodes_completed`,
`global_step`, `train/updates` and `run/episode_budget_percent`. Training
and evaluation fields remain blank until real episodes/Validation/Test finish.

A read-then-write through the external W&B public `run.summary.update` API was
tried once with the existing summary backed up locally. Although the update
returned without an error, **fresh W&B reads showed Runtime and Episodes
still absent**. Do **not** claim that existing in-flight run was fixed online
or overwrite the whole summary repeatedly while a Kaggle learner is logging.
The fix is effective for notebooks **launched anew after its code is built**;
there is no need to restart a healthy older training run merely for two
formatting columns.
