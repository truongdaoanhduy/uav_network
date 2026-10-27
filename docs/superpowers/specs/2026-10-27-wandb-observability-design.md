# W&B observability: compact evidence, useful logs

User explicitly requests meaningful W&B online plots/logs and Created/Runtime/run-table fields, with recommendations applied without metric clutter. Baseline a046a3b.

## Current gaps

Personal workspace has about 50 plots and >30 pinned/visible columns; Created is hidden. Primary saved view has 18 older plots, different sorting and some wrong evaluation axes. Existing diagnostics and raw telemetry are extensive, but only cohorts/resolved config are registered during normal tracking. Loss alone cannot grade a policy; periodic evaluation is disabled in the canonical profile.

## Selected approach

Reuse existing metrics/diagnostics and W&B native system/console capture. No new grading system, training-loop changes, eval schedule changes or dependency. Main and primary saved workspace get 26 curated plots plus one explanatory Markdown panel. Plot at most 12 individual runs (no cross-algorithm aggregation); table retains all authentic states and newest-first order.

Core groups: four completed-outcome quality plots; three training progress/throughput/sample-use plots; two periodic-checkpoint eval plots (collapsed) and two final CPU episode plots; four delivery/safety/energy bottlenecks; five numerical plots (collapsed, losses labelled diagnostic rather than good/bad); four native CPU/RAM/GPU0+1/VRAM plots on runtime axis; two recovery plots (collapsed). Empty metric evidence remains empty.

Runs table: Name, State, Created, Runtime, Stage, Algorithm, Mode, Account, Seed, Global step, completed training episodes, cohort delivery/success, eval delivery/success, true budget completion, stop reason, fatal error type. Pin only first four; explicitly hide other columns including previously pinned ones.

## Logging contract

A small stdlib RunJournal writes credential-free scalar context to training-events.jsonl (append), training-status.json (atomic latest snapshot), and metrics-guide.md. Records include UTC, event, run/source/algorithm/seed identity, actual progress/runtime/phase, selected learner and mission metrics. Record at existing progress cadence and checkpoint/recovery/evaluation/failure/end transitions, not each optimizer step; never serialize replay/tensors or full config into logs. Nonfinite numbers are explicit strings, missing values remain null. File/console/optional upload errors must not introduce a learner failure.

W&B native Logs stream structured events live. Latest status is policy=live, guide=now, growing events/diagnostics=end and explicitly flushed on caught failure/close; cohorts retain current live policy. Hard kills may interrupt final upload; prior console/status/checkpoint evidence remains, no complete final traceback guarantee. Save lightweight diagnostic files only; preserve existing failed-batch/state handling and checkpoint semantics.

OnlineTracker caches selected learner evidence, exposes real episode-budget progress separately from in-episode percent, reports seconds since confirmed full upload (not captured-state age) separately from light policy uploads, and publishes failure_type for the table. PeriodicRecovery emits best-effort snapshot_started and upload_started before those potentially slow operations, preserving existing error containment.

## Constraints and verification

No production launch/restart/cancel, history/state rewriting or fabricated historical metrics. Update live views only after local tests, retain a pre-change workspace backup. New logging applies to future source-pinned jobs; existing historic metrics can be replotted now. Preserve scientific configs, actor locality, precision policy and evaluation protocol.

Tests must exercise real SDK serialization, removed stale columns/pins, correct training/eval/system axes, bounded plot selection, event context/nonfinite/secret-safe fields and failure containment, tracker progress/confirmed checkpoint age/failure uploads, and recovery stage markers. Full offline CPU suite plus independent review, live workspace readback, source/secret checks and documentation before completion.

## Builder-check resolutions

No user input needed. Aggregate eval columns use eval/mean_delivery_rate_percent and eval/success_rate, never final_* training aggregates or the last CPU episode. The 12-run cap is per LinePlot.max_runs_to_show, while workspace maxRuns=1000 preserves the table; startup reconciliation must preserve panel configuration. Recovery timing is explicitly seconds since confirmed full upload, not age of captured training state. CPU visualization callbacks route scalar records via tracker.diagnostic so the journal sees actual CPU progress and no training-budget percentage is invented there. Cohort n/partial/source and distinct BASE/PQL sample-use populations are explained in the guide. Console, append, status writes and optional file uploads fail independently. Native resource series are left empty where unavailable.
