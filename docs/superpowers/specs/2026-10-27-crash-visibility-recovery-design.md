# Crash visibility and bounded recovery checkpoints

User request: inspect missing/crashed G3 training, identify causes, fix code to reduce recurrence/loss, and show failures in W&B.

Evidence: fresh 2026-10-27T05:06Z W&B has271finished/81failed/30crashed, zero active; four MATD3 learners completed20k and CPU controllers complete, four MASAC learners/controllers lost heartbeat inside11s. All used immutable77a1d0f/off-off; newer acceleration source was not injected. Kaggle owner authentication now fails. Exact external cause remains unproven; no caught traceback. Latest MASAC HF manifests contain only policy; no exact-resume replay/optimizer. Completed MATD3 manifests show~15.86GB full checkpoints and~20.9GB working disk.

Goals:
- Preserve true server run states and completed results; surface crashed/failed before finished in default and explicitly selected workspaces, pin State/Algorithm/Mode and completion metadata.
- Publish periodic full resumable HF artifacts from consistent BASE/PQL boundaries even before the first episode completes, without marking a segment complete or moving its immutable source/config.
- Bound writes per session and disk/RAM peak. A staged checkpoint lives in a unique temporary directory, is uploaded+size/SHA verified and W&B-confirmed, then its own staging files are released; never delete a final/input/resume_latest checkpoint or any remote history.
- Keep the last confirmed resumable artifact pointer/alias across later light policy snapshots or failed backup attempts. Backup failures are visible and bounded rather than killing an otherwise healthy learner.
- Preserve scientific parameters, observation/generator protocol and measured acceleration policy.

Approaches: only workspace sorting fixes visibility but not lost progress; frequent full uploads without bounds exceed working-disk/storage constraints; selected approach reuses existing exact checkpoint and immutable HF manifest code with a small wall-clock coordinator. This is an implementation choice within the user's explicit fix request, not a new external control-plane deployment.

Settings: generic tracking.resumable_checkpoint_interval_seconds=0 disables opt-in, tracking.resumable_checkpoint_max_uploads=4. Canonical1KiB profile opts into3600seconds with max4attempts. For a finite soft budget, effective interval=max(requested interval,soft budget/max attempts), spacing attempts across the session;10h/4 yields9000seconds. Final upload remains independent. Actual bytes/time and attempts are visible. These are bounds, not a guarantee against external SIGKILL/account restriction.

Checkpoint semantics:
- Existing save callbacks gain an optional resume_path, leaving legacy final paths unchanged.
- Periodic PQL drains workers before snapshot; BASE snapshots only after reset at a full collection boundary, retaining pending-learning/RNG flags.
- OnlineTracker.checkpoint(...,resumable=True) includes resume.pt, bypasses light-policy throttling, applies a resumable alias, publishes pointer only after confirmation, and never sets final/completion.
- A failed attempt retains the previous confirmed remote pointer; only its owned temporary staging is removed. No automatic resubmission, cross-account switch or fake successful summary.
- Independent supervision/account credential repair is operationally separate; code cannot prevent platform/account termination or reconstruct replay already lost in these historical runs.

Validation: TDD visibility/order + publisher/alias/failed-upload invariants + wall-clock/cap/staging cleanup; real small BASE/PQL train snapshots and exact restore; full offline suite; a bounded legitimate2T4 diagnostic if available. Inspect actual artifacts/read back live workspace changes. No production relaunch or checkpoint cleanup.
