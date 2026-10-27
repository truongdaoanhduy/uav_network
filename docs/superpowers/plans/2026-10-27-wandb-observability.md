# W&B Observability Implementation Plan

> **For agentic workers:** Execute sequentially with superpowers:executing-plans and TDD; review before publishing live views.

**Goal:** Make online W&B evidence readable and failures diagnosable without metric clutter.
**Architecture:** Curate existing workspace metrics/columns, add a small stdlib journal and wire existing tracker/recovery transitions; preserve all scientific configuration.
**Tech Stack:** Python, pytest, W&B/Workspaces SDK, stdlib JSON/files.
**Spec:** docs/superpowers/specs/2026-10-27-wandb-observability-design.md

## Global constraints

No training launch, history/state rewrite, extra dependencies or automatic good/bad grades from loss. No new scalar flood or live syncing large raw episode/replay files. Keep periodic-eval schedule unchanged and expose when absent.

## Review focus

True observation axes and units; latest-state vs completed-outcome vs fixed-seed evaluation semantics; hiding/pinning only curated columns without hiding failed runs; fail-open logging/redaction; confirmed full-state age vs light checkpoint age; live/finish file durability and hard-kill limits.

### Task 1: Compact workspace and run table

Files: dashboard.py, dashboard_columns.py, tests/test_dashboard.py and new targeted assertions.
- [x] RED: Created/Runtime visible, four pinned columns, stale columns/pins removed; real model has curated axes and <=12 individual runs/plot plus guide panel.
- [x] GREEN: 26 plots plus guide, 18 columns, preserve identity/all states/newest ordering.

### Task 2: Structured useful logs

Files: run_journal.py (record scalar event/context/status/guide with contained I/O failures), tracking.py, recovery_checkpoints.py, cloud/visualize.py; journal/tracking/recovery tests.
- [x] RED: actual progress/start/close/failure context, nonfinite values, I/O containment, seconds since confirmed full upload, safe file policies, serialization/upload transition markers.
- [x] GREEN: integrate journal and selected evidence without altering scientific work or retry/checkpoint contracts.

### Task 3: Verify and publish

- [ ] Focused and full offline CPU tests, source/secret/static checks, independent review.
- [ ] Backup/read live Personal and primary saved views, apply curated core profiles, read back identities/table/axes/panel selection and real metric availability.
- [ ] Update human guide/evidence/CURRENT_TASK with exact checks and limits, commit and push main + working branch. No production run is needed merely to test UI/logging.
