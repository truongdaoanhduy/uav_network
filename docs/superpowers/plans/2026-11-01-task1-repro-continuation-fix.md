# Task 1 Reproducibility and Continuation Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make deterministic PQL actually reproducible without collapsing the split critic/policy architecture, and restore scheduled Kaggle continuation for time-limited runs.

**Architecture:** Keep `pql.mode=async` as the split-learner architecture. Add a separate concurrent-scheduling decision: deterministic runs execute collector/critic/policy work in a fixed inline order, while nondeterministic runs keep current worker threads/queues. Separately, make the default-branch supervisor workflow expose the same account slots as the feature branch so scheduled cron can reconcile continuations.

**Tech Stack:** Python 3.11, PyTorch, pytest, GitHub Actions, Kaggle/W&B supervisor.

**Spec:** Root-cause evidence from MASAC PQL runs `335074a2` and `eb654c77` plus GitHub scheduled run `41293075575`.

## Global Constraints

- Do not change reward, scenario, CBF, MASAC/MATD3 hyperparameters, seed, or replay policy.
- Preserve split PQL critic/policy learners for `pql.mode=async`.
- `runtime.deterministic=true` must not depend on thread scheduling order.
- `runtime.deterministic=false` must preserve current concurrent PQL behavior.
- Do not modify `docs/superpowers/CURRENT_TASK.md`.

## Review Focus

- Deterministic async-config PQL retains separate critic/policy models.
- Deterministic async-config PQL creates no worker threads and reports deterministic scheduling.
- Nondeterministic async-config PQL still creates workers and reports nondeterministic scheduling.
- Two deterministic async-config runs repeat hashes on the same platform/source.
- Default-branch cron workflow exposes every enabled account secret used by `KAGGLE_ENABLED_ACCOUNTS`.

### Task 1A: Deterministic PQL scheduling

**Files:**
- Modify: `uav_network/experiments/pql_system.py`
- Modify: `uav_network/experiments/pql_training.py`
- Test: `tests/test_pql_system.py`
- Test: `tests/test_pql_training.py`

- [x] Write failing scheduling and repeatability tests.
- [x] Run tests and confirm RED.
- [x] Separate split-learner architecture from concurrent scheduling.
- [x] Run targeted tests and confirm GREEN.

### Task 1B: Scheduled continuation

**Files:**
- Default branch: `.github/workflows/uav_external_supervisor.yml`
- Test: `tests/test_github_supervisor_workflow.py`

- [x] Verify current feature workflow account wiring passes tests.
- [x] Copy only the corrected workflow to default branch `main`.
- [x] Dispatch/dry-run supervisor and verify successful reconciliation.
- [x] Confirm MASAC 16,384 checkpoints become eligible for continuation rather than remaining stranded.
