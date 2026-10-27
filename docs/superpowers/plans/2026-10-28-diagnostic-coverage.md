# Diagnostic Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement sequentially; independent review before integration.

**Goal:** Make failure and weak-policy evidence useful and resilient without a universal root-cause/quality claim.
**Architecture:** Reuse existing journals, diagnostics, system telemetry and evaluation protocol. Stdlib bounded evidence helpers and a process-owned traceback watchdog; preserve all scientific behavior.
**Tech Stack:** Python stdlib, existing Torch/W&B/Workspaces SDK, pytest.
**Spec:** docs/superpowers/specs/2026-10-28-diagnostic-coverage-design.md

## Global Constraints

No production training, state/history rewrite, scientific config or checkpoint format change. Algorithm remains hidden. Personal17 columns,4 pins.27core plots+guide. No new dependency or external monitoring service. Missing evidence stays unknown.

## Review Focus

Nested secret/nonfinite/object evidence; disk/console failure independence; watchdog global timer/FD ownership and cleanup; evaluation count/protocol/source semantics; exact exit signals without inferred OOM; existing science/lifecycle contracts.

### Task1: Safe failure and diagnostic evidence

Files: new evidence.py, run_journal.py, diagnostics.py, async_evaluation.py, training.py and pql_training.py fallback guards; tests/test_diagnostics.py, test_run_journal.py, test_async_evaluation.py.
- [x] RED: closed console/disk-full diagnostics do not abort, nested NaN/objects/secrets persist safely, primary cause metadata/resources remain usable.
- [x] GREEN: shared safe serializer/writer, fail-open optional channels, conservative category/resources.
- [x] Run focused existing+new tests and commit (48passed,2subtests).

### Task2: Forward progress, root context and publication boundaries

Files: new progress_watchdog.py, tracking.py, training.py, pql_training.py, cloud/remote.py, cloud/visualize.py; focused tests.
- [ ] RED: real short subprocess emits stack on absent progress; reset/cancel/owner switch leave no dangling FD; optional failure cannot kill learner.
- [ ] GREEN:300s non-terminating owned timer; bounded resource samples and warning/replay/PQL/source evidence; source protocol/count in CPU records; named negative exit signal with unknown cause.
- [ ] Verify no-torch supervisor import and existing BASE/PQL/evaluation/checkpoint behavior.

### Task3: Curate and verify Personal evidence

Files: dashboard_columns.py, dashboard.py, guide; tests/test_dashboard.py and test_diagnostics_axes.py.
- [ ] RED→GREEN:17 columns replace redundant Mode with eval n;27plots+guide, cohort count/disk visible and actor loss retained in advanced profile; correct axes,12-run individual cap, no aggregation.
- [ ] Independent final review, source freeze, full offline CPU suite and secret/static checks.
- [ ] Update Personal view only, backup/readback IDs/columns/plots and native UI; no stale data or state rewriting.
- [ ] Coverage matrix/evidence/CURRENT_TASK, commit/push and report measured limits.
