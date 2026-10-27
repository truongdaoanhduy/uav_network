# General Failure Hardening Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Complete each checked test cycle sequentially.

**Goal:** General, bounded recovery with isolated account failures and truthful status.
**Architecture:** Extend existing supervisor/continuation/checkpoint metadata; persist a sanitized report in Actions.
**Tech Stack:** Python, pytest, GitHub Actions, existing Kaggle/W&B/HF clients.
**Spec:** docs/superpowers/specs/2026-10-27-general-failure-hardening-design.md

## Global Constraints

No production launch/cancellation, history mutation, secret output, scientific-config/precision changes or account-owner rotation. Do not present a historical external cause as proven.

## Review Focus

One failed account must not starve others; blocked/unknown states must not be green; zero-episode/child recovery needs real environment/prefix evidence; quota waits and cancellation must not submit; retry counter must survive child creation and reset only after a clean segment.

### Task 1: Reproduce and repair failure boundaries

Files: cloud/github_supervisor.py, cloud/continuation.py, experiments/training.py, experiments/pql_training.py, experiments/tracking.py; their existing tests.

- [x] Add failing tests for account isolation/report exit status, zero-episode and continued-segment exact recovery, real published environment-state marker, cancelled GPU/CPU no-op, quota-wait honesty and three-recovery budget.
- [x] Run RED and inspect failures.
- [x] Implement minimal fixes at shared boundaries; report only fixed reason codes and exception types.
- [x] GREEN focused tests and meaningful actual small BASE/PQL snapshot checks.

### Task 2: Make evidence available and verify deployment

Files: workflow, evidence/current-task/operations guide. Interface: --report PATH writes sanitized reconciliation JSON; Actions uploads it with always().

- [ ] Wire report artifact and test CLI report creation on account/lineage failures.
- [ ] Run full offline CPU suite, review, static/secret checks.
- [ ] Commit/push feature, hosted dry-run; fast-forward main after checks; verify report artifact and preserve all production jobs.
- [ ] Record cause assessment, failure-response matrix, test/commit/cloud evidence and limits; push final record.
