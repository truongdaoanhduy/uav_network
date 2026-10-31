# MASAC PQL Replicate Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and dispatch one manual GitHub Actions workflow that launches an exact same-seed MASAC PQL replicate on account_06 without exposing credentials.

**Architecture:** A manual-only workflow consumes existing GitHub Secrets, checks out the pinned scientific commit, creates an ephemeral credential file on the runner, and calls the existing cloud launcher. The existing supervisor then owns continuation and CPU evaluation.

**Tech Stack:** GitHub Actions, Python 3.11, existing `uav_network.cloud.cli`, pytest.

**Spec:** `docs/superpowers/designs/2026-11-01-masac-pql-replicate-launch.md`

## Global Constraints
- Scientific source commit must be exactly `0ac776bd20f1abd472c0ed016bb9e3d95f7a95b4`.
- Account must be `account_06`.
- Same MASAC PQL config/seed/overrides as original run `335074a2`.
- No secret values in workflow logs or artifacts.
- Manual dispatch only; no schedule.

## Review Focus
- Workflow cannot accidentally run MATD3 or BASE.
- Workflow cannot silently use a different source commit.
- Credential file stays outside the repository and is removed.
- Launch manifest is sanitized.
- Auto-continuation remains enabled for the new root.

---

### Task 1: Pin the replicate workflow contract
**Files:**
- Create: `.github/workflows/uav_masac_pql_replicate.yml`
- Create: `tests/test_masac_pql_replicate_workflow.py`

- [ ] Write failing workflow contract test.
- [ ] Create the minimal manual workflow pinned to account_06 and commit `0ac776b`.
- [ ] Run targeted workflow tests.
- [ ] Commit and push.

### Task 2: Dispatch and verify the root launch
- [ ] Dispatch the workflow on `feat/cpu-replay-20261022`.
- [ ] Verify workflow success and download sanitized launch artifact.
- [ ] Verify manifest scientific fields and overrides match the original MASAC PQL root.
- [ ] Verify Kaggle root is running and W&B run is live.
- [ ] Verify the external supervisor recognizes the new account_06 root.
