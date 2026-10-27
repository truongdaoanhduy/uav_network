# GitHub Actions External Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Kaggle long-lived controller with a scheduled, idempotent GitHub Actions reconciler that drives continuation and CPU visualization after Sandbox shutdown.

**Architecture:** Production `--auto-continue` jobs emit durable control receipts and never launch their own controller/next stage. A short-lived GitHub Actions workflow reads Kaggle terminal receipts, validates exact recovery through W&B/HF when needed, and submits deterministic child kernels without duplicates.

**Tech Stack:** Python 3.11+, Kaggle CLI/API, W&B API, Hugging Face Hub API, GitHub Actions, pytest/unittest, PyYAML.

**Spec:** `docs/superpowers/specs/2026-10-27-github-external-supervisor-design.md`

## Global Constraints

- Preserve 20,000 episodes, max 1,800 steps/episode, E4096, B8192, replay 1M, seed 44, and final CPU evaluation 30 episodes at seed 2000000044.
- Never place credential values in git, workflow YAML, logs, manifests, or test fixtures derived from real secrets.
- Never silently switch a lineage to another Kaggle account.
- Historical W&B/HF data and old pinned Kaggle jobs remain untouched.
- Exact crash recovery requires a verified resumable full-state artifact; policy-only checkpoints are insufficient.

## Review Focus

- Duplicate scheduled invocations must not submit duplicate child kernels.
- A missing W&B/HF service must not corrupt a normal receipt-driven handoff.
- A hard-killed lineage without full resumable state must block rather than warm restart.
- Disabled/missing GitHub account secrets must be skipped without exposing adjacent secrets.
- Old `continuation-watch` code must not remain reachable from new launch/kernel generation paths.

---

### Task 1: Secret sync and GitHub workflow

**Files:**
- Create: `scripts/sync_github_supervisor_secrets.py`
- Create: `.github/workflows/uav_external_supervisor.yml`
- Create: `tests/test_github_supervisor_workflow.py`

**Interfaces:**
- Consumes: `key.txt` JSON schema from `uav_network.cloud.credentials`.
- Produces: GitHub secret names `KAGGLE_API_TOKEN_ACCOUNT_XX`, `WANDB_API_KEY`, `HF_TOKEN`; nonsecret variables `WANDB_ENTITY`, `WANDB_PROJECT`, `HF_REPO_ID`.

- [ ] Write workflow/secret-sync RED tests: 15-minute schedule + manual dispatch, non-overlapping concurrency, all 24 secret references in one job, no literal token values, secret sync uses stdin and only enabled accounts.
- [ ] Run focused tests and observe RED.
- [ ] Implement minimal sync script and workflow.
- [ ] Run focused tests GREEN.
- [ ] Commit.

### Task 2: Make `--auto-continue` external-only

**Files:**
- Modify: `uav_network/cloud/cli.py`
- Modify: `uav_network/cloud/kernel.py`
- Modify: `tests/test_controller_launch.py`
- Modify: `tests/test_cloud_continuation.py`

**Interfaces:**
- Produces: launch spec field `orchestration="github_actions"`; no Kaggle controller submission; GPU/CPU compute specs remain credential-safe.
- Consumes: Task 1 workflow expecting GitHub-orchestrated specs.

- [ ] RED tests assert auto-continue launch does not call `submit_controller`, records GitHub orchestration, and generated kernels never select `continuation-watch`.
- [ ] Run RED.
- [ ] Remove controller launch from CLI and `continuation-watch` runner generation for new jobs.
- [ ] Run GREEN and related launch tests.
- [ ] Commit.

### Task 3: Stop GPU jobs from self-submitting next stages

**Files:**
- Modify: `uav_network/cloud/remote.py`
- Modify: `uav_network/cloud/pipeline_status.py` if needed.
- Modify/Create tests in `tests/test_cloud_handoff_failures.py` and `tests/test_pipeline_status.py`.

**Interfaces:**
- Produces: `continuation.json` external transition receipt containing desired `gpu` or `cpu` next stage and validated progress.
- Consumes: existing `next_stage(parent, results)` scientific validation.

- [ ] RED tests assert GitHub-orchestrated auto-continue writes a transition receipt and never calls `submit_continuation`/`submit_visualization`.
- [ ] Run RED.
- [ ] Implement receipt-only finalization for GitHub orchestration while preserving non-GitHub legacy behavior only where explicitly requested.
- [ ] Run GREEN and lifecycle regression tests.
- [ ] Commit.

### Task 4: Short-lived external reconciler

**Files:**
- Create: `uav_network/cloud/github_supervisor.py`
- Modify: `uav_network/cloud/visualize.py` to lazy-import torch so orchestration does not install training dependencies.
- Create: `tests/test_github_supervisor.py`

**Interfaces:**
- `credentials_from_environment() -> Credentials bundle/list` from masked GitHub env vars.
- `reconcile_account(account, credentials, *, limit=...) -> list[dict]` scans owned kernels and reconciles terminal GitHub-orchestrated stages.
- Uses deterministic continuation/CPU refs and existing submit functions.

- [ ] RED tests cover active GPU no-op, clean incomplete segment -> one continuation, complete budget -> one CPU job, existing child adoption, unrelated/legacy jobs ignored, partial CPU retry idempotency.
- [ ] Run RED.
- [ ] Implement minimal receipt-driven reconciler with bounded output downloads and deterministic duplicate checks.
- [ ] Run GREEN.
- [ ] Commit.

### Task 5: Verified hard-kill recovery

**Files:**
- Modify: `uav_network/cloud/github_supervisor.py`
- Modify/Create: `uav_network/cloud/continuation.py` exact recovery builder if necessary.
- Extend: `tests/test_github_supervisor.py`.

**Interfaces:**
- Consumes W&B recovery fields `recovery/checkpoint_artifact`, `recovery/last_verified_global_step`, `recovery/last_verified_episodes` and immutable artifact metadata.
- Produces an exact continuation only when the artifact is resumable and provenance matches run/commit/root/account.

- [ ] RED tests: crashed lineage with verified resumable artifact -> deterministic recovery child; missing/policy-only/wrong-run/wrong-commit artifact -> blocked and no push.
- [ ] Run RED.
- [ ] Implement strict recovery validation and exact continuation spec construction.
- [ ] Run GREEN.
- [ ] Commit.

### Task 6: Operational deployment and retirement proof

**Files:**
- Update: `docs/superpowers/CURRENT_TASK.md`
- Add evidence under `docs/superpowers/evidence/`.

**Interfaces:**
- Consumes Task 1 secret-sync command and pushed workflow.
- Produces live GitHub repository secret/variable names, workflow visibility, and a manual dry reconcile receipt without launching duplicate production training.

- [ ] Run focused cloud lifecycle tests, full CPU suite, compile/static checks and credential leak scan.
- [ ] Sync enabled local credentials to GitHub using stdin-only helper; list names only and verify workflow can see required configuration without printing values.
- [ ] Push branch, verify workflow is registered, run manual reconcile in dry/no-transition mode first, inspect result.
- [ ] Confirm no new `continuation-watch` kernel was submitted and no active production run was cancelled/restarted.
- [ ] Record evidence/current task and commit/push.
