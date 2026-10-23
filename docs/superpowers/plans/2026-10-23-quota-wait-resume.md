# Quota-safe Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve 20,000-episode/1,800-step scientific targets while making a completed Kaggle segment wait safely for same-account quota refresh instead of marking the pipeline failed, and provide a verified resume path from kernel outputs.

**Architecture:** Keep the existing immutable checkpoint + exact environment/RNG continuation. Treat only `quota_unavailable`/`insufficient_quota` as a non-failure `waiting_for_quota` scheduling state. Add a minimal CLI resume command that downloads only `job.json`, `job-result.json`, and `continuation.json` from a completed kernel and reuses the existing `submit_continuation` validation; no cross-account rotation.

**Tech Stack:** Python, pytest, Kaggle CLI, W&B.

**Spec:** `docs/superpowers/CURRENT_TASK.md`

## Global Constraints
- Keep episodes=20000 and max_steps=1800 unchanged.
- Keep algorithm/scenario/replay settings unchanged.
- Resume only the same Kaggle account and pinned commit.
- Never copy credentials into manifests/logs.
- Do not restart/cancel healthy canonical jobs.

## Review Focus
- Insufficient quota must not become learner failure.
- Real credential/submission errors must still fail.
- Resume must reject a running kernel or a non-waiting/non-time-budget result.
- Resume must use exact parent `job.json` and `job-result.json` and existing continuation validation.
- No cross-account continuation.

### Task 1: Waiting-for-quota lifecycle
**Files:** modify `uav_network/cloud/continuation.py`, `uav_network/cloud/remote.py`, `uav_network/cloud/pipeline_status.py`; test `tests/test_pipeline_status.py`.
- [x] Add failing tests for quota exhaustion returning `waiting_for_quota` without push and pipeline status not failed.
- [x] Implement minimal quota-wait status while keeping other errors as failures.
- [x] Run targeted tests and full suite.

### Task 2: Verified resume command
**Files:** modify `uav_network/cloud/cli.py`; test `tests/test_cloud_continuation.py` or new focused test.
- [x] Add failing test for resume downloading only metadata/result files and calling existing `submit_continuation` with same account.
- [x] Implement minimal `resume` CLI path using Kaggle `kernels output --file-pattern` and existing validators.
- [x] Run targeted tests and full suite.

### Task 3: Documentation and verification
- [x] Record root cause, feasibility, and exact completion criteria in `docs/superpowers/CURRENT_TASK.md`.
- [x] `git diff --check`, full tests, commit, push, remote read-back.
