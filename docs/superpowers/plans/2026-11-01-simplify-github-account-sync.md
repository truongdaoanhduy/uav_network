# Simplify GitHub Account Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Kaggle account naming identical between `key.txt` and GitHub Actions: `account_NN` always maps to `KAGGLE_API_TOKEN_ACCOUNT_NN`, with no `github_account` remapping.

**Architecture:** `key.txt` is the single source of truth for account names. The sync script derives both GitHub secret names and `KAGGLE_ENABLED_ACCOUNTS` directly from `name`; any legacy `github_account` field is rejected so renumbering cannot silently return. The workflow keeps generic secret slots 01..24, while the allowlist controls which are active.

**Tech Stack:** Python 3.11, pytest, GitHub CLI, GitHub Actions YAML.

**Spec:** User request in current conversation: remove confusing GitHub account renumbering and use 1:1 account names.

## Global Constraints
- Never print Kaggle/W&B/HF secret values.
- Do not modify or restart currently running training jobs.
- Preserve existing `docs/superpowers/CURRENT_TASK.md` changes.
- GitHub supervisor must continue to reconcile existing lineages by authenticated Kaggle owner.

## Review Focus
- Legacy `github_account` cannot silently remap an account.
- Duplicate local account names fail closed.
- Disabled accounts are not placed in the allowlist.
- GitHub stale account secrets are removed after successful resync.
- Existing running lineages remain visible in supervisor dry-run.

---

### Task 1: Enforce one-to-one sync semantics
**Files:**
- Modify: `scripts/sync_github_supervisor_secrets.py`
- Modify: `tests/test_github_supervisor_workflow.py`

- [x] Write RED tests for rejecting `github_account` and duplicate local names.
- [x] Implement direct `name` -> same-number GitHub secret mapping.
- [x] Run targeted supervisor/workflow tests.

### Task 2: Normalize credentials and resync GitHub
**Files:**
- Modify external/private: `key.txt` only if legacy fields exist.

- [x] Ensure enabled accounts are named sequentially and contain no `github_account` field.
- [x] Run sync script against `truongdaoanhduy/uav_network`.
- [x] Remove stale Kaggle secrets not represented by enabled `key.txt` names.
- [x] Verify actual secret-name set equals expected set.

### Task 3: End-to-end verification
- [x] Run cloud-related regression tests and `git diff --check`.
- [x] Commit/push code changes without `CURRENT_TASK.md`.
- [x] Run GitHub supervisor dry-run and verify existing running lineages are still recognized.

## Verification
- Cloud regression: 55 passed, 23 subtests passed.
- GitHub supervisor dry-run: run `41239997446`, conclusion `success`, head `d07a2a2`.
- Supervisor account set: `account_01` through `account_16`.
- Running lineages remained active on accounts 01, 02, 03, 04, 05, and 07.
