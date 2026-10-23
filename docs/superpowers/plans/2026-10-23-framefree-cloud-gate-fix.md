# Framefree Cloud Gate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the experimental Framefree CUDA exactness gate verifiable without private-kernel owner credentials, while pinning the exact branch commit being tested.

**Architecture:** Add one small, credential-free launcher module for the Framefree CUDA gate. It generates a public Kaggle diagnostic kernel with no dataset/model/credential sources, pins a supplied 40-char Git commit, writes the exactness result to kernel output, and verifies status/output using the normal Kaggle CLI credential. Production cloud kernels remain private and unchanged.

**Tech Stack:** Python, pytest, Kaggle CLI, Git.

**Spec:** `docs/superpowers/plans/2026-10-23-framefree-bitmap.md`

## Global Constraints
- Do not modify production training kernel privacy or credential handling.
- Diagnostic kernel must contain no W&B/Kaggle/HF secrets or private datasets.
- Diagnostic kernel must be public so `kernels.get` verification is not tied to the submitter account.
- CUDA gate must test the exact supplied commit, never a hard-coded stale SHA.
- Do not launch a new Kaggle GPU session while the current environment blocks the intended credential path; fix and verify generation locally first.

## Review Focus
- Public metadata contains no credential datasets.
- Commit validation rejects non-40-char lowercase SHA values.
- Kernel result records CUDA availability, GPU identity, pytest return code/stdout/stderr.
- Verification refuses a missing/failed result instead of treating a kernel status as scientific success.
- Production `uav_network.cloud.kernel_metadata()` remains private.

### Task 1: Public Framefree CUDA gate generator
**Files:** create `uav_network/experiments/framefree_cuda_gate.py`; create `tests/test_framefree_cloud_gate.py`.

- [x] Write failing tests for public metadata, secret-free sources, dynamic commit pinning, and result validation.
- [x] Run focused tests and verify RED.
- [x] Implement the minimum generator/validator.
- [x] Run focused tests and verify GREEN.

### Task 2: CLI wrapper and documentation
**Files:** create `scripts/framefree_cuda_gate.py`; update `docs/superpowers/plans/2026-10-23-framefree-bitmap.md`.

- [x] Add CLI contract test for `prepare` output directory and metadata.
- [x] Implement `prepare` and `verify` commands using the module from Task 1.
- [x] Verify no production kernel path changed.

### Task 3: End-to-end verification
- [x] Run Framefree focused parity tests.
- [x] Run full branch suite.
- [x] Run `git diff --check` and compile checks.
- [ ] Commit and push the Framefree branch; verify GitHub remote read-back.
