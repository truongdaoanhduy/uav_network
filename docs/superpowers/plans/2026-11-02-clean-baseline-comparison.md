# Clean MASAC vs MATD3 Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the canonical XYZ MASAC/MATD3 study a fair vanilla baseline comparison by removing MATD3-only OutputNorm, enforcing equal 20k sample budget, and logging MASAC discrete-entropy diagnostics without changing algorithm math.

**Architecture:** Keep one shared scientific config for both algorithms. BASE remains the config default (`pql.enabled=false`); PQL remains an explicit launch override. Add diagnostics only to the existing MASAC metric return path so W&B/local evidence can observe target entropy and the target-minus-actual gap.

**Tech Stack:** Python, PyTorch, pytest, YAML, W&B tracking.

**Spec:** User-approved 7-point baseline cleanup in the 2026-11-02 conversation.

## Global Constraints
- Do not change reward or CBF behavior.
- Do not change MASAC loss, alpha update, replay, actor, or critic math.
- MATD3 `output_normalization=false` for the baseline.
- BASE default is `pql.enabled=false`.
- Both algorithms target exactly 20,000 completed training episodes; internal soft deadline disabled with `runtime.max_train_seconds=0`.
- Preserve existing alpha and CBF metrics; add `target_discrete_entropy` and `discrete_entropy_gap = target - actual`.

## Review Focus
- Config must not silently alter world/reward/network/sensor between MASAC and MATD3.
- PQL must remain opt-in rather than part of BASE.
- Removing the soft deadline must not alter episode target 20,000.
- Entropy diagnostics must be metrics only and must not retain autograd graphs.
- Existing old runs remain reproducible through their pinned source commits.

### Task 1: Baseline config contract
**Files:** Modify `configs/comparison-20k-hybrid75-1kb-shared-actor-critic-xyz-sensing.yaml`; test `tests/test_controlled_study.py`.
- [x] Write failing test for OutputNorm off, PQL off, 20k episodes, no soft deadline, and matched scientific environment sections.
- [x] Run RED.
- [x] Apply minimal YAML changes.
- [x] Run GREEN.

### Task 2: MASAC entropy telemetry
**Files:** Modify `uav_network/algorithms/masac.py`; test `tests/test_masac.py`.
- [x] Write failing test for target entropy and target-minus-actual gap.
- [x] Run RED.
- [x] Add detached scalar metrics without changing losses.
- [x] Run GREEN.

### Task 3: Local evidence retention
**Files:** Modify `uav_network/experiments/run_journal.py`; test `tests/test_run_journal.py`.
- [x] Write failing test that evidence selector keeps the two entropy diagnostics.
- [x] Run RED.
- [x] Add the two keys to evidence metrics.
- [x] Run GREEN.

### Task 4: Verification and publication
- [x] Run focused tests, then full `pytest -q`.
- [x] Review git diff to ensure reward/CBF/MASAC math unchanged.
- [x] Commit only intended files, leaving pre-existing `docs/superpowers/CURRENT_TASK.md` untouched.
- [x] Push feature branch and verify remote commit through GitHub.
