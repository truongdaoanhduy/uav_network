# Shared Actor + Shared Critic MASAC/MATD3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated controlled 20T/15O/1200 profile that runs MASAC and MATD3 with one shared actor and one shared centralized critic.

**Architecture:** Reuse the existing sharing implementation in `masac.py` and `matd3.py`; change configuration only. Preserve the independent canonical profile and encode the shared topology in the new profile's output/group/tag identity.

**Tech Stack:** Python, PyYAML/dataclass config loader, pytest, PyTorch.

**Spec:** `docs/superpowers/specs/2026-10-29-shared-actor-critic-masac-matd3-design.md`

## Global Constraints
- Keep seed 44, FP32, BASE, E4096, B8192, replay 1M, HYBRID75, 20k episodes, 1200 steps, 20 targets, 15 obstacles.
- Do not alter the existing canonical baseline file.
- Both MASAC and MATD3 must set actor sharing and critic sharing true.
- Distinct W&B/artifact identity is required.

## Review Focus
- Accidental drift in scenario/training hyperparameters: test exact equality against canonical after excluding allowed sharing/identity fields.
- Only one algorithm gets sharing flags: test both MASAC and MATD3.
- Run identity collides with baseline: test output/group/tag include shared identity.
- Existing shared topology/checkpoint/compile code regresses: run `tests/test_parameter_sharing.py`.
- Profile cannot execute: run tiny CPU smoke for both algorithms.

---

### Task 1: Pin the shared profile contract

**Files:**
- Modify: `tests/test_controlled_study.py`
- Create: `configs/comparison-20k-hybrid75-1kb-shared-actor-critic.yaml`

**Interfaces:**
- Consumes: `load_config(path)` and canonical profile.
- Produces: a standalone shared-both profile for MASAC and MATD3.

- [ ] **Step 1: Write the failing profile test**
Assert both algorithms have both sharing flags true, identity is distinct, and the canonical profile remains independent.

- [ ] **Step 2: Run the focused test and verify RED**
Run: `pytest -q tests/test_controlled_study.py -k shared_actor_critic_profile`
Expected: FAIL because the new profile does not exist.

- [ ] **Step 3: Create the new profile**
Copy canonical settings and add both sharing flags under MASAC and MATD3, plus distinct output/group/tag identity.

- [ ] **Step 4: Run focused tests and verify GREEN**
Run: `pytest -q tests/test_controlled_study.py -k 'shared_actor_critic_profile or canonical_easier_profile'`
Expected: PASS.

- [ ] **Step 5: Run sharing regression tests**
Run: `pytest -q tests/test_parameter_sharing.py`
Expected: PASS.

### Task 2: Execute tiny smoke for both algorithms

**Files:** no production code changes expected.

**Interfaces:**
- Consumes: new shared profile.
- Produces: execution evidence for MASAC and MATD3.

- [ ] **Step 1: Run a tiny CPU MASAC smoke**
Use new profile with overrides for CPU, 1 env, 1 episode, 2 steps, compile off, tracking off, tiny hidden/batch/replay.

- [ ] **Step 2: Run a tiny CPU MATD3 smoke**
Use the same shape and limits, changing only algorithm-specific tiny learner settings.

- [ ] **Step 3: Verify resolved configs preserve both sharing flags and complete without non-finite metrics or crashes.**

### Task 3: Full verification and handoff

**Files:**
- Update selectively: `docs/superpowers/CURRENT_TASK.md` without discarding pre-existing unstaged edits.

- [ ] **Step 1: Run `git diff --check` and `python -m compileall -q uav_network`.**
- [ ] **Step 2: Run the relevant controlled-study + parameter-sharing suites.**
- [ ] **Step 3: Record exact verification evidence and next action in CURRENT_TASK without overwriting unrelated dirty content.**
