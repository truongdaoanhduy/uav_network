# MASAC Power Causal Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a bounded causal test of MASAC TX-power critic miscalibration without changing SAC mathematics.

**Architecture:** Correct checkpoint provenance first, then add an opt-in runtime collector intervention and a standalone calibration probe. The A/B resumes the same 8192-episode PQL checkpoint so model, optimizer, replay, environment and RNG lineage are matched; only collector TX-power support differs.

**Tech Stack:** Python, PyTorch, Warp packet simulator, pytest, W&B/Kaggle cloud launcher.

**Spec:** `docs/superpowers/specs/2026-11-03-masac-power-causal-audit.md`

## Global Constraints
- Base commit is `05beef6cd0d97d39d51b220f9f7789e915b5bc10`.
- No SAC loss/reward/entropy/optimizer/scenario changes.
- No 20k training.
- `runtime.power_support_fraction` defaults to `0.0` and is diagnostic-only.
- A/B uses identical 8192-episode PQL full resume checkpoint and different Kaggle accounts only for parallel wall-clock execution.

## Review Focus
- Version 4 XYZ checkpoints without coarse belief must still validate.
- Version 5 XYZ coarse-belief checkpoints saved by the same config must validate.
- Power support must be a no-op at fraction 0 and in deterministic evaluation.
- Intervention must not alter destination or XYZ motion components.
- Calibration must restore the exact same simulator snapshot before each branch.

---

### Task 1: Fix observation-version validation
**Files:** Modify `uav_network/algorithms/checkpoints.py`; Test `tests/test_checkpoint_observation_version.py`.

**Interfaces:** Produces config-derived `validate_observation_version(saved)` compatible with versions 3/4/5 according to saved scenario config.

- [ ] Add failing tests for XYZ coarse-belief version 5 and XYZ non-coarse version 4.
- [ ] Run tests and verify the version-5 case fails on the hard-coded validator.
- [ ] Implement config-derived expected observation version.
- [ ] Run checkpoint/evaluation snapshot tests and commit.

### Task 2: Add runtime-only power-support intervention
**Files:** Modify `uav_network/configuration.py`, `uav_network/experiments/training.py`, `uav_network/experiments/pql_training.py`; Create `uav_network/experiments/power_support.py`; Test `tests/test_power_support.py`.

**Interfaces:** Produces `runtime.power_support_fraction: float = 0.0` and collector-only behavior that deterministically covers a fixed power grid only for eligible non-silent MASAC training actions while preserving the scientific MASAC source hash.

- [ ] Add failing tests for fraction 0 no-op, deterministic no-op, selected-row power override, and unchanged destination/motion.
- [ ] Validate fraction is in `[0,1]`.
- [ ] Implement deterministic selection/grid without consuming RNG.
- [ ] Run MASAC/PQL/reproducibility tests and commit.

### Task 3: Add critic-calibration probe
**Files:** Create `uav_network/experiments/power_calibration.py`, `scripts/probe_masac_power_calibration.py`; Test `tests/test_power_calibration.py`.

**Interfaces:** Produces JSON metrics including `critic_q_high_minus_low`, `environment_return_high_minus_low`, `delivery_high_minus_low`, `phy_success_high_minus_low`, `ordering_agreement`, and support/action diagnostics.

- [ ] Add failing unit tests for matched snapshot restoration and metric sign/order computation using a small deterministic fixture.
- [ ] Implement probe helpers and CLI.
- [ ] Verify probe against downloaded 8192/16362 MASAC checkpoints on CPU Warp.
- [ ] Commit.

### Task 4: Define and verify bounded A/B
**Files:** Create `configs/diagnostic-masac-power-support.yaml`; modify cloud identity/tagging only if needed; Test config/cloud parsing.

**Interfaces:** Baseline override `runtime.power_support_fraction=0.0`; intervention `0.10`; both resume `uav-network-masac-2ce9fb9e:v13` and stop at a bounded diagnostic budget, never 20k.

- [ ] Add config test proving requested episode/time budget is bounded and scenario/learner settings match cohort.
- [ ] Create diagnostic config using cohort settings, PQL enabled, requested episodes below 20k, and bounded `max_train_seconds`.
- [ ] Run local config/preflight tests and commit.

### Task 5: Execute matched cloud A/B and analyze
**Files:** No production code unless a test exposes a bug.

**Interfaces:** Two distinct Kaggle accounts; same pinned commit and warm-start artifact; W&B outputs plus pre/post calibration JSON.

- [ ] Check account quota/active kernels and select two free accounts.
- [ ] Launch baseline and power-support run in parallel from the same full resume artifact.
- [ ] Verify manifests differ only by account/run identity and `runtime.power_support_fraction`.
- [ ] Collect final metrics/logs/checkpoints and run calibration probe on both.
- [ ] Classify support-collapse hypothesis as supported, rejected, or unresolved from predefined criteria.
