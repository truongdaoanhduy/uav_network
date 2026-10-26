# Measured Acceleration Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode the measured T4/G3 acceleration policy in the canonical cloud production launcher while preserving explicit overrides and benchmark freedom.

**Architecture:** Add a small pure cloud policy helper that appends only missing acceleration overrides for the canonical profile in train/validate stages. PQL `both` production launch is rejected because MATD3 and MASAC have different measured compile choices; single-algorithm policy becomes ordinary spec overrides, so continuation needs no new mechanism. W&B tags are derived from resolved runtime flags.

**Tech Stack:** Python 3, dataclass YAML config, pytest, Kaggle cloud launcher.

**Spec:** `docs/superpowers/specs/2026-10-27-measured-acceleration-policy-design.md`

## Global Constraints

- Canonical scope only: `configs/comparison-20k-hybrid75-1kb.yaml`, cloud `train`/`validate`.
- Explicit runtime acceleration overrides win.
- Benchmark stage remains policy-free.
- Fused Adam remains OFF by measured default.
- Existing production lineages/checkpoints are not changed.

## Review Focus

- Exact path scoping must not affect copied/other configs.
- `--algorithm both` must remain valid for BASE and benchmark but refuse canonical PQL production.
- Explicit compile override for MATD3 PQL must survive policy application.
- Effective overrides must be present in the manifest so continuation inherits them.
- W&B tags must not retain stale `torch-compile` when compile resolves false.

---

### Task 1: Pure measured acceleration policy

**Files:**
- Create: `uav_network/cloud/acceleration.py`
- Create: `tests/test_acceleration_policy.py`

**Interfaces:**
- Produces: `apply_measured_acceleration_policy(config_path, stage, algorithm, config, overrides, canonical_path) -> list[str]`.

- [ ] Write failing tests for the four measured modes, benchmark/noncanonical no-op, explicit-override precedence, and PQL-both refusal.
- [ ] Run focused tests and confirm RED because the helper does not exist.
- [ ] Implement the minimal pure helper.
- [ ] Run focused tests and confirm GREEN.

### Task 2: Launcher integration and provenance

**Files:**
- Modify: `uav_network/cloud/cli.py`
- Modify: `uav_network/cloud/identities.py`
- Modify: `configs/comparison-20k-hybrid75-1kb.yaml`
- Modify: `tests/test_cloud_identities.py`
- Modify: `tests/test_acceleration_policy.py`

**Interfaces:**
- Consumes the Task 1 helper.
- Produces manifests whose normal `overrides` contain the effective measured policy and whose tags match resolved flags.

- [ ] Add failing launch/tag integration tests: MATD3 PQL gets compile false; MASAC PQL gets true; BASE both stays valid; benchmark is untouched; stale acceleration tags are removed.
- [ ] Verify RED.
- [ ] Apply policy before final config-dependent launch metadata, reload config from effective overrides, and make authoritative tags runtime-derived.
- [ ] Change canonical tracking group to acceleration-neutral naming.
- [ ] Verify focused cloud/tag tests GREEN.

### Task 3: Regression verification and durable evidence

**Files:**
- Modify: `docs/superpowers/CURRENT_TASK.md`
- Create: `docs/superpowers/evidence/2026-10-27-measured-acceleration-policy.md`

- [ ] Run focused acceleration/cloud/resume tests.
- [ ] Run full offline pytest suite.
- [ ] Run `git diff --check` and `python3 -m compileall -q uav_network`.
- [ ] Record exact results and policy matrix; preserve untracked `wandb/`.
- [ ] Commit source/tests first, then docs/evidence, and push branch.
