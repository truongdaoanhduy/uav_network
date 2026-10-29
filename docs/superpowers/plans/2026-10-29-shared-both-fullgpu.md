# Shared Actor + Critic FULLGPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a FULLGPU Shared Actor + Shared Critic training profile for MASAC and MATD3 while preserving the HYBRID75 control.

**Architecture:** Reuse existing learner topology. Only replay placement and run identity differ from the HYBRID shared-both profile.

**Tech Stack:** YAML config, Python config loader, pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-shared-both-fullgpu-design.md`

## Global Constraints
- FULLGPU means `replay_device: cuda`, `replay_cpu_fraction: 0`.
- Preserve all scientific/training settings other than replay placement and identity metadata.
- Preserve HYBRID shared-both file unchanged.

## Review Focus
- Both algorithms retain actor+critic sharing.
- FULLGPU profile does not retain HYBRID tags/names.
- Scientific settings remain matched.
- Run identity resolves to FULLGPU.
- Existing HYBRID profile remains unchanged.

---

### Task 1: Pin and create FULLGPU profile
**Files:** Modify `tests/test_controlled_study.py`; Create `configs/comparison-20k-fullgpu-1kb-shared-actor-critic.yaml`.
- [ ] Add failing test for FULLGPU profile parity with HYBRID shared-both.
- [ ] Verify RED because the profile is absent.
- [ ] Create profile with replay on CUDA and distinct FULLGPU metadata.
- [ ] Verify GREEN.

### Task 2: Verify execution identity and regressions
- [ ] Run controlled-study, parameter-sharing, cloud-identity focused tests.
- [ ] Resolve run names/tags for MASAC and MATD3 and confirm FULLGPU.
- [ ] Run compileall and diff check.
- [ ] Record handoff without overwriting pre-existing CURRENT_TASK edits.
