# Shield-Aware Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make shield-dependent unsafe policies score worse even when CBF prevents collisions, while exposing raw-policy safety separately from executed safety.

**Architecture:** Compute raw hard-safety violations from nominal motion before CBF, carry their counts/fraction through native and Warp telemetry, and charge a dedicated reward term. Keep CBF geometry unchanged, but normalize its correction in acceleration/action space. Rename training-history aggregates so W&B cannot present them as final-policy evaluation.

**Tech Stack:** Python, NumPy, NVIDIA Warp, pytest, W&B metrics.

**Spec:** `docs/superpowers/specs/2026-11-02-shield-aware-reward.md`

## Global Constraints
- Preserve CPU/Warp parity.
- Preserve legacy config compatibility with `reward.unsafe_action = 0.0` by default.
- Do not touch unrelated dirty checkpoint files.
- No remote push in this task.

## Review Focus
- Raw boundary command is penalized even if CBF prevents clipping.
- Safe command receives zero unsafe-action penalty and no CBF correction.
- Peer and obstacle raw attempts count every involved UAV once per step.
- Disabled/battery-dead UAVs do not contribute to unsafe fractions.
- CPU and Warp reward-component accounting still sums to episode return.

---

### Task 1: Raw-policy safety telemetry and reward
**Files:** `uav_network/environment/xyz_motion.py`, `uav_network/environment/gpu/xyz_motion.py`, `uav_network/environment/env.py`, `uav_network/environment/gpu/state.py`, `uav_network/communication/gpu/reports.py`, `uav_network/communication/gpu/metrics.py`, `uav_network/configuration.py`; tests in `tests/test_xyz_cbf.py`, `tests/test_xyz_cbf_warp.py`, `tests/test_gpu_physics.py`.

- [x] Write failing native tests for raw unsafe attempts, shield rescue, and acceleration-normalized correction.
- [x] Write failing Warp parity/reward test.
- [x] Run RED tests and confirm failures are missing telemetry/reward semantics.
- [x] Implement native telemetry and reward.
- [x] Implement Warp state/kernel/reward parity.
- [x] Run GREEN targeted tests.

### Task 2: W&B policy-safety metrics and honest aggregate names
**Files:** `uav_network/experiments/tracking.py`, `uav_network/experiments/comparison_metrics.py`, `uav_network/experiments/dashboard.py`; tests in `tests/test_tracking.py`, `tests/test_comparison_metrics.py`.

- [x] Write failing tests for raw unsafe/rescue rates and training-history aggregate names.
- [x] Run RED tests.
- [x] Implement metrics/dashboard labels and remove misleading `final_*` training-history aliases.
- [x] Run GREEN targeted tests.

### Task 3: Canonical config and verification
**Files:** `configs/comparison-20k-hybrid75-1kb-shared-actor-critic-xyz-sensing.yaml`, source-contract hash files if required by tests.

- [x] Enable `reward.unsafe_action` in canonical config with coefficient `1.0`.
- [x] Run CBF/reward/tracking/comparison parity tests.
- [x] Run full pytest suite and `git diff --check`.
- [x] Run a small matched MATD3 smoke/eval to confirm shield-dependent floor-diving earns a clearly worse safety reward and new telemetry is nonzero.

## Verification evidence

- Targeted regression: `113 passed, 1 skipped, 531 subtests passed`.
- Full suite: `1200 passed, 103 skipped, 1597 subtests passed` in 647.39 s.
- Native floor-dive smoke at z=0.5 m: CBF prevents physical violation, while reward reports `unsafe_action=-1.0`, `shield_intervention=-0.5`, `shield_correction=-0.075`.
- Two-step target smoke: safe hover searched 100% with return +9.84; repeated floor-dive searched 0% with return -3.40.
- MATD3 micro 30-seed deterministic evaluation after 128 training episodes: action_z mean +0.429, raw unsafe 85.5%, CBF 87.8%, return -121.3. Bad behavior is now scored as bad.
- Extending the same micro setup to 512 episodes reduced raw unsafe 85.5% -> 50.5% and CBF intervention 87.8% -> 60.9%; mean action_z moved +0.429 -> +0.183. This is directional evidence only, not a claim of converged policy quality.
