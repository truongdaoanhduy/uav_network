# CBF Safety Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DWA collision avoidance with a minimum-intervention acceleration-space CBF filter for XYZ UAV motion.

**Architecture:** MARL continues to emit normalized acceleration. Native and Warp safety convert it to physical acceleration, project the nominal joint acceleration onto obstacle/peer/boundary CBF half-spaces plus actuation bounds, then integrate the safe acceleration. DWA sampling is removed only after CPU and Warp parity are proven.

**Tech Stack:** Python, NumPy, NVIDIA Warp, pytest; no new optimization dependency.

**Spec:** `docs/superpowers/CBF_SAFETY_DESIGN.md`

## Global Constraints
- Preserve target-search/Bayesian/network semantics.
- Safe nominal acceleration must pass through unchanged.
- Safety may only modify motion acceleration, never RF power.
- Preserve max acceleration/speed, map/altitude bounds, cylinder clearance, and peer separation.
- CPU/Warp behavior must match before DWA removal is complete.

## Review Focus
- Head-on static obstacle at high closing speed must brake before contact.
- Head-on peers must receive a jointly feasible separating correction.
- Parallel safe peers must not trigger intervention.
- Boundary approach must be corrected without changing safe axes.
- Infeasible/inside-safe-set violations must fail closed without NaN/Inf.

---

### Task 1: Native XYZ acceleration CBF core
**Files:**
- Create: `uav_network/environment/cbf.py`
- Modify: `uav_network/environment/xyz_motion.py`
- Test: `tests/test_xyz_cbf.py`

**Interfaces:**
- Produces: `filter_xyz_acceleration(positions, velocities, nominal_accel, obstacles, heights, world) -> (safe_accel, diagnostics)`.

- [x] Write failing tests for pass-through, obstacle braking, peer separation, boundary correction, and finite outputs.
- [x] Run `PYTHONPATH=. pytest -q tests/test_xyz_cbf.py` and confirm RED.
- [x] Implement linear CBF constraints plus fixed-iteration most-violated projection with no new dependency.
- [x] Integrate the CPU XYZ `cbf` branch without deleting DWA yet.
- [x] Run targeted XYZ/native regression tests and `git diff --check`.

### Task 2: Configuration and telemetry migration
**Files:** configuration/defaults/config YAMLs/checkpoint compatibility/metrics tests.
- [x] Make `cbf` the canonical safety mode and remove production DWA selection.
- [x] Rename/alias safety telemetry so W&B describes CBF corrections accurately.
- [x] Preserve legacy checkpoint load behavior explicitly.
- [x] Verify controlled-study configs.

### Task 3: Warp/GPU CBF parity
**Files:** `uav_network/environment/gpu/xyz_motion.py`, GPU state/config plumbing, parity tests.
- [ ] Port the same half-space construction and projection to Warp.
- [ ] Verify CPU/Warp numeric and behavioral parity.
- [ ] Remove XYZ DWA candidate search after parity passes.

### Task 4: Verification and ablation
- [ ] Full test suite and diff check.
- [ ] Deterministic obstacle/peer/boundary stress tests.
- [ ] Short MASAC/MATD3 micro-training comparing intervention/stall/collision metrics.
- [ ] Commit/push only after all checks pass.
