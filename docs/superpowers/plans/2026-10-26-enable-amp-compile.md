# Enable AMP + torch.compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate the repository's existing AMP FP16 and `torch.compile` production paths for future 1 KiB G3-profile runs without altering current pinned jobs or scientific settings beyond these two runtime accelerators.

**Architecture:** Reuse the existing `MixedPrecision`, `compile_models`, BASE/PQL wiring, CUDA execution preflight, and resume guards. Change only the primary YAML runtime flags plus W&B provenance metadata, then verify the already-implemented execution path end-to-end with existing tests.

**Tech Stack:** Python 3, PyTorch 2.11 APIs (`torch.amp`, `torch.compile`/Inductor), pytest, YAML configuration.

**Spec:** `docs/superpowers/specs/2026-10-26-enable-amp-compile-design.md`

## Global Constraints

- Existing eight G3 jobs pinned to source `77a1d0f` must not be restarted, cancelled, or mutated.
- Keep env=4096, batch=8192, replay=1,000,000, CPU75/GPU25 default layout, 1 KiB reports, seed=44, 20k episode budget, rewards/network/scenario and model architecture unchanged.
- Enable only `runtime.amp_mode=fp16` and `runtime.compile_models=true`; keep `fused_optimizers=false`.
- Do not change checkpoint resume compatibility rules; old off/off full-state checkpoints must remain incompatible with on/on exact resume.
- Do not claim speed or memory improvement without a real matched CUDA benchmark.

## Review Focus

- Primary config really resolves to FP16 AMP + compile for future runs.
- W&B group/tags clearly distinguish accelerated runs from historical off/off runs.
- Both MASAC and MATD3 still execute real AMP updates with finite parameters.
- Compile path still executes Dynamo/Inductor graphs without changing state-dict keys.
- Fixed and PQL resume guards continue rejecting acceleration-flag mismatches.

---

### Task 1: Activate acceleration in the primary 1 KiB profile

**Files:**
- Modify: `tests/test_production_hybrid_replay.py`
- Modify: `configs/comparison-20k-hybrid75-1kb.yaml`

**Interfaces:**
- Consumes: `load_config(path)` and existing runtime fields `amp_mode`, `compile_models`.
- Produces: the primary profile resolving to `amp_mode == "fp16"` and `compile_models is True`, with W&B provenance tags/group naming the acceleration mode.

- [ ] **Step 1: Write the failing test**

Extend `test_harp_lawn_matched_profile_uses_one_kib_report` to assert:
- `c.runtime.amp_mode == "fp16"`
- `c.runtime.compile_models is True`
- `c.runtime.fused_optimizers is False`
- tracking group contains `ampfp16-compile`
- tags include `amp-fp16` and `torch-compile`

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q tests/test_production_hybrid_replay.py::test_harp_lawn_matched_profile_uses_one_kib_report`
Expected: FAIL on the AMP/compile assertions because the current profile is off/off.

- [ ] **Step 3: Implement the minimal profile change**

In `configs/comparison-20k-hybrid75-1kb.yaml`:
- add/set `amp_mode: fp16`
- set `compile_models: true`
- keep `fused_optimizers: false`
- change tracking group to include `ampfp16-compile`
- add tags `amp-fp16`, `torch-compile`

- [ ] **Step 4: Run test to verify it passes**

Run the same focused test.
Expected: PASS.

### Task 2: Verify existing production execution and resume contracts

**Files:**
- Modify: `docs/superpowers/evidence/2026-10-26-amp-compile-activation.md`
- Modify: `docs/superpowers/CURRENT_TASK.md`

**Interfaces:**
- Consumes: activated profile from Task 1 and existing AMP/compile/checkpoint code.
- Produces: durable evidence of local verification and a current-task record stating what was and was not proven.

- [ ] **Step 1: Run focused execution regressions**

Run: `HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q tests/test_precision.py tests/test_production_execution.py tests/test_production_hybrid_replay.py tests/test_baseline_resume.py`
Expected: 0 failures.

- [ ] **Step 2: Run full offline CPU suite**

Run: `HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q`
Expected: 0 failures; CUDA-only tests may skip on this host.

- [ ] **Step 3: Verify repository hygiene**

Run: `git diff --check && python3 -m compileall -q uav_network`
Expected: exit 0.

- [ ] **Step 4: Record evidence and current task**

Document the exact commands/results, changed profile flags, unchanged scientific settings, source pin isolation for active G3 jobs, and explicit limitation that no new real-T4 performance benchmark was run.

- [ ] **Step 5: Commit and push**

Commit tracked changes on `feat/cpu-replay-20261022`, preserving the existing untracked `wandb/`, then push according to `AGENTS.md`.
