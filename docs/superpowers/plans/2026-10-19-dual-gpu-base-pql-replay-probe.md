# Dual-GPU BASE/PQL Replay Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BASE and PQL replay probes genuinely use both Kaggle T4 GPUs and measure the largest passing replay capacity from 5M down to 1M at 4096 envs and batch 8192.

**Architecture:** Reuse PQL's existing role-split pattern. Add the minimum BASE collector/learner split: GPU0 owns Warp collection/inference, GPU1 owns replay and the authoritative learner; synchronize actor weights only at collection boundaries. Keep replay unsharded.

**Tech Stack:** Python, PyTorch CUDA explicit device placement, NVIDIA Warp, Kaggle runner, W&B benchmark summaries.

**Spec:** `docs/superpowers/specs/2026-10-19-dual-gpu-base-pql-replay-probe-design.md`

## Global Constraints
- Preserve MASAC/MATD3 algorithm semantics and deterministic seed behavior as far as explicit cross-device execution permits.
- BASE and PQL must both visibly and materially use two GPUs when two are requested.
- Replay capacity sequence is exactly 5M, 4M, 3M, 2M, 1M; envs=4096; batch=8192.
- No replay sharding, AMP, compile, or DDP in this change.
- Do not include the existing untracked `configs/comparison-20k-steps2000-scenario-v3-replay2m.yaml` in unrelated commits.

## Review Focus
- One visible GPU must preserve existing BASE behavior instead of requiring cuda:1.
- Two visible GPUs must put collector/env on cuda:0 and replay/learner on cuda:1.
- Cross-device replay inserts must copy correctly without retaining references to mutable GPU0 buffers.
- Collector policy must refresh after learner actor updates without changing action semantics.
- Kaggle runner must grant both GPUs to BASE probe jobs instead of the current one-GPU-per-learner scheduling.

---

### Task 1: BASE Device Roles

**Files:**
- Modify: `uav_network/experiments/training.py`
- Test: `tests/test_gpu_cuda.py` or a focused new BASE device test in the existing CUDA test module

**Interfaces:**
- Produces: BASE role resolution returning environment/collector device and learner/replay device.

- [ ] Write failing tests for 1-GPU fallback and 2-GPU role split.
- [ ] Run focused tests and verify RED.
- [ ] Implement the minimum explicit role resolver and use it in BASE training.
- [ ] Add a read-only collector inference copy on GPU0 only when learner is on a different device; refresh actors after learner policy updates.
- [ ] Run focused tests and verify GREEN.

### Task 2: BASE Dual-GPU Benchmark Evidence

**Files:**
- Modify: `uav_network/experiments/benchmark.py` or add the smallest dedicated replay-capacity probe helper if existing benchmark cannot exercise collection + actor update.
- Test: `tests/test_benchmark_devices.py`

**Interfaces:**
- Produces: result fields for devices, replay/environment bytes, update counts, peak memory, and pass/OOM status.

- [ ] Write a failing test that a two-GPU BASE benchmark cannot report pass unless learner and collector roles are distinct and at least one actor update occurred.
- [ ] Verify RED.
- [ ] Implement benchmark reporting and OOM capture with per-GPU memory counters.
- [ ] Verify GREEN.

### Task 3: Kaggle Dual-GPU Allocation for BASE Probe

**Files:**
- Modify: `uav_network/cloud/remote.py`
- Test: `tests/test_cloud.py`

**Interfaces:**
- Produces: benchmark/probe jobs that set `CUDA_VISIBLE_DEVICES=0,1` for BASE as well as PQL when dual-GPU replay probing.

- [ ] Write failing tests for BASE dual-GPU probe allocation while preserving ordinary independent-run scheduling.
- [ ] Verify RED.
- [ ] Implement a probe-only dual-GPU allocation path; do not globally force all BASE training runs to consume both GPUs.
- [ ] Verify GREEN.

### Task 4: Verification and Pinned Commit

**Files:**
- No new production files unless tests reveal a bug.

- [ ] Run focused CUDA/device/cloud tests.
- [ ] Run `python -m pytest tests -q` and require zero failures.
- [ ] Run `git diff --check`.
- [ ] Commit only the dual-GPU probe changes and docs; exclude the untracked replay2m draft.
- [ ] Push the branch and record the pinned commit SHA.

### Task 5: Kaggle Replay Capacity Sweep

**Files:**
- Generated outputs only under `outputs/`.

**Interfaces:**
- Consumes: pinned commit from Task 4.
- Produces: BASE and PQL reports for capacities 5M, 4M, 3M, 2M, 1M until the largest passing value is identified.

- [ ] Launch BASE MATD3 dual-GPU 5M probe and verify both GPUs in hardware/device evidence.
- [ ] Launch PQL MATD3 dual-GPU 5M probe with the same env/batch.
- [ ] For each mode independently, if OOM, repeat 4M -> 3M -> 2M -> 1M without changing env/batch.
- [ ] Download/read job reports and logs; distinguish OOM from infrastructure failure.
- [ ] Summarize largest passing replay per mode, peak VRAM per GPU, and compare the winning config to FlashSAC/FastTD3 on envs, replay depth, batch, updates/sample-use, warmup, compile, and AMP.
