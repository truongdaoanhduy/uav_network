# Scenario V3 Streaming Reports and Radio Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 1 MB report relay practical at 4096 environments with bounded packet memory and physically explicit radio TX/RX battery accounting, then freeze a separate Scenario V3 profile without changing MASAC/MATD3.

**Architecture:** Replace dense per-session/per-fragment Warp state with bounded streaming session progress and queue refill, while keeping packet slots as the materialized fragment window. Add shared configurable radio hardware energy accounting to native and Warp paths, extend pooled diagnostics, and freeze the 1 MB workload in a new V3 config.

**Tech Stack:** Python 3.13, NVIDIA Warp, PyTorch, SimPy/UavNetSim, unittest/pytest, W&B/Kaggle launch tooling.

**Spec:** `docs/superpowers/specs/2026-10-18-scenario-v3-streaming-radio-energy-design.md`

## Global Constraints

- Preserve `configs/comparison-25k-scenario-v2.yaml` and existing V2 semantics.
- V3 mission constants: 6 UAVs, 35 targets, 25 obstacles, map 1000 m, max_steps 1800, seed 44, num_envs 4096, target episodes 25000.
- V3 network constants: report 1,000,000 B; buffer 3,000,000 B; payload 1024 B; queue 64; TTL 300 s; 2 Mbps; peer/GCS range 500 m; TX 0.001-0.4 W log mapping; max retries 5.
- MASAC and MATD3 architectures and action semantics remain unchanged.
- No rule-based forced relay or automatic minimum-power override.
- New radio hardware fields preserve legacy behavior by default: PA efficiency 1.0, TX circuit 0 W, RX circuit 0 W.
- Physical radio energy and reward shaping stay separate; first V3 baseline keeps `propulsion_energy_per_kj=0.01` and adds no power-margin reward.
- All behavior changes follow RED -> GREEN -> full-suite verification.

## Review Focus

- A 1 MB report with queue 64 must stream to completion in one live session rather than stall at the first window.
- Retry/lifetime cleanup must not skip a fragment forever or double-count completed bytes.
- Session state memory must remain independent of total fragment count.
- Native and Warp energy accounting must charge TX/RX consistently and must not drain UAV battery for GCS energy.
- Legacy V2/default configs must reproduce old energy semantics and remain loadable.

---

### Task 1: Bounded streaming fragment sessions

**Files:**
- Modify: `uav_network/environment/gpu/state.py`
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `uav_network/communication/gpu/reports.py`
- Modify: `uav_network/communication/backend.py`
- Test: `tests/test_gpu_packets.py`
- Test: `tests/test_communication.py`
- Test: `tests/test_retry_recovery.py`

**Interfaces:**
- Produces: bounded per-session progress state and refill behavior used by both direct and multi-hop report transfer.
- Preserves: packet-level fragment index and existing report custody/hop semantics.

- [ ] **Step 1: Add failing tests for a report larger than one queue window**
  - Use report size large enough to require more than `max_queue_packets` payloads.
  - Assert direct delivery completes without a second RL action solely to refill the queue.
  - Assert generated packet/window occupancy never exceeds queue capacity.

- [ ] **Step 2: Run focused tests and confirm RED for queue-window truncation/dense fragment assumptions.**

- [ ] **Step 3: Replace dense Warp `[session, fragment]` arrays with O(1) session progress plus bounded packet-slot state.**
  - Keep exact packet fragment indices in packet slots.
  - Ensure success/drop/expiry frees capacity and schedules/refills the next missing fragment while session TTL remains live.

- [ ] **Step 4: Make native backend refill the same live transfer as queue slots free inside the current network interval.**

- [ ] **Step 5: Add deterministic direct, two-hop and three-hop large-report tests for native/Warp CPU semantics.**

- [ ] **Step 6: Run focused packet/report/retry suites, then full `pytest -q`; commit.**

### Task 2: Shared physical TX/RX radio energy

**Files:**
- Modify: `uav_network/configuration.py`
- Modify: `uav_network/communication/energy.py`
- Modify: `uav_network/communication/gpu/phy.py`
- Modify: `uav_network/communication/gpu/state.py`
- Modify: `uav_network/communication/backend.py`
- Modify: `uav_network/communication/gpu/metrics.py`
- Modify: `uav_network/environment/env.py`
- Test: `tests/test_energy.py`
- Test: `tests/test_gpu_physics.py`
- Test: `tests/test_communication.py`

**Interfaces:**
- Produces configuration fields `pa_efficiency`, `tx_circuit_power_w`, `rx_circuit_power_w` and matched TX/RX joule accounting.
- Consumes Task 1 packet/session lifecycle so each frame is charged exactly once per TX and receiver listen event.

- [ ] **Step 1: Add failing validation and formula tests for PA efficiency and TX/RX circuit power; verify legacy defaults.**
- [ ] **Step 2: Add failing native/Warp tests that failed TX attempts consume TX energy and failed decodes consume RX listening energy.**
- [ ] **Step 3: Implement shared energy formulas and native accounting, including GCS separation.**
- [ ] **Step 4: Implement matching Warp accounting and expose per-UAV TX, RX and total radio energy.**
- [ ] **Step 5: Verify battery decreases only for UAV energy and legacy defaults match previous RF-only values.**
- [ ] **Step 6: Run focused energy/physics suites, then full `pytest -q`; commit.**

### Task 3: V3 config, diagnostics and resource verification

**Files:**
- Create: `configs/comparison-25k-scenario-v3.yaml`
- Modify: `uav_network/experiments/telemetry.py`
- Modify: `uav_network/experiments/tracking.py`
- Modify: `uav_network/experiments/comparison_metrics.py`
- Modify: `uav_network/experiments/rendering.py` only if needed to display TX/RX split without changing mission semantics.
- Modify/Create: resource estimator/benchmark under existing `uav_network/experiments/` or `scripts/` pattern.
- Test: `tests/test_network_telemetry.py`
- Test: `tests/test_tracking.py`
- Test: `tests/test_comparison_metrics.py`
- Test: `tests/test_diagnostics_v1.py`

**Interfaces:**
- Produces a frozen V3 workload and W&B diagnostics that can distinguish underpower, over-power, direct/relay behavior and physical radio energy.

- [ ] **Step 1: Add failing tests that V3 resolves to the exact mission/network constants in Global Constraints and V2 remains unchanged.**
- [ ] **Step 2: Add failing pooled-metric tests for TX/RX radio joules, radio J/delivery and existing power/relay diagnostics.**
- [ ] **Step 3: Implement the V3 config and telemetry/tracking/comparison aggregation.**
- [ ] **Step 4: Add resource-state estimator assertions for env counts 1, 64, 512, 4096 and prove session allocation does not multiply by total fragments.**
- [ ] **Step 5: Run config/telemetry/resource tests, then full `pytest -q`; commit.**

### Task 4: Integration validation and training gate

**Files:**
- Test/Create: one focused Scenario V3 integration test file following existing Scenario V2 conventions.
- Modify: documentation only if integration evidence requires it.

**Interfaces:**
- Consumes Tasks 1-3 and produces evidence that V3 is safe to hand to cloud smoke training.

- [ ] **Step 1: Add/execute deterministic 1 MB direct, two-hop, three-hop and packet-loss/retry integration cases on native and Warp CPU.**
- [ ] **Step 2: Verify V2 regression suite and entire test suite pass.**
- [ ] **Step 3: If CUDA is locally available, run V3 CUDA parity/resource smoke; otherwise produce the exact existing Kaggle validation command/config needed for a GPU gate without launching a 25k training job.**
- [ ] **Step 4: Run `git diff --check`, inspect branch diff against `4c75c5c`, and perform final code review.**
- [ ] **Step 5: Commit any integration-only tests/docs and leave the branch ready for an explicitly separated short cloud smoke run.**
