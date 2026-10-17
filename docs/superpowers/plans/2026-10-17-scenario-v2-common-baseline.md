# Scenario V2 Common Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Scenario V2 as a shared, backward-compatible environment formulation for standard MASAC and MATD3, making transmit-power control and safety-shield dependence scientifically meaningful without algorithm-specific changes.

**Architecture:** Keep V1 defaults unchanged. Add pure link-budget/power helpers shared by native and Warp paths, conditional link-state observations and power-limited admission, a normalized communication-efficiency reward term, and a shield-correction reward/metric. Mirror every scientific behavior in native and Warp implementations and expose bounded telemetry/evaluation metrics. Do not modify either learner implementation.

**Tech Stack:** Python 3, NumPy, PyTorch, NVIDIA Warp, UavNetSim, pytest/unittest, YAML, W&B telemetry.

**Spec:** `docs/superpowers/specs/2026-10-17-scenario-v2-common-baseline-design.md`

## Global Constraints

- `uav_network/algorithms/masac.py` must remain byte-for-byte unchanged from SHA-256 `e6f38a562c7ae4a1bccaa5324689876450f4a75b4f4eb5f63e36c1b869660419`.
- `uav_network/algorithms/matd3.py` must remain byte-for-byte unchanged from SHA-256 `fee6b3692d0ac6115c340ab397bb8098c811a97331e0bca2cb0f8aab729ecf4b`.
- `configs/comparison-25k-diagnostics.yaml` is immutable V1 evidence and must not be edited.
- V1 defaults stay: linear TX mapping, no power-limited admission, no link-state features, no normalized communication-efficiency reward, no shield-correction reward.
- V2 uses `tx_power_min_w=0.001`, `tx_power_max_w=0.4`, `tx_power_mapping=log`, existing 500 m peer/GCS protocol radius, `communication_efficiency=1.0`, `shield_correction=0.1`.
- V2 core keeps report size 1024 B, buffer 3072 B, TTL 300 s, destructive custody, rediscovery recovery and max retries 5.
- No direct relay reward, no MATD3-specific or MASAC-specific modification, no APF replacement, no buffer/TTL change in V2 core.
- Native and Warp must implement identical scientific semantics within numeric tolerance.
- Replay continues storing commanded policy actions.
- Authoritative active-suite command is `python3 -m pytest -q tests`; never use bare root `pytest` because archived tests under `outputs/` collide by basename.

## Review Focus

- V1 checkpoint/observation compatibility when all V2 fields remain at defaults; pin with exact observation-dimension and action-to-watts tests in Task 1/2.
- Links whose nominal required power is above `Pmax`; pin mask behavior without falsely guaranteeing actual delivery in Task 2.
- GCS and peer feature ordering/mask-tail location; pin exact column layout in native/Warp parity tests in Task 2.
- Shield correction when no active UAV exists or when commands are already safe; pin zero/finiteness behavior in Task 4.
- Pooled ratios and power-margin statistics; aggregate raw numerators/denominators rather than averaging episode ratios in Task 5.

---

### Task 1: Shared V2 configuration, power mapping and analytic link budget

**Files:**
- Create: `uav_network/communication/link_budget.py`
- Modify: `uav_network/configuration.py`
- Modify: `uav_network/communication/backend.py`
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `uav_network/environment/gpu/state.py`
- Test: `tests/test_channel_models.py`
- Test: `tests/test_communication.py`
- Test: `tests/test_gpu_packets.py`

**Interfaces:**
- Produces `power_from_action(action_value: float, network_config) -> float` for native callers.
- Produces `required_power_w(distance_m: float, frequency_hz: float, bandwidth_hz: float, noise_density_dbm_hz: float, noise_figure_db: float, sinr_threshold_db: float, nlos_model: str, line_of_sight: bool, obstacle_extra_loss_db: float = 0.0) -> float`.
- Produces `power_fraction(power_w: float, network_config) -> float` in the configured mapping coordinate, clipped to `[0,1]`.
- Adds `NetworkConfig.tx_power_mapping: str = "linear"`, `NetworkConfig.power_limited_admission: bool = False`, `NetworkConfig.link_state_observation: bool = False`.
- Adds `RewardConfig.communication_efficiency: float = 0.0`, `RewardConfig.shield_correction: float = 0.0`.
- Warp `WorldParams` gains mapping/reward flags/values required by later tasks.

- [ ] **Step 1: Write failing configuration and link-budget tests**

Add tests asserting:
- default linear mapping still maps `-1 -> 0.1 W`, `0 -> 0.25 W`, `1 -> 0.4 W`;
- log mapping with `Pmin=0.001`, `Pmax=0.4` maps endpoints exactly and midpoint to `sqrt(0.001*0.4)`;
- `power_fraction(power_from_action(a)) == (a+1)/2` for representative actions under log mapping;
- analytic free-space required power at 500 m is approximately `0.00442144 W` for 2.4 GHz, 22 MHz, NF 7 dB, threshold 6 dB;
- invalid mapping strings and non-positive power values fail validation;
- new reward coefficients must be finite and non-negative.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python3 -m pytest -q tests/test_channel_models.py tests/test_communication.py -k 'power or mapping or required'`
Expected: new tests fail because the fields/helpers do not exist.

- [ ] **Step 3: Implement pure helpers and backward-compatible config validation**

Implement only analytic no-interference A2A required-power math in `link_budget.py`; do not call or mutate the live native PHY cache. Preserve free-space LOS exponent 2 and configured urban/suburban NLOS parameters already pinned by UavNetSim.

- [ ] **Step 4: Route native and Warp action-to-watts through the same semantics**

Native `backend.py` calls `power_from_action`. Warp `mac.py` implements the algebraically equivalent branch selected by a `WorldParams` flag. Linear remains default; log uses `Pmin * (Pmax/Pmin)^u`.

- [ ] **Step 5: Run focused native/Warp power tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_channel_models.py tests/test_communication.py tests/test_gpu_packets.py`
Expected: PASS/optional CUDA skips only.

- [ ] **Step 6: Commit**

Commit message: `feat: add shared scenario-v2 power semantics`

---

### Task 2: Power-limited admission and shared link-state observations

**Files:**
- Modify: `uav_network/environment/observations.py`
- Modify: `uav_network/communication/backend.py`
- Modify: `uav_network/environment/gpu/observations.py`
- Modify: `uav_network/communication/gpu/mac.py` or the existing Warp destination-admission helper used by it
- Modify: `uav_network/environment/gpu/state.py`
- Test: `tests/test_receiver_admission.py`
- Test: `tests/test_environment.py`
- Test: `tests/test_gpu_vector.py`
- Test: `tests/test_gpu_physics.py`

**Interfaces:**
- Consumes Task 1 `required_power_w`, `power_fraction`, mapping fields and Warp params.
- Produces conditional per-destination feature triplets in ordered peers then GCS: `[required_power_fraction, gcs_progress, receiver_free_buffer_fraction]`.
- Preserves the final `num_uavs + 1` observation entries as the destination mask.
- Produces a shared native helper for nominal link features/admission that CPU tests can call without touching live packet state.

- [ ] **Step 1: Write failing observation-layout and admission tests**

Add tests asserting:
- V1 `observation_dim()` is unchanged when `link_state_observation=False`;
- V2 dimension increases by `3 * num_uavs` (one triplet for every non-silent destination: `num_uavs-1` peers + GCS);
- feature order is peers in existing destination order, then GCS, then the unchanged mask tail;
- a peer/GCS nominally requiring `>Pmax` is masked only when `power_limited_admission=True`;
- disabling power-limited admission reproduces the old range/admission mask;
- `gcs_progress` is positive for a receiver closer to GCS, negative for farther, and GCS destination uses zero destination distance;
- peer free-buffer fraction matches remaining report slots/bytes and GCS is `1`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python3 -m pytest -q tests/test_receiver_admission.py tests/test_environment.py -k 'admission or observation or link_state or progress or buffer'`
Expected: FAIL on new V2 assertions.

- [ ] **Step 3: Implement native conditional features and admission**

Use deterministic geometry plus the pure Task 1 link-budget helper. LOS/NLOS classification for the nominal feature must match the existing obstacle geometry semantics without querying live PHY state. Required-power feature is clipped through `power_fraction`; links above Pmax can still expose a clipped feature but are masked when the V2 admission flag is enabled.

- [ ] **Step 4: Implement equivalent Warp feature/admission math**

Reuse existing `los()`/path-gain constants or factor a Warp-compatible helper so observation/admission and packet PHY use the same configured model parameters. Do not move the mask from the observation tail.

- [ ] **Step 5: Add fixed-geometry native/Warp parity tests**

Assert identical masks and feature triplets for LOS, NLOS, closer-peer, farther-peer and partially full-buffer cases within `atol<=1e-6` for float observation entries.

- [ ] **Step 6: Run focused parity tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_receiver_admission.py tests/test_environment.py tests/test_gpu_vector.py tests/test_gpu_physics.py`
Expected: PASS/optional GPU-specific skips only.

- [ ] **Step 7: Commit**

Commit message: `feat: expose shared scenario-v2 link state`

---

### Task 3: Make low/high TX power produce the intended real PHY behavior

**Files:**
- Modify only if needed for consistency: `uav_network/communication/backend.py`
- Modify only if needed for consistency: `uav_network/communication/gpu/phy.py`
- Test: `tests/test_communication.py`
- Test: `tests/test_gpu_physics.py`
- Test: `tests/test_network_telemetry.py`

**Interfaces:**
- Consumes Task 1 power mapping and Task 2 admission.
- No new public algorithm interface.
- Confirms that admission is nominal feasibility only; actual chosen power, interference, ACK/retry and TTL remain real packet behavior.

- [ ] **Step 1: Write failing deterministic micro-link tests**

Construct V2 fixed geometries with no interference and assert:
- a near LOS link succeeds at a low chosen power;
- a farther LOS link inside 500 m fails at the same low power but succeeds at a higher chosen power;
- choosing below the advertised required power is not masked if Pmax makes the link feasible, and produces real PHY failure/retry;
- an overlapping interferer can cause failure even when selected power exceeds nominal required power;
- source custody remains after failed report delivery attempts.

- [ ] **Step 2: Run focused tests and verify RED where current semantics diverge**

Run: `python3 -m pytest -q tests/test_communication.py tests/test_gpu_physics.py tests/test_network_telemetry.py -k 'low_power or high_power or interference or nominal'`
Expected: at least the new V2 scenarios fail before final consistency wiring.

- [ ] **Step 3: Apply the minimal consistency fixes only**

Do not alter retry count, ACK semantics, TTL, interference model or custody transfer. Fix only discrepancies between native/Warp V2 selected-power semantics exposed by the tests.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_communication.py tests/test_gpu_physics.py tests/test_network_telemetry.py`
Expected: PASS/optional GPU-specific skips only.

- [ ] **Step 5: Commit**

Commit message: `test: verify scenario-v2 power-sensitive links`

---

### Task 4: Normalized communication-efficiency reward and shield-correction learning signal

**Files:**
- Modify: `uav_network/communication/link_budget.py`
- Modify: `uav_network/environment/avoidance.py`
- Modify: `uav_network/environment/motion.py`
- Modify: `uav_network/environment/energy.py`
- Modify: `uav_network/environment/env.py`
- Modify: `uav_network/environment/gpu/state.py`
- Modify: `uav_network/environment/gpu/motion.py`
- Modify: `uav_network/communication/gpu/reports.py`
- Modify: `uav_network/communication/gpu/metrics.py`
- Test: `tests/test_energy.py`
- Test: `tests/test_avoidance.py`
- Test: `tests/test_boundary_escape.py`
- Test: `tests/test_environment.py`
- Test: `tests/test_gpu_physics.py`

**Interfaces:**
- Produces `report_reference_tx_energy_j(network_config) -> float`: one complete DATA report sent once at Pmax including existing DATA header bits and fragmentation, excluding ACKs/retries.
- Motion diagnostics gain accumulated/step shield correction magnitude computed as mean over active UAVs of `||v_safe-v_nominal||/max_speed`.
- Reward components gain `communication_efficiency` and `shield_correction` while keeping all existing physical battery accounting untouched.

- [ ] **Step 1: Write failing reward-reference and shield-correction tests**

Assert:
- `report_reference_tx_energy_j` exactly matches configured fragments, DATA header bits, bit rate and Pmax;
- with V2 weight zero, reward is unchanged from V1 for the same deterministic transition;
- with `communication_efficiency=1.0`, exactly one full-report-equivalent of additional UAV RF energy contributes `-1.0` normalized reward cost;
- retries/ACK UAV RF energy increase the normalized cost because the numerator uses actual delta UAV RF energy;
- physical `battery_j`, `uav_communication_energy_j`, and propagation-energy accounting do not change when only the reward coefficient changes;
- shield correction is exactly zero for safe pass-through command;
- shield correction is positive and finite when a boundary/obstacle command is modified;
- no-active-UAV case yields zero, not NaN;
- replay/action interfaces remain commanded-action based.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python3 -m pytest -q tests/test_energy.py tests/test_avoidance.py tests/test_boundary_escape.py tests/test_environment.py`
Expected: new reward/diagnostic assertions fail.

- [ ] **Step 3: Implement native reward terms and diagnostics**

Compute correction from the nominal reachable velocity before the shield and the shield-selected safe velocity. Keep hard safety behavior unchanged. Add the two reward components only when their coefficients are non-zero; defaults must reproduce V1.

- [ ] **Step 4: Mirror reward/correction semantics in Warp**

Extend fixed-size state/reward component arrays deliberately and update extraction order everywhere that consumes them. Preserve existing component names/order for old entries and append new entries rather than reinterpreting old indices.

- [ ] **Step 5: Run native/Warp reward parity tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_energy.py tests/test_avoidance.py tests/test_boundary_escape.py tests/test_environment.py tests/test_gpu_physics.py`
Expected: PASS/optional GPU-specific skips only.

- [ ] **Step 6: Commit**

Commit message: `feat: add shared communication and shield costs`

---

### Task 5: Scientific telemetry for relay progress, nominal power and shield correction

**Files:**
- Modify: `uav_network/communication/diagnostics.py`
- Modify: `uav_network/communication/telemetry.py`
- Modify: `uav_network/communication/backend.py`
- Modify: `uav_network/communication/reports.py`
- Modify: `uav_network/communication/gpu/diagnostics.py`
- Modify: `uav_network/communication/gpu/phy.py`
- Modify: `uav_network/communication/gpu/reports.py`
- Modify: `uav_network/communication/gpu/metrics.py`
- Modify: `uav_network/experiments/telemetry.py`
- Modify: `uav_network/experiments/tracking.py`
- Modify: `uav_network/experiments/comparison_metrics.py`
- Test: `tests/test_network_telemetry.py`
- Test: `tests/test_diagnostics_v1.py`
- Test: `tests/test_tracking.py`
- Test: `tests/test_comparison_metrics.py`

**Interfaces:**
- Attempt diagnostics add nominal required TX power and actual-vs-required margin using raw count/sum/min/max storage.
- Report diagnostics add peer custody transfers toward-GCS, away-from-GCS, and total/sum GCS-distance improvement.
- Episode/cohort telemetry exposes normalized communication-efficiency reward cost and accumulated shield correction.
- `OutcomePool` pools raw counts/sums, never averages per-episode ratios for the new fractions.

- [ ] **Step 1: Write failing telemetry tests**

Assert:
- an attempted link records required-power count/sum and actual/required margin;
- attempts below nominal requirement increment a raw counter;
- a successful peer custody transfer closer to GCS increments toward counter and positive distance-improvement sum; farther increments away counter with negative contribution retained in the sum;
- GCS transfers are not counted as peer relay-progress transfers;
- pooled below-required fraction and toward-relay fraction are formed from total numerators/denominators;
- existing diagnostics-v1 keys remain present and unchanged;
- shield correction and normalized communication cost appear as bounded scalar W&B metrics.

- [ ] **Step 2: Run focused telemetry tests and verify RED**

Run: `python3 -m pytest -q tests/test_network_telemetry.py tests/test_diagnostics_v1.py tests/test_tracking.py tests/test_comparison_metrics.py`
Expected: new keys/aggregation tests fail.

- [ ] **Step 3: Implement passive native telemetry**

Calculate required power from already-known geometry/config at the actual attempt; do not draw RNG or issue a second live PHY query. Relay-progress counters update only on successful whole-report custody transfer.

- [ ] **Step 4: Implement equivalent Warp telemetry**

Use the cached/analytic link information already available in the packet/PHY path and append fixed-size counters/stats consistently with native names.

- [ ] **Step 5: Extend comparison metrics without adding a composite winner score**

Add scientifically interpretable V2 metrics with units/direction where appropriate. Keep algorithm ranking as per-metric paired comparison.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_network_telemetry.py tests/test_diagnostics_v1.py tests/test_tracking.py tests/test_comparison_metrics.py`
Expected: PASS.

- [ ] **Step 7: Commit**

Commit message: `feat: log scenario-v2 routing and power diagnostics`

---

### Task 6: Freeze Scenario V2 config, backward compatibility and end-to-end parity

**Files:**
- Create: `configs/comparison-25k-scenario-v2.yaml`
- Modify: `tests/test_diagnostics_v1.py`
- Modify: `tests/test_gpu_comparison.py`
- Modify: `tests/test_comparison_evaluation.py`
- Modify: `tests/test_project_structure.py`
- Modify if needed: `README.md` or `docs/diagnostics-v1.md` only to point to the new V2 profile without rewriting V1 evidence

**Interfaces:**
- V2 profile derives scientific conditions from diagnostics-v1 and explicitly sets only V2 fields plus `tx_power_min_w=0.001`.
- Evaluation protocol scenario hash must differ between V1 and V2 because network/reward/observation-relevant config differs.
- Both algorithms consume the exact same V2 scenario config except `runtime.algorithm` and execution-flow BASE/PQL selection.

- [ ] **Step 1: Write failing profile/invariant tests**

Assert:
- `configs/comparison-25k-diagnostics.yaml` file hash/content is unchanged from branch base;
- V2 profile has 35 targets, 25 obstacles, max_steps 1800, 6 UAVs, same buffer/TTL/retries/range, and exact V2 power/reward flags;
- loading V1 yields the old observation dimension and linear power semantics;
- loading V2 yields the expanded observation dimension/log power semantics;
- V1 and V2 evaluation scenario hashes differ;
- MASAC and MATD3 source SHA-256 values exactly match Global Constraints.

- [ ] **Step 2: Run profile tests and verify RED**

Run: `python3 -m pytest -q tests/test_diagnostics_v1.py tests/test_comparison_evaluation.py tests/test_project_structure.py`
Expected: V2 profile assertions fail because the profile does not exist yet.

- [ ] **Step 3: Create the V2 profile without editing V1**

Copy diagnostics-v1 scientific/training conditions, change only explicit Scenario V2 fields required by the spec, give V2 a distinct output/group/tag naming namespace, and keep the same final 30-seed evaluation bank.

- [ ] **Step 4: Run deterministic native/Warp end-to-end smoke for both algorithms**

Use tiny CPU/Warp configurations only to verify pipeline compatibility and observation/checkpoint construction for MASAC and MATD3. This is not a learning-quality comparison and must not tune either learner.

- [ ] **Step 5: Run authoritative full suite**

Run: `python3 -m pytest -q tests`
Expected: zero failures; CUDA-dependent tests may skip according to hardware.

- [ ] **Step 6: Verify algorithm/V1 invariants explicitly**

Run:
`sha256sum uav_network/algorithms/masac.py uav_network/algorithms/matd3.py configs/comparison-25k-diagnostics.yaml`
Compare algorithm hashes to Global Constraints and V1 config hash to the recorded pre-implementation base hash in the SDD ledger.

- [ ] **Step 7: Commit**

Commit message: `feat: freeze shared scenario-v2 comparison profile`

---

### Task 7: Final scale smoke, documentation and branch review

**Files:**
- Create: `docs/scenario-v2-validation-20261017.md`
- Create: `docs/benchmarks/scenario-v2-validation-20261017.json`
- Modify only if validation finds a real defect: files owned by Tasks 1-6, with a new failing regression test first

**Interfaces:**
- Consumes frozen V2 config/commit from Task 6.
- Produces evidence required before any long Kaggle training launch.

- [ ] **Step 1: Run deterministic physics/safety scale smoke**

Record fixed-geometry results showing low/high power threshold behavior, admission, interference failure, native/Warp feature parity, normalized communication cost scale, and shield-correction scale.

- [ ] **Step 2: Run matched tiny MASAC/MATD3 BASE/PQL pipeline smoke**

Use the same tiny V2 scientific config and seed. Verify both learners construct, collect, update, checkpoint/evaluate and emit the new telemetry. Do not interpret which tiny smoke return is higher.

- [ ] **Step 3: Write validation evidence**

Document exact commit, commands, pass/fail counts, V1 invariant hashes, power/link-budget examples, V2 observation dimension, and smoke metric ranges. State explicitly that long-run ranking has not yet been measured.

- [ ] **Step 4: Run authoritative full suite again after documentation/fixes**

Run: `python3 -m pytest -q tests`
Expected: zero failures.

- [ ] **Step 5: Whole-branch review**

Review against the spec with special attention to: no algorithm-specific edits, no V1 semantic drift, no direct relay reward, native/Warp parity, truthful battery physics, and raw-sum telemetry aggregation.

- [ ] **Step 6: Commit final evidence/fixes**

Commit message: `docs: validate shared scenario-v2 baseline`

