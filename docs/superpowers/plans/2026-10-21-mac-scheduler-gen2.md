# MAC Scheduler Gen2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove a semantics-preserving MAC Scheduler Gen2 that materially accelerates Scenario V3 `warp_packet` collection without changing the scientific trajectory.

**Architecture:** Keep Gen1 as the reference/fallback and add Gen2 as an internal A/B-selectable scheduler. Gen2 first proves event-order equivalence, then conservatively fuses only deterministic event chains whose successor is provably the same event Gen1 would pop next under `(timestamp, priority, insertion_order)`. Every optimization is gated by Gen1↔Gen2 state parity before CUDA performance measurements.

**Tech Stack:** Python 3, NVIDIA Warp, PyTorch, NumPy, pytest/unittest, Kaggle Tesla T4, existing Scenario V3 `warp_packet` simulator.

**Spec:** `docs/superpowers/specs/2026-10-21-mac-scheduler-gen2-design.md`

## Global Constraints

- Scenario V3 science remains unchanged: 6 UAVs, 35 targets, 25 obstacles, 1,800-step mission horizon, seed 44, 1,000,000-byte reports, 3,000,000-byte buffers, 1,024-byte payloads, existing PHY/MAC/energy/reward semantics.
- Gen1 remains the production default and exact fallback until every promotion gate passes.
- No change to MASAC, MATD3, PQL, replay, AMP, `torch.compile`, optimizer equations, reward equations, observation/state semantics, or checkpoint schema.
- Gen2 must preserve RNG draw count/order, heap order, packet/session/frame state, TX/RX timestamps, relay custody, energy, reward, termination, and all future-relevant state for identical inputs.
- No approximate CSMA/CA, report-size reduction, payload-size change, ACK/retry removal, timestep change, or backend simplification.
- Development starts from production SHA `6de210f44c6c7da70c94aaa9bd629be9e45efcd7`; the currently running four Scenario V3 production jobs are not modified or stopped by this plan.
- Performance promotion requires scientific parity, full regression success, CUDA exactness, no new OOM/capacity failures, and either at least 300 sustained joint transitions/s on the representative Scenario V3 collector benchmark or a measured quota-feasibility proof. A 10–20% gain is insufficient.
- Three failed parity-safe optimization hypotheses without sufficient speedup trigger architectural review instead of another speculative patch.

## Review Focus

1. **Equal timestamp/priority competitors:** Gen2 must preserve insertion-order ties exactly; Task 1 and Task 3 pin this with explicit equal-key and same-time contention tests.
2. **RNG around backoff/retry:** fusion must not move or duplicate MT19937 draws; Task 3 compares RNG/device state after fused and fallback paths.
3. **Collision/receiver competition:** Gen2 must not fuse through uncertain overlapping transmitters or receiver FIFO ordering; Task 4 covers hidden sender, equal backoff, overlapping frames, ACK and receive continuations.
4. **Cross-step 1 MB streams and resume:** live sessions/events must survive step and checkpoint boundaries identically; Task 5 covers Scenario V3 1 MB direct/relay/failed-PHY sequences and exact resume.
5. **Fast idle microbenchmark hiding active-stream cost:** performance decisions must use active-streaming as well as low-contention states; Task 6 requires both corpora and rejects microbenchmark-only gains.

---

### Task 1: Centralize Exact Event-Key Ordering

**Files:**
- Modify: `uav_network/communication/gpu/events.py`
- Create: `tests/test_mac_scheduler_gen2.py`

**Interfaces:**
- Consumes: existing event fields `event_time`, `event_priority`, `event_order`, `event_count` and Gen1 `earlier()` semantics.
- Produces: `key_before(time_a: wp.float64, priority_a: int, order_a: wp.int64, time_b: wp.float64, priority_b: int, order_b: wp.int64) -> bool` and `candidate_before_root(s: WorldState, e: int, time: wp.float64, priority: int, order: wp.int64) -> bool`. Gen1 `earlier()` delegates to the same key comparator.

- [ ] **Step 1: Write failing event-key tests**

Add tests that execute `key_before` through a tiny Warp test kernel and assert all ordering dimensions independently: earlier/later timestamp, lower/higher priority at equal time, lower/higher insertion order at equal time+priority, and exact-key equality returning false. Add a heap-root test where `candidate_before_root` returns true only when the candidate key is strictly earlier than the current root, and true when the heap is empty.

- [ ] **Step 2: Run the new tests and verify RED**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py --tb=short
```
Expected: FAIL because `key_before` / `candidate_before_root` do not exist.

- [ ] **Step 3: Implement the shared exact-order helpers**

Implement the two signatures above in `events.py`, then rewrite `earlier()` to call `key_before` rather than duplicate the comparison. `candidate_before_root` is read-only and must not alter `event_sequence`, heap fields, or counts.

- [ ] **Step 4: Verify GREEN and Gen1 regression**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py tests/test_gpu_packets.py --tb=short
```
Expected: all tests pass, including every existing Native↔Warp packet parity test.

- [ ] **Step 5: Commit Task 1**

```bash
git add uav_network/communication/gpu/events.py tests/test_mac_scheduler_gen2.py
git commit -m "refactor(mac): centralize exact event ordering"
```

### Task 2: Add Internal Gen1/Gen2 Selection and a Fallback-Only Gen2 Baseline

**Files:**
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `uav_network/environment/gpu/vector.py`
- Modify: `tests/test_mac_scheduler_gen2.py`
- Modify: `tests/test_gpu_vector.py`

**Interfaces:**
- Consumes: Task 1 ordering helpers and existing `packet_step` event semantics.
- Produces: `WarpVectorEnv(config, device=None, packet_scheduler="gen1")`, accepting only `"gen1"` and `"gen2"`; existing callers remain unchanged. Produces `packet_step_gen2` with the same state/action semantics as `packet_step`. Gen2 is fallback-only in this task: it uses the same Gen1 event execution path and therefore establishes the A/B harness before optimization.

- [ ] **Step 1: Write failing selector and whole-state parity tests**

In `tests/test_mac_scheduler_gen2.py`, instantiate paired CPU `WarpVectorEnv` objects with identical config/seeds, one `gen1` and one `gen2`; apply identical deterministic actions for several steps and compare reward/done plus every array returned by `snapshot_state()["arrays"]` exactly (`rtol=0`, `atol=0`, `equal_nan=True`). In `tests/test_gpu_vector.py`, assert invalid scheduler names raise `ValueError` and the default constructor remains Gen1.

- [ ] **Step 2: Run the selector/parity tests and verify RED**

Run the exact new test nodes. Expected: FAIL because `packet_scheduler` and `packet_step_gen2` are absent.

- [ ] **Step 3: Refactor `packet_step` without semantic changes**

Extract the per-world packet-step body into shared Warp functions so both `packet_step` and `packet_step_gen2` execute the same preamble, Gen1 event loop, and finalization in this task. Preserve the existing `packet_step` kernel signature for current direct tests. Do not modify `event()` branch semantics.

- [ ] **Step 4: Add the internal vector selector**

Extend `WarpVectorEnv.__init__` with `packet_scheduler="gen1"`; store the validated internal mode and select `packet_step` versus `packet_step_gen2` in `step_async`. Do not add a Scenario/config field and do not change `make_vector()` defaults, checkpoint payloads, or `_snapshot_config()`.

- [ ] **Step 5: Verify exact fallback baseline**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py tests/test_gpu_vector.py tests/test_gpu_packets.py tests/test_exact_environment_resume.py --tb=short
```
Expected: all pass; Gen1 and fallback-only Gen2 snapshots are identical.

- [ ] **Step 6: Commit Task 2**

```bash
git add uav_network/communication/gpu/mac.py uav_network/environment/gpu/vector.py tests/test_mac_scheduler_gen2.py tests/test_gpu_vector.py
git commit -m "feat(mac): add exact Gen2 fallback path"
```

### Task 3: Fuse the Single-Successor START→MAC→IDLE Prefix Conservatively

**Files:**
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `uav_network/environment/gpu/vector.py`
- Modify: `tests/test_mac_scheduler_gen2.py`
- Test: `tests/test_gpu_packets.py`

**Interfaces:**
- Consumes: Task 1 `candidate_before_root`, Task 2 Gen2 dispatch.
- Produces: internal `try_macro_event(...) -> bool` in `mac.py`, which may consume the already-popped current event plus one or more proven deterministic successors; `False` means the caller executes existing `event()` with no scientific-state mutation from the failed attempt. Produces passive Gen2 scheduler counters stored outside `WorldState`/snapshots and exposed through `WarpVectorEnv.scheduler_metrics()`.

- [ ] **Step 1: Write RED tests for safe fusion and unsafe fallback**

Add controlled heap states covering: no competitor; a later competitor; an earlier competitor; exact timestamp+priority tie where insertion order forces fallback; and a capacity/error boundary. Assert Gen1↔Gen2 snapshots, event sequences, RNG/MT state, transmission trace, and reward remain exact. Assert scheduler metrics show fused events only in the safe cases and fallback in the unsafe cases.

- [ ] **Step 2: Verify RED**

Run the new tests. Expected: FAIL because no macro event path or scheduler metrics exist.

- [ ] **Step 3: Add passive scheduler counters outside checkpoint state**

Allocate a small Gen2-only Warp counter array owned by `WarpVectorEnv`, not `WorldState`, with counters for attempted chains, completed chains, fused events, competitor fallbacks, unsupported/boundary fallbacks, and maximum chain length. Implement `scheduler_metrics()` as a diagnostic read; it must not participate in snapshot/restore or observations.

- [ ] **Step 4: Implement only the one-successor prefix fusion**

Implement `try_macro_event` for the safe `START -> MAC -> IDLE` prefix. Before consuming each successor inline, compute the would-be insertion order exactly and use Task 1's strict key proof against the current competitor root. Increment `event_sequence` exactly as Gen1 scheduling would. Preserve the existing MT19937 draw point in `MAC`. Stop before any branch with multiple successors or uncertain receiver/airtime interaction. On any failed proof, leave the uncertain successor to normal heap/event execution.

- [ ] **Step 5: Run exact parity regressions**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py tests/test_gpu_packets.py tests/test_gpu_vector.py --tb=short
```
Expected: exact parity suite green; macro counters confirm the path is exercised.

- [ ] **Step 6: Run a bounded local throughput sanity check**

Use a CPU/Warp benchmark with fixed seed/actions only to confirm the fused path is exercised and does not regress catastrophically. Do not use this result for promotion; Task 6 owns T4 performance.

- [ ] **Step 7: Commit Task 3**

```bash
git add uav_network/communication/gpu/mac.py uav_network/environment/gpu/vector.py tests/test_mac_scheduler_gen2.py
git commit -m "perf(mac): fuse exact single-successor prefix"
```

### Task 4: Extend Exact Fusion Through Split Successors, TX/RX, and ACK Boundaries

**Files:**
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `tests/test_mac_scheduler_gen2.py`
- Test: `tests/test_gpu_packets.py`

**Interfaces:**
- Consumes: Task 3 `try_macro_event`, exact ordering proof, scheduler counters.
- Produces: additional conservative macro chains for `IDLE_DONE/LISTEN/BACKOFF_DONE`, `TX_START -> DATA_END`, and receive/ACK continuations only where all newly scheduled successors receive the same sequence/order numbers Gen1 would assign and the chosen inline successor is strictly the next Gen1 event.

- [ ] **Step 1: Write RED parity tests for split-successor ordering**

Add focused cases where an event schedules two successors and verify their insertion orders against Gen1. Include a case where the nominal inline event loses to an existing equal-time competitor and must fall back.

- [ ] **Step 2: Write RED contention/PHY tests**

Pin Gen1↔Gen2 equality for equal-backoff ties, hidden-sender airtime overlap, busy-channel freeze, weak-link retries, ACK timeout/retry, receiver FIFO competition, and cross-step active DATA/ACK state. Assert scheduler counters prove both fusion and fallback paths execute.

- [ ] **Step 3: Verify RED**

Run the new Task 4 tests. Expected: failures because Task 3 deliberately stops before these event families.

- [ ] **Step 4: Implement split-successor and TX/RX/ACK fusion one family at a time**

For each event family: assign all would-be `event_sequence` values in Gen1 order; materialize non-inline successors in the heap exactly as Gen1 would; inline only the single successor that is strictly next by Task 1's comparator; otherwise fallback before uncertain execution. Reuse existing `busy`, `emit`, `evaluate`, `next_receive`, `event`, RNG and energy functions rather than reimplementing PHY/report semantics.

- [ ] **Step 5: Verify after each event-family addition**

After each family, run its focused tests plus `tests/test_gpu_packets.py`. Do not stack another family on a red parity state.

- [ ] **Step 6: Run the complete packet parity suite**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py tests/test_gpu_packets.py tests/test_retry_recovery.py tests/test_propagation.py --tb=short
```
Expected: all pass with no scientific-state divergence.

- [ ] **Step 7: Commit Task 4**

```bash
git add uav_network/communication/gpu/mac.py tests/test_mac_scheduler_gen2.py
git commit -m "perf(mac): fuse exact transmit receive chains"
```

### Task 5: Prove Scenario V3 1 MB, Relay, Energy, and Resume Exactness

**Files:**
- Modify: `tests/test_scenario_v3_integration.py`
- Modify: `tests/test_exact_environment_resume.py`
- Modify: `tests/test_mac_scheduler_gen2.py`
- Test: `tests/test_communication_trace.py`

**Interfaces:**
- Consumes: Gen1/Gen2 selector and Task 4 fusion implementation.
- Produces: a fixed Scenario V3 Gen1↔Gen2 corpus covering 1 MB streaming and a proof that scheduler choice does not require checkpoint schema changes.

- [ ] **Step 1: Write a Gen1↔Gen2 Scenario V3 corpus**

Extend the V3 fixture/helpers so Warp Gen1 and Warp Gen2 can run the exact same positions/actions. Add explicit comparisons for 1 MB direct delivery, two-hop relay, three-hop relay, low-power failure followed by recovery, and a contention action sequence. Compare delivery/known/custody, packet metrics, communication energy, report hops/delay, trace timestamps/order, rewards, and final snapshot arrays.

- [ ] **Step 2: Verify RED where Gen2 coverage is incomplete**

Run the new Gen2 V3 test nodes before any fixes required by the corpus. If they already pass, record that the existing Task 4 implementation covers the corpus and do not invent extra code.

- [ ] **Step 3: Add exact-resume scheduler test**

Run Gen2 into an active 1 MB session, snapshot, restore into another Gen2 vector, continue both with identical actions, and compare every snapshot array exactly. Also assert the serialized snapshot keys/schema remain the same as Gen1 because scheduler diagnostics are external to `WorldState`.

- [ ] **Step 4: Fix only minimized parity failures**

For every divergence, first reduce it to the shortest event/action sequence and add that sequence to `tests/test_mac_scheduler_gen2.py`; then change Gen2. Do not modify Scenario V3 constants or Gen1 semantics to make parity pass.

- [ ] **Step 5: Verify V3 + resume + trace**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_gen2.py tests/test_scenario_v3_integration.py tests/test_exact_environment_resume.py tests/test_communication_trace.py --tb=short
```
Expected: all pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add tests/test_scenario_v3_integration.py tests/test_exact_environment_resume.py tests/test_mac_scheduler_gen2.py
git commit -m "test(mac): prove Gen2 V3 and resume parity"
```

### Task 6: Add Reproducible Gen1/Gen2 T4 Benchmark and CUDA Exactness Gate

**Files:**
- Create: `uav_network/experiments/mac_scheduler_benchmark.py`
- Create: `tests/test_mac_scheduler_benchmark.py`
- Modify: `tests/test_gpu_cuda.py`
- Modify: `tests/test_cuda_exact_resume.py` only if the existing CUDA harness needs a scheduler selector parameter

**Interfaces:**
- Consumes: `WarpVectorEnv(..., packet_scheduler=...)`, `scheduler_metrics()`, Scenario V3 config.
- Produces: `benchmark_mac_scheduler(config, *, num_envs: int, steps: int, scheduler: str, corpus: str) -> dict` where `corpus` is `"low_contention"` or `"active_streaming"`. Report fields include scheduler, commit, device, num_envs, steps, packet-step/collector timing, joint transitions/s, scheduler metrics, GPU memory, simulator error counts, and corpus identity.

- [ ] **Step 1: Write RED benchmark contract tests**

Test deterministic argument validation, required result keys, distinct low-contention/active-streaming corpus labels, and rejection of non-Gen1/Gen2 modes. Use tiny CPU fixtures; do not assert performance in unit tests.

- [ ] **Step 2: Verify RED**

Run `tests/test_mac_scheduler_benchmark.py`; expected FAIL because the benchmark module does not exist.

- [ ] **Step 3: Implement the benchmark harness**

Use fixed seeds and deterministic action sequences. Time the collector with device synchronization around the measured region. Record `packet_step` timing separately using the same instrumentation for Gen1 and Gen2. Keep tracking disabled and do not train a learner in this benchmark.

- [ ] **Step 4: Add CUDA Gen1↔Gen2 exactness test**

On CUDA, run a bounded fixed-seed/action corpus through Gen1 and Gen2 and compare reward/done plus every future-relevant snapshot array exactly. Include an active report session, not only idle/random states. The test must skip cleanly when CUDA is unavailable locally.

- [ ] **Step 5: Run local contract/regression tests**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q tests/test_mac_scheduler_benchmark.py tests/test_mac_scheduler_gen2.py tests/test_gpu_packets.py tests/test_gpu_cuda.py --tb=short
```
Expected locally: CPU tests pass and CUDA-only tests skip if no CUDA.

- [ ] **Step 6: Launch bounded Tesla T4 A/B jobs on a spare Kaggle account**

Run Gen1 and Gen2 from the same candidate commit with the pinned Scenario V3 profile. Measure at minimum: 2,048 envs, low-contention corpus and active-streaming corpus, enough post-JIT steps to exclude first-launch compilation. Do not use accounts 21–24 that own the four production runs.

- [ ] **Step 7: Evaluate the performance gate**

Require all of: CUDA state parity, no simulator errors/OOM, no scientific metric divergence, and sustained representative Gen2 throughput meeting `>=300 joint transitions/s` or a measured quota-feasibility proof. If Gen2 is only 10–20% faster, record the result as insufficient and do not prepare a production restart.

- [ ] **Step 8: Commit Task 6**

```bash
git add uav_network/experiments/mac_scheduler_benchmark.py tests/test_mac_scheduler_benchmark.py tests/test_gpu_cuda.py tests/test_cuda_exact_resume.py
git commit -m "bench(mac): gate Gen2 on CUDA parity and throughput"
```

### Task 7: Full Verification, Review, and Promotion Decision

**Files:**
- Modify only files required to fix issues found by review/verification; no unrelated refactors.
- Review: all files changed since `14998dbb00b39dc8396896bf224a568e9e01b315`.

**Interfaces:**
- Consumes: all previous tasks and T4 benchmark evidence.
- Produces: a clean reviewed branch plus one explicit decision: `promote_candidate`, `keep_gen1_insufficient_speedup`, or `keep_gen1_parity_failure`.

- [ ] **Step 1: Run the complete local test suite**

Run:
```bash
PYTHONPATH=.:tests /home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014/.venv/bin/python -m pytest -q
```
Expected: no failures. Report exact pass/skip/subtest counts.

- [ ] **Step 2: Run static/diff hygiene checks**

Run `git diff --check`, inspect `git status --short`, and audit the branch diff for accidental Scenario V3/config/reward/algorithm/checkpoint-schema changes and secret/output leakage.

- [ ] **Step 3: Request code review**

Review the full branch with special focus on event ordering, mutation-before-proof, RNG sequence, split-successor insertion order, receiver/collision edge cases, and passive scheduler diagnostics. Fix only verified issues and rerun affected parity suites after each fix.

- [ ] **Step 4: Re-run authoritative T4 evidence on the final candidate SHA**

Repeat the CUDA exactness and representative low-contention + active-streaming benchmarks on the final commit, not an earlier working tree. Record the exact SHA, T4 model, timing, throughput, memory, fusion/fallback metrics, and error counts.

- [ ] **Step 5: Apply the promotion gates mechanically**

- `promote_candidate`: every scientific/CUDA/full-suite gate passes and the performance/quota gate is satisfied.
- `keep_gen1_insufficient_speedup`: parity passes but performance gate fails.
- `keep_gen1_parity_failure`: any unresolved exactness gate fails.

Do not restart the four production lineages as part of this task; a restart/continuation migration requires a separate explicit rollout decision after presenting the measured result.

- [ ] **Step 6: Commit final verification fixes, if any**

Use a focused commit message describing only the verified fixes. If no source changes were needed, do not create an empty commit.

- [ ] **Step 7: Push the Gen2 branch only after the final SHA is verified**

Push `perf/mac-scheduler-gen2-20261021`, then confirm the remote branch resolves to the same full SHA. Do not merge into production/main automatically.
