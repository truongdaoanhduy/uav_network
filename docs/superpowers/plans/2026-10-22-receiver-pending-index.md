# Receiver Pending Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental `rxindex` scheduler that replaces Gen1's full `frame_capacity` scan in receiver pending selection with an exact per-receiver heap while preserving UavNetSim-derived canonical behavior bit-for-bit.

**Architecture:** Keep Gen1 unchanged as oracle/default. Allocate receiver-index arrays outside `WorldState`; `rxindex` specializes only `DELIVER`, `RECEIVE`, and `RECEIVE_DONE`, while every other event uses the existing Gen1 `event()` path. The heap key is exactly `(tx_receive_order, frame_index)`, snapshot v1 remains unchanged, and restore rebuilds the heap deterministically from canonical frame state.

**Tech Stack:** Python 3, NVIDIA Warp 1.18.0, PyTorch, unittest/pytest, Tesla T4 Kaggle validation.

**Spec:** `docs/superpowers/specs/2026-10-22-receiver-pending-index-design.md`

## Global Constraints

- Gen1 remains the production default and its `event()`/`next_receive()` code path must stay semantically unchanged.
- No Task-5 edits to `phy.py`, `reports.py`, `streaming.py`, `rng.py`, reward code, MASAC, MATD3, or PQL.
- Preserve CSMA/CA, DIFS/SIFS, CCA, backoff, DATA/ACK timing, retries, propagation, path loss, LoS/NLoS, SINR/interference/collision, TX/RX energy, RNG, report custody, relay hops, TTL, and event ordering exactly.
- Receiver heap key is exactly `(tx_receive_order, frame_index)`; smaller frame index wins equal-order ties.
- Derived receiver-index arrays must not be stored in `WorldState` or snapshot version 1.
- `rxindex` is experimental and must not become the default in this plan.
- No new dependency. Use Warp arrays/functions already available in Warp 1.18.0.
- Do not optimize `emit()` frame recycling/free-slot scans in Task 5.
- Any canonical-state divergence blocks CUDA/performance work.

## Review Focus

- Equal `tx_receive_order` values must choose the smaller frame index exactly as Gen1's ascending scan does; Task 1 pins this with a unit test.
- A pending frame with `tx_active == 2` must remain eligible after PHY retention removes it from `frame_order`; Task 1 rebuild tests pin this.
- Partial reset must clear only selected environments' derived heaps; Task 3 pins untouched-row continuity.
- Restore must rebuild pending frames without mutating canonical arrays or scheduling duplicate `RECEIVE` events; Task 3 pins snapshot/continue parity.
- Receiver busy/queued and invalid TTL/inactive paths must pop exactly one pending frame at the same points Gen1 calls `next_receive()`; Task 2/4 parity tests pin these transitions.

---

### Task 1: Exact Receiver Heap Primitives

**Files:**
- Create: `uav_network/communication/gpu/receiver_index.py`
- Create: `tests/test_receiver_index.py`

**Interfaces:**
- Consumes: `WorldState`, `WorldParams`, existing `events.push()` semantics.
- Produces:
  - `receiver_before(s: WorldState, e: int, left: int, right: int) -> bool`
  - `receiver_push(s: WorldState, p: WorldParams, e: int, receiver: int, frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32), frame: int)`
  - `receiver_pop_min(s: WorldState, p: WorldParams, e: int, receiver: int, frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32)) -> int`
  - `next_receive_indexed(s: WorldState, p: WorldParams, e: int, receiver: int, now: wp.float64, receive_kind: int, frames: ..., sizes: ...)`
  - `reset_receiver_index(p: WorldParams, mask: wp.array(dtype=wp.int32), frames: ..., sizes: ...)` kernel
  - `rebuild_receiver_index(s: WorldState, p: WorldParams, frames: ..., sizes: ...)` kernel

- [ ] **Step 1: Write failing heap-order tests**

Add CPU Warp tests that seed canonical `tx_receive_order` and assert:

```python
# lower order wins
orders = {3: 20, 5: 10}
assert pop_sequence == [5, 3]

# equal order uses smaller frame index
orders = {7: 12, 2: 12}
assert pop_sequence == [2, 7]
```

Also verify two receivers keep independent heaps and push/pop leaves `sizes` exact.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q tests/test_receiver_index.py -k 'heap or tie or receiver'
```

Expected: FAIL because `receiver_index.py`/heap functions do not exist.

- [ ] **Step 3: Implement minimal heap primitives**

Create `receiver_index.py` using only fixed Warp arrays. Comparator reads canonical `s.tx_receive_order` and uses frame index as the second key. Do not copy order values into derived arrays and do not mutate any canonical array in `receiver_push()`/`receiver_pop_min()`.

- [ ] **Step 4: Write failing rebuild/reset tests**

Construct canonical frame states and assert rebuild:

- includes `tx_active != 0`, matching receiver, `tx_receive == 1`;
- excludes `tx_receive` 0, 2, 3;
- includes `tx_active == 2` pending frames;
- leaves every canonical array byte-identical;
- masked reset clears selected `sizes` rows only.

- [ ] **Step 5: Implement reset/rebuild kernels and indexed selection**

`rebuild_receiver_index` clears sizes then scans canonical frame slots in ascending frame index and pushes pending frames. `next_receive_indexed` does nothing when `node_receiving != 0`; otherwise pop min, then perform the exact Gen1 canonical updates and `events.push(... RECEIVE ..., priority=0)` using the `receive_kind` argument.

- [ ] **Step 6: Run Task-1 tests**

Run:

```bash
pytest -q tests/test_receiver_index.py
```

Expected: all pass, no simulator errors.

- [ ] **Step 7: Commit Task 1**

```bash
git add uav_network/communication/gpu/receiver_index.py tests/test_receiver_index.py
git commit -m "feat(net): add exact receiver pending heap"
```

---

### Task 2: `rxindex` Event Path and Vector Mode

**Files:**
- Modify: `uav_network/communication/gpu/mac.py`
- Modify: `uav_network/environment/gpu/vector.py`
- Modify: `tests/test_receiver_index.py`
- Modify: `tests/test_gpu_vector.py`

**Interfaces:**
- Consumes: Task-1 receiver heap functions.
- Produces:
  - `event_rxindex(s, p, e, kind, arg, generation, now, frames, sizes)`
  - `run_event_loop_rxindex(s, p, e, end, frames, sizes)`
  - `packet_step_world_rxindex(s, p, continuous, destination, e, frames, sizes)`
  - `packet_step_rxindex(...)` Warp kernel
  - `WarpVectorEnv(..., packet_scheduler="rxindex")`

- [ ] **Step 1: Write failing vector-mode test**

Update scheduler validation to accept exactly `{"gen1", "gen2", "rxindex"}` while asserting the default remains `gen1` and unknown modes still fail.

Run:

```bash
pytest -q tests/test_gpu_vector.py::GPUVectorTests::test_warp_packet_scheduler_defaults_to_gen1_and_rejects_unknown_mode
```

Expected: FAIL because `rxindex` is not accepted yet.

- [ ] **Step 2: Write failing receive-path parity tests**

Add focused Gen1↔rxindex tests for:

- one successful direct delivery;
- two pending frames with equal `tx_receive_order`, asserting the smaller frame index is serviced first;
- receiver already busy with queued pending frames;
- inactive receiver / TTL rejection selecting the next pending frame.

At each completed step compare all canonical `WorldState` arrays exactly; derived arrays are inspected separately only for heap consistency.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_receiver_index.py tests/test_gpu_vector.py::GPUVectorTests::test_warp_packet_scheduler_defaults_to_gen1_and_rejects_unknown_mode
```

Expected: FAIL because `rxindex` packet dispatch does not exist.

- [ ] **Step 4: Implement receive-only event specialization**

In `mac.py`, leave `event()` and `next_receive()` unchanged. Add `event_rxindex` that duplicates only the canonical branches for `DELIVER`, `RECEIVE`, and `RECEIVE_DONE`, substituting `receiver_push()`/`next_receive_indexed()` where Gen1 calls `next_receive()`. Delegate every other kind directly to existing `event()`.

Add `run_event_loop_rxindex` by reusing the Gen1 heap/`handle_fast_root()` path, then add world/kernel wrappers.

- [ ] **Step 5: Wire experimental vector mode**

In `WarpVectorEnv`:

- allocate `self._receiver_frames = wp.zeros((num_envs, p.n+1, p.frame_capacity), wp.int32, ...)` only for `rxindex`;
- allocate `self._receiver_sizes = wp.zeros((num_envs, p.n+1), wp.int32, ...)` only for `rxindex`;
- launch `packet_step_rxindex` only in `rxindex` mode;
- keep `_scheduler_counters` Gen2-only;
- include receiver arrays in `allocated_bytes` only when allocated;
- keep snapshot enumeration unchanged (`vars(self.s)` only).

- [ ] **Step 6: Run Task-2 tests**

Run:

```bash
pytest -q tests/test_receiver_index.py tests/test_gpu_vector.py
```

Expected: pass with Gen1 still default and unchanged.

- [ ] **Step 7: Commit Task 2**

```bash
git add uav_network/communication/gpu/mac.py uav_network/environment/gpu/vector.py tests/test_receiver_index.py tests/test_gpu_vector.py
git commit -m "feat(net): add experimental rxindex scheduler"
```

---

### Task 3: Partial Reset and Exact Snapshot Restore

**Files:**
- Modify: `uav_network/environment/gpu/vector.py`
- Modify: `tests/test_receiver_index.py`
- Modify: `tests/test_exact_environment_resume.py`

**Interfaces:**
- Consumes: Task-1 reset/rebuild kernels and Task-2 vector arrays.
- Produces: reset/restore lifecycle for derived receiver index without snapshot-schema changes.

- [ ] **Step 1: Write failing partial-reset test**

Use at least two worlds with pending receive state, reset only one world, then assert:

- reset world's receiver heap sizes are zero;
- untouched world's heap arrays/sizes are unchanged;
- canonical untouched world remains exact.

- [ ] **Step 2: Write failing snapshot-schema/rebuild tests**

Assert:

```python
rx = WarpVectorEnv(c, packet_scheduler="rxindex")
snapshot = rx.snapshot_state()
assert snapshot["arrays"].keys() == gen1.snapshot_state()["arrays"].keys()
assert "receiver_frames" not in snapshot["arrays"]
assert "receiver_sizes" not in snapshot["arrays"]
```

Then restore a Gen1 snapshot containing live packet/session state into `rxindex`, continue identical actions, and compare all canonical arrays against Gen1.

Also test `rxindex -> snapshot -> fresh rxindex restore -> continue` against uninterrupted `rxindex`.

- [ ] **Step 3: Run reset/restore tests and verify RED**

Run:

```bash
pytest -q tests/test_receiver_index.py tests/test_exact_environment_resume.py::ExactWorldTests::test_split_world_preserves_live_packet_sessions_and_mac_rng
```

Expected: FAIL because reset/restore do not maintain derived state yet.

- [ ] **Step 4: Wire reset and restore rebuild**

After `reset_network` in `WarpVectorEnv.reset()`, launch `reset_receiver_index` with the same reset mask when mode is `rxindex`.

After canonical copies in `restore_state()`, launch `rebuild_receiver_index` for `rxindex` before returning restored buffers. The rebuild must not push canonical events or alter canonical arrays.

- [ ] **Step 5: Run Task-3 tests**

Run:

```bash
pytest -q tests/test_receiver_index.py tests/test_exact_environment_resume.py
```

Expected: all pass; snapshot version and array keys unchanged.

- [ ] **Step 6: Commit Task 3**

```bash
git add uav_network/environment/gpu/vector.py tests/test_receiver_index.py tests/test_exact_environment_resume.py
git commit -m "test(net): preserve rxindex reset and resume exactness"
```

---

### Task 4: Native UavNetSim and Scenario-V3 Parity Corpus

**Files:**
- Modify: `tests/test_gpu_packets.py`
- Modify: `tests/test_scenario_v3_integration.py`

**Interfaces:**
- Consumes: `rxindex` mode from Tasks 1–3.
- Produces: authoritative semantic evidence across native oracle, Gen1, and `rxindex`.

- [ ] **Step 1: Extend packet-test harness to launch rxindex**

In test-only helpers, allocate receiver arrays and launch `packet_step_rxindex` when `scheduler="rxindex"`. Do not change native oracle behavior or existing Gen1 expectations.

- [ ] **Step 2: Add Gen1↔rxindex canonical comparison to existing oracle cases**

Run rxindex through the existing cases covering at minimum:

- direct DATA/ACK timing;
- weak-link power/retries;
- busy-channel freeze/two senders;
- polygon NLoS/retries;
- equal-backoff collision/retry order;
- hidden-sender airtime interference;
- receiver FIFO after retention prune;
- exhausted-energy transmission denial.

For native-supported outputs, retain Native↔Gen1 checks; then assert rxindex canonical state/timestamps match Gen1 exactly.

- [ ] **Step 3: Add rxindex to Scenario-V3 backends**

Extend `ScenarioV3IntegrationTests.backends()`/`prepare()` with `warp_rxindex` and include it in:

- 1 MB direct delivery;
- two-hop relay;
- three-hop relay;
- failed low-power then high-power recovery.

Add or rename the contention parity test to compare Gen1 and rxindex every step on all snapshot arrays, selected network metrics, rewards/termination, and communication events.

- [ ] **Step 4: Run semantic corpus**

Run:

```bash
pytest -q tests/test_gpu_packets.py tests/test_scenario_v3_integration.py
```

Expected: Native↔Gen1 remains green; Gen1↔rxindex exact comparisons pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add tests/test_gpu_packets.py tests/test_scenario_v3_integration.py
git commit -m "test(net): verify rxindex against UavNetSim corpus"
```

---

### Task 5: Benchmark Harness and Full Local Regression

**Files:**
- Modify: `uav_network/experiments/mac_scheduler_benchmark.py`
- Modify: `tests/test_mac_scheduler_benchmark.py`
- Modify: `tests/test_gpu_cuda.py`

**Interfaces:**
- Consumes: `packet_step_rxindex`, `WarpVectorEnv(... packet_scheduler="rxindex")`.
- Produces: local benchmark contract and CUDA exactness test entry for Task 6.

- [ ] **Step 1: Write failing benchmark-contract test**

Assert `benchmark_mac_scheduler(... scheduler="rxindex")` is accepted, reports scheduler `rxindex`, scientific metrics contract unchanged, scheduler metrics remain zero, and allocated memory includes derived receiver arrays.

- [ ] **Step 2: Implement benchmark scheduler selection**

Extend `SCHEDULERS` with `rxindex`, select `packet_step_rxindex` for timing, and keep Gen1/Gen2 logic unchanged.

- [ ] **Step 3: Add CUDA exactness test path**

Extend the existing active 1 MB CUDA exactness test or add a sibling test that runs Gen1 and rxindex on the same fixed corpus and compares every canonical snapshot array with `rtol=0`, `atol=0`, `equal_nan=True`. Assert an active 1 MB session exists so the test cannot pass on an idle corpus.

- [ ] **Step 4: Run focused local suite**

Run:

```bash
pytest -q \
  tests/test_receiver_index.py \
  tests/test_gpu_packets.py \
  tests/test_scenario_v3_integration.py \
  tests/test_exact_environment_resume.py \
  tests/test_mac_scheduler_benchmark.py \
  tests/test_gpu_cuda.py
```

Expected: all CPU-capable tests pass; CUDA-only tests may skip locally when no driver exists.

- [ ] **Step 5: Run full project regression**

Run:

```bash
pytest -q
```

Expected: zero failures. Record pass/skip/subtest totals before any T4 claim.

- [ ] **Step 6: Verify diff hygiene**

Run:

```bash
git diff --check
git status --short
git diff --name-only d1e7c6a32322a4c8e1c5d5aa0faa046b54f2b4bd...HEAD
```

Confirm no Task-5 changes to `phy.py`, training algorithms, Scenario-V3 config, reward definitions, or checkpoint version/schema.

- [ ] **Step 7: Commit Task 5**

```bash
git add uav_network/experiments/mac_scheduler_benchmark.py tests/test_mac_scheduler_benchmark.py tests/test_gpu_cuda.py
git commit -m "bench(net): gate rxindex exactness and throughput"
```

---

### Task 6: Tesla T4 Exactness, Throughput, and Promotion Decision

**Files:**
- Create: `docs/superpowers/reports/2026-10-22-receiver-pending-index-t4.md`
- Generated outputs under: `outputs/task5-rxindex-t4/` (not product source)

**Interfaces:**
- Consumes: final Task-5 branch SHA after local regression.
- Produces: authoritative T4 correctness/performance report and promote/reject decision.

- [ ] **Step 1: Freeze candidate SHA and verify remote/local identity**

Record:

```bash
git rev-parse HEAD
git status --short
```

Push the experimental branch only after local gates are green. Verify GitHub branch head equals the local SHA before Kaggle launch.

- [ ] **Step 2: Run T4 CUDA exactness first**

On a spare Kaggle Tesla T4 account, run only the Gen1↔rxindex active 1 MB CUDA exactness test on the frozen SHA.

Promotion work stops immediately if canonical state differs or simulator errors occur.

- [ ] **Step 3: Run repeated 2,048-env benchmark**

Using `configs/comparison-20k-steps1800-scenario-v3.yaml`, fixed seed, 2,048 envs, four measured post-JIT steps, run at least three repeats for each scheduler/corpus:

- Gen1 low-contention;
- rxindex low-contention;
- Gen1 active-streaming 1 MB;
- rxindex active-streaming 1 MB.

Alternate scheduler order between repeats to reduce warmup/order bias.

- [ ] **Step 4: Validate benchmark science and memory**

For every repeat assert:

- scientific metrics Gen1 == rxindex;
- simulator error count == 0;
- unexpected terminated worlds == 0;
- receiver-index added memory is reported;
- no OOM/capacity failure.

Calculate median `joint_transitions_per_second`, median `packet_step_seconds`, and active-streaming speed ratio.

- [ ] **Step 5: Write decision report**

In `docs/superpowers/reports/2026-10-22-receiver-pending-index-t4.md`, record:

- frozen SHA / Kaggle kernel refs;
- CUDA exactness result;
- per-repeat and median throughput;
- packet-step time;
- added memory;
- scientific parity/errors;
- decision: `promote_to_next_gate` only when repeated active-streaming speedup is positive and correctness gates pass; otherwise `keep_gen1`.

No production restart occurs in Task 6.

- [ ] **Step 6: Commit report only after evidence is downloaded**

```bash
git add docs/superpowers/reports/2026-10-22-receiver-pending-index-t4.md
git commit -m "docs(net): record rxindex T4 decision"
```

- [ ] **Step 7: Final verification before claiming completion**

Run fresh:

```bash
pytest -q
git diff --check
git status --short
git log -1 --oneline
```

Then verify the downloaded T4 JSON checksum and GitHub frozen SHA before reporting Task 5 complete.
