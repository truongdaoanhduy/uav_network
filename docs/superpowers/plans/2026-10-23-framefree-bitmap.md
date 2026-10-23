# Frame-Free Bitmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental `framefree` scheduler that removes Gen1's full `frame_capacity` free/reclaim scan from `emit()` while preserving exact Gen1/UavNetSim-derived behavior, reset/resume semantics, and canonical state.

**Architecture:** Keep Gen1 untouched. Add two derived per-environment `uint64` bitmaps—`free_words` and `reclaim_words`—owned by `WarpVectorEnv` only when `packet_scheduler="framefree"`. `framefree` keeps Gen1 `frame_order` retention compaction and event ordering, substitutes an `emit_framefree()` for DATA and ACK emission, and rebuilds bitmap state from canonical arrays after reset/restore. Promotion is gated on exact parity first, then a repeated 2×T4 benchmark requiring at least 2% median active-streaming improvement and no material low-contention regression.

**Tech Stack:** Python 3.13, NVIDIA Warp 1.18, PyTorch, NumPy, pytest/unittest, Kaggle 2×Tesla T4 benchmark harness.

**Spec:** `docs/superpowers/specs/2026-10-23-framefree-bitmap-design.md`

## Global Constraints

- Start from branch `perf/framefree-bitmap-20261023`; Gen1 remains production default and oracle.
- Do not import, call, or combine the rejected `rxindex` implementation.
- Preserve Gen1 event timestamp, priority, insertion-order, RNG, PHY, report, reward, observation, and canonical cleanup semantics exactly.
- Preserve `frame_order` compaction and interference accumulation order exactly.
- `framefree` must choose the globally smallest reusable frame index, matching Gen1.
- `tx_active == 2 && tx_receive == 3` must remain canonical until the next `emit()` call; derived reclaim state must not mutate `tx_active` early.
- `free_words` and `reclaim_words` must not appear in `snapshot_state()["arrays"]`; they must not enter `WorldState`, BASE/PQL checkpoints, or change snapshot schema/version.
- Scenario V3 bitmap target is 26 words per row per bitmap at `frame_capacity=1632`; total derived state at 2,048 envs should be about 0.8125 MiB.
- Do not stop, restart, migrate, or modify the four current MASAC/MATD3 BASE/PQL training runs.
- No performance benchmark is valid until CPU and CUDA exactness gates pass.
- Promotion gate: active-streaming median throughput gain >= 2.0%, low-contention regression <= 2.0%, zero simulator errors, no OOM.

## Review Focus

1. **Tail bits in the last bitmap word** — invalid indices >= `frame_capacity` must stay zero so allocation can never return an out-of-range frame. Task 1 adds an explicit non-multiple-of-64 test.
2. **Deferred receiver cleanup timing** — marking a reclaim candidate must not change canonical `tx_active` before the next `emit()`. Tasks 1–2 test the pre-emit and post-emit states explicitly.
3. **All reclaims, not only the chosen frame** — each `emit_framefree()` must canonicalize every pending `tx_active==2 && tx_receive==3` frame before allocating. Task 2 checks multiple reclaim candidates including indices above and below the chosen slot.
4. **Partial reset/restore isolation** — rebuilding derived state must change only selected reset rows and must not leak bitmap arrays into snapshots. Task 3 tests partial reset plus Gen1↔framefree exact continuation.
5. **Benchmark order/JIT bias** — cold compile/cache effects must not favor the second scheduler. Task 5 uses compile outside measured regions and alternating/repeated scheduler order before computing the median gate.

---

### Task 1: Add exact frame bitmap primitives

**Files:**
- Create: `uav_network/communication/gpu/frame_free.py`
- Create: `tests/test_frame_free.py`

**Interfaces:**
- Consumes: canonical `WorldState.tx_active`, `WorldState.tx_receive`, and `WorldParams.frame_capacity`.
- Produces:
  - `bitmap_words(frame_capacity: int) -> int` (host helper)
  - `rebuild_frame_bitmaps(s: WorldState, p: WorldParams, e: int, free_words: wp.array2d(dtype=wp.uint64), reclaim_words: wp.array2d(dtype=wp.uint64)) -> None` (`@wp.func` or kernel-backed equivalent used by later reset/restore integration)
  - `mark_reclaim_if_ready(s: WorldState, e: int, frame: int, reclaim_words: wp.array2d(dtype=wp.uint64)) -> None`
  - `set_free_bit(free_words, e, frame)`, `clear_free_bit(free_words, e, frame)`, `set_reclaim_bit(reclaim_words, e, frame)`, and `clear_reclaim_bit(reclaim_words, e, frame)` internal helpers
  - `apply_reclaims_and_take_smallest(s, p, e, free_words, reclaim_words) -> int`: canonicalizes all queued reclaims, then returns/clears the globally smallest free valid frame, or `-1`.

- [ ] **Step 1: Write failing primitive tests in `tests/test_frame_free.py`**

Cover all of these assertions:

```python
def test_word_count_and_tail_mask_for_1632_frames():
    assert bitmap_words(1632) == 26
    # after rebuild from all-free canonical state, frame 1631 is selectable
    # and no bit for frame 1632+ can ever be selected


def test_smallest_free_crosses_word_boundary():
    # frames 0..63 busy; 64 and 130 free
    # allocation returns 64 and clears only frame 64


def test_reclaim_mark_does_not_mutate_canonical_state():
    # tx_active=2, tx_receive transitions to 3
    # marking sets reclaim bit but leaves tx_active==2


def test_apply_reclaims_canonicalizes_all_candidates_before_selection():
    # reclaim frames 2 and 120, ordinary free frame 50
    # after call both reclaim frames have tx_active=0,
    # all reclaim bits are clear, and selected index is 2


def test_no_free_frame_returns_minus_one_without_out_of_range_bit():
    # all valid frames busy and tail bits masked
    assert selected == -1
```

- [ ] **Step 2: Run the focused test file and verify RED**

Run: `python3 -m pytest -q tests/test_frame_free.py`

Expected: FAIL because `uav_network.communication.gpu.frame_free` and the new primitives do not exist.

- [ ] **Step 3: Implement minimal bitmap primitives in `uav_network/communication/gpu/frame_free.py`**

Requirements:
- word index = `frame // 64`; bit index = `frame % 64`;
- last-word invalid tail bits are never set;
- rebuild derives state only from canonical `tx_active`/`tx_receive`;
- `mark_reclaim_if_ready()` only sets a bit when post-state is exactly `tx_active==2 && tx_receive==3`;
- `apply_reclaims_and_take_smallest()` scans words and bits deterministically in ascending order because Warp 1.18 has no project-available `ctz/ffs` primitive;
- process every reclaim bit before choosing the smallest free bit;
- do not mutate any canonical field except the Gen1-equivalent `tx_active=0` reclaim transition.

- [ ] **Step 4: Re-run primitive tests and verify GREEN**

Run: `python3 -m pytest -q tests/test_frame_free.py`

Expected: all Task-1 tests PASS.

- [ ] **Step 5: Run existing retention/PHY regressions**

Run: `python3 -m pytest -q tests/test_propagation.py tests/test_gpu_packets.py::GPUPacketTests::test_short_frames_preserve_receiver_fifo_after_phy_retention_prune`

Expected: PASS with Gen1 unchanged.

- [ ] **Step 6: Commit Task 1**

```bash
git add uav_network/communication/gpu/frame_free.py tests/test_frame_free.py
git commit -m "feat(net): add exact frame-free bitmap primitives"
```

---

### Task 2: Add `emit_framefree()` and exact DATA/ACK event integration

**Files:**
- Modify: `uav_network/communication/gpu/phy.py` near `emit()`
- Modify: `uav_network/communication/gpu/mac.py` near DATA `TX_START`, ACK `ACK_START`, and receiver completion branches
- Modify: `tests/test_frame_free.py`

**Interfaces:**
- Consumes Task 1 `free_words`, `reclaim_words`, `apply_reclaims_and_take_smallest()`, and `mark_reclaim_if_ready()`.
- Produces:
  - `emit_framefree(s: WorldState, p: WorldParams, e: int, packet: int, source: int, destination: int, kind: int, now: wp.float64, free_words: wp.array2d(dtype=wp.uint64), reclaim_words: wp.array2d(dtype=wp.uint64)) -> int`
  - `event_framefree(s, p, e, kind, arg, generation, now, free_words, reclaim_words)` preserving Gen1 behavior for all non-emission logic.

- [ ] **Step 1: Add failing Gen1-vs-framefree emit tests**

Add tests that construct identical canonical state and compare Gen1 `emit()` with `emit_framefree()` for:

```python
def test_emit_framefree_matches_gen1_retention_compaction_and_selected_index(): pass
def test_emit_framefree_reclaims_every_completed_receiver_frame(): pass
def test_emit_framefree_preserves_frame_order_and_interference_order(): pass
def test_emit_framefree_matches_error_three_when_capacity_exhausted(): pass
```

Assertions must include exact equality of `frame_count`, `frame_order`, `tx_active`, `tx_receive`, selected frame index, and all `tx_*` fields modified by emission.

- [ ] **Step 2: Run the new emit tests and verify RED**

Run: `python3 -m pytest -q tests/test_frame_free.py -k 'emit or retention or reclaim'`

Expected: FAIL because `emit_framefree()` / `event_framefree()` do not exist.

- [ ] **Step 3: Implement `emit_framefree()` in `phy.py`**

Keep these Gen1 regions semantically identical:
- power/energy denial and energy accounting;
- `frame_order` retention compaction in existing order;
- all diagnostics and required-power sampling;
- frame canonical field population;
- propagation-delay table updates;
- trace writes.

Replace only the full `for j in range(p.frame_capacity)` free/reclaim scan with:
1. update bitmap state for frames made immediately free by retention compaction;
2. preserve `tx_active=2` for receiver-retained frames;
3. call Task-1 reclaim/allocation primitive;
4. set error 3 and return `-1` when no bit is free.

- [ ] **Step 4: Implement narrow `event_framefree()` in `mac.py`**

Requirements:
- specialize DATA `TX_START` to use `emit_framefree()` with the same duration/event pushes as Gen1;
- specialize ACK `ACK_START` the same way;
- delegate every other event kind to Gen1 `event()`;
- after receiver-related delegated events (`DELIVER`, `RECEIVE`, `RECEIVE_DONE`), call `mark_reclaim_if_ready()` for the involved frame;
- do not copy unrelated Gen1 branches.

- [ ] **Step 5: Verify emit/event exactness locally**

Run:

```bash
python3 -m pytest -q tests/test_frame_free.py
python3 -m pytest -q tests/test_gpu_packets.py tests/test_propagation.py
```

Expected: PASS; Gen1 test counts remain unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add uav_network/communication/gpu/phy.py uav_network/communication/gpu/mac.py tests/test_frame_free.py
git commit -m "feat(net): add experimental framefree emit path"
```

---

### Task 3: Wire scheduler allocation, reset, restore, and canonical snapshot isolation

**Files:**
- Modify: `uav_network/environment/gpu/vector.py`
- Modify: `uav_network/communication/gpu/mac.py` for `packet_step_framefree` entry point
- Modify: `tests/test_frame_free.py`
- Modify: `tests/test_exact_environment_resume.py`

**Interfaces:**
- Consumes Task 1 bitmap rebuild functions and Task 2 `event_framefree()`.
- Produces:
  - scheduler name `framefree` accepted by `WarpVectorEnv(config, packet_scheduler="framefree")`;
  - private derived arrays `self._frame_free_words`, `self._frame_reclaim_words` allocated only for framefree;
  - `packet_step_framefree` entry point with the same preamble/event-loop/finish semantics as Gen1 while dispatching through `event_framefree()`;
  - vector-side `_rebuild_framefree(mask=None)` helper used after reset and restore.

- [ ] **Step 1: Add failing scheduler/reset/snapshot tests**

Add tests for:

```python
def test_framefree_scheduler_allocates_about_expected_bitmap_bytes():
    # 2048 env, frame_capacity 1632 => 851,968 derived bitmap bytes


def test_framefree_partial_reset_rebuilds_only_selected_rows(): pass

def test_framefree_derived_arrays_are_absent_from_snapshot_arrays(): pass
def test_gen1_snapshot_restores_into_framefree_and_continues_exactly(): pass
def test_framefree_snapshot_restores_into_gen1_and_continues_exactly(): pass
```

For cross-scheduler continuation, compare canonical snapshot arrays after several identical actions with `rtol=0, atol=0`.

- [ ] **Step 2: Run the new integration tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_frame_free.py tests/test_exact_environment_resume.py -k 'framefree'
```

Expected: FAIL because the scheduler is not accepted and no private bitmap arrays/rebuild path exist.

- [ ] **Step 3: Add `framefree` scheduler state to `WarpVectorEnv`**

In `__init__`:
- accept `framefree` alongside existing scheduler values;
- allocate `(num_envs, bitmap_words(p.frame_capacity))` `wp.uint64` arrays for free/reclaim only when framefree;
- keep them as private vector attributes, not fields of `WorldState`;
- include derived arrays explicitly in `allocated_bytes` so benchmark memory accounting sees them.

- [ ] **Step 4: Add exact reset/rebuild integration**

- after `reset_network` for selected rows, rebuild framefree bitmaps only for the reset mask;
- after copying canonical arrays in `restore_state`, rebuild all rows before returning;
- do not alter snapshot `version`, `config`, or `arrays` keys.

- [ ] **Step 5: Add `packet_step_framefree()` and dispatch it from `step_async()`**

Match Gen1 packet-step structure and limits. The only event-dispatch difference is calling `event_framefree()` with derived bitmaps. Gen1 and Gen2 code paths remain unchanged.

- [ ] **Step 6: Verify reset/resume/snapshot exactness**

Run:

```bash
python3 -m pytest -q tests/test_frame_free.py tests/test_exact_environment_resume.py
python3 -m pytest -q tests/test_gpu_packets.py tests/test_propagation.py
```

Expected: PASS, including Gen1→framefree and framefree→Gen1 exact continuation.

- [ ] **Step 7: Commit Task 3**

```bash
git add uav_network/environment/gpu/vector.py uav_network/communication/gpu/mac.py tests/test_frame_free.py tests/test_exact_environment_resume.py
git commit -m "feat(net): wire framefree scheduler reset and resume"
```

---

### Task 4: Prove scientific parity across the network corpus and CUDA

**Files:**
- Modify: `tests/test_frame_free.py`
- Modify: `tests/test_gpu_packets.py` only if a reusable comparison helper must accept a scheduler argument; do not alter expected Gen1/native results.
- Create or modify: `tests/test_framefree_cuda.py` if a dedicated CUDA gate is clearer than parameterizing existing tests.

**Interfaces:**
- Consumes fully wired `WarpVectorEnv(config, packet_scheduler="framefree")`.
- Produces exactness evidence only; no new production behavior.

- [ ] **Step 1: Add failing/parameterized scientific corpus comparisons**

Exercise framefree against Gen1/native expectations for:
- direct DATA+ACK delivery;
- simultaneous contention/collision;
- hidden sender interference;
- retry/ACK timeout;
- propagation delay on/off;
- energy denial and exact budget depletion;
- short-frame FIFO after retention prune;
- 1 MB streaming/relay path;
- report expiration/TTL;
- equal-time event-order cases.

Where the harness can inspect canonical state, assert exact arrays; otherwise assert exact network/scientific metrics and event trace ordering.

- [ ] **Step 2: Run focused CPU corpus**

Run:

```bash
python3 -m pytest -q tests/test_frame_free.py tests/test_gpu_packets.py tests/test_propagation.py tests/test_exact_environment_resume.py
```

Expected: PASS with zero framefree divergences.

- [ ] **Step 3: Add/verify the CUDA exactness entry point**

The test must run the same deterministic corpus on CUDA and assert:
- Gen1 vs framefree scientific metrics equal;
- canonical arrays equal at completed boundaries where supported;
- `errors == 0`;
- no out-of-range frame allocation;
- derived state memory matches the expected shape and dtype.

- [ ] **Step 4: Run local non-CUDA wide suite before cloud use**

Run:

```bash
python3 -m pytest -q tests/test_frame_free.py tests/test_gpu_packets.py tests/test_propagation.py tests/test_exact_environment_resume.py tests/test_mac_scheduler_gen2.py
```

Expected: PASS.

- [ ] **Step 5: Launch one bounded T4 CUDA exactness job on an idle account not used by the four production trainings**

Pin the current commit SHA in the Kaggle payload. The job must run only the focused exactness suite; do not start the performance benchmark if exactness fails.

Evidence to save under a new task-specific output directory:
- kernel reference/account;
- pinned commit;
- test command;
- test counts;
- exit status;
- CUDA device names;
- checksum of downloaded log/result.

- [ ] **Step 6: Commit Task 4 exactness harness/evidence metadata**

```bash
git add tests/test_frame_free.py tests/test_gpu_packets.py
# Add tests/test_framefree_cuda.py only if Task 4 creates it.
# Add the exact evidence file path created by the CUDA launcher, if committed.
git commit -m "test(net): prove framefree scientific exactness"
```

If the implementation required no new committed evidence file, omit nonexistent paths and commit only test/harness changes.

---

### Task 5: Benchmark Gen1 vs framefree on T4 and close or advance the candidate

**Files:**
- Modify: `uav_network/experiments/mac_scheduler_benchmark.py`
- Modify/Create: task-specific Kaggle benchmark launcher under `uav_network/experiments/` or `outputs/` following the existing benchmark pattern; do not add secrets.
- Create: `docs/superpowers/reports/2026-10-23-framefree-bitmap-t4.md`
- Test: `tests/test_mac_scheduler_benchmark.py` and/or `tests/test_frame_free.py`

**Interfaces:**
- Consumes scheduler `framefree` from Task 3 and exactness approval from Task 4.
- Produces repeated Gen1/framefree measurements for `active_streaming` and `low_contention`, memory accounting, and the final `ADVANCE` or `KEEP GEN1` decision.

- [ ] **Step 1: Add failing benchmark-harness tests**

Assert that:
- `framefree` is accepted by benchmark scheduler choices;
- selected packet-step timing measures `packet_step_framefree` for that scheduler;
- result payload reports derived bitmap bytes separately or in a clearly auditable total;
- scientific metric keys remain identical across schedulers.

- [ ] **Step 2: Run benchmark harness tests and verify RED**

Run: `python3 -m pytest -q tests/test_mac_scheduler_benchmark.py tests/test_frame_free.py -k 'benchmark or memory'`

Expected: FAIL until `framefree` is added to the benchmark harness.

- [ ] **Step 3: Extend benchmark harness minimally**

Add `framefree` to the scheduler set and map it to `packet_step_framefree`; do not change corpus preparation or Gen1 behavior. Record derived bitmap allocation so the expected ~0.8125 MiB can be audited.

- [ ] **Step 4: Verify local harness**

Run:

```bash
python3 -m pytest -q tests/test_mac_scheduler_benchmark.py tests/test_frame_free.py
python3 -m pytest -q tests/test_gpu_packets.py tests/test_propagation.py tests/test_exact_environment_resume.py
```

Expected: PASS.

- [ ] **Step 5: Launch repeated T4 benchmark on an idle non-production account**

Use Scenario V3, 2,048 envs, 1 MB reports. For each corpus (`active_streaming`, `low_contention`), collect at least three measured repetitions per scheduler with compilation outside the measured region. Alternate order, for example:

```text
repeat 1: Gen1 -> framefree
repeat 2: framefree -> Gen1
repeat 3: Gen1 -> framefree
```

Do not use accounts currently backing the four BASE/PQL training jobs.

- [ ] **Step 6: Download and validate benchmark evidence**

For every run/repetition verify:
- pinned commit matches branch HEAD;
- `simulator_error_count == 0`;
- scientific metrics match Gen1 exactly within each paired corpus run;
- no OOM/failure;
- derived memory is consistent with the expected bitmap allocation;
- raw JSON/log checksums are recorded.

Compute medians from measured steady regions only; exclude JIT/cold setup.

- [ ] **Step 7: Apply the promotion gate exactly**

Decision logic:

```text
if parity fails or simulator errors/OOM:
    REJECT FRAMEFREE
elif active_streaming median gain < 2.0%:
    KEEP GEN1
elif low_contention median regression > 2.0%:
    KEEP GEN1
else:
    ADVANCE FRAMEFREE TO SEPARATE PRODUCTION-SHAPED GATE
```

Passing this task does not change the production scheduler.

- [ ] **Step 8: Write the T4 decision report**

Create `docs/superpowers/reports/2026-10-23-framefree-bitmap-t4.md` containing:
- branch/commit/kernel/account evidence;
- exact config/hardware;
- scientific parity result;
- per-repeat and median throughput/wall-time;
- memory delta;
- checksums;
- explicit final decision.

- [ ] **Step 9: Run final verification before claiming completion**

Run the focused suite fresh:

```bash
python3 -m pytest -q tests/test_frame_free.py tests/test_gpu_packets.py tests/test_propagation.py tests/test_exact_environment_resume.py tests/test_mac_scheduler_benchmark.py tests/test_mac_scheduler_gen2.py

git diff --check
git status --short --branch
```

Expected: zero test failures, no unexpected working-tree changes.

- [ ] **Step 10: Commit and push the benchmark decision**

```bash
git add uav_network/experiments/mac_scheduler_benchmark.py tests/test_mac_scheduler_benchmark.py tests/test_frame_free.py docs/superpowers/reports/2026-10-23-framefree-bitmap-t4.md
git commit -m "bench(net): gate framefree exactness and throughput"
git push origin perf/framefree-bitmap-20261023
```

If the gate returns `KEEP GEN1`, stop this optimization task. If it returns `ADVANCE`, the next task is a separate approved production-shaped collector/learner probe; do not merge or promote `framefree` automatically.


## Cloud CUDA gate audit — 2026-10-23

- Local Framefree scientific/parity gate remains green: focused cloud+framefree suite 46 passed / 717 subtests; full branch suite 648 passed / 100 skipped / 1246 subtests. CUDA-only cases are skipped locally, so this is not a CUDA exactness claim.
- The earlier CUDA kernel `bjkjbhm/uav-framefree-cuda-exact-20261023-090433` is private and was pinned to stale commit `5146efe`, while the Framefree branch advanced to `d7cf8bb` and later. The default Kaggle credential is not that private kernel owner, so `kaggle kernels status` reproduces `Permission 'kernels.get' was denied`.
- Root-cause fix: `uav_network.experiments.framefree_cuda_gate` now generates a credential-free **public** GPU diagnostic kernel, rejects non-exact commit SHAs, records CUDA/GPU/pytest evidence, and validates the downloaded result. `scripts/framefree_cuda_gate.py` provides `prepare` and `verify`; it never reads `key.txt` or account-specific tokens. Production training kernels remain private and unchanged.
- Do not promote Framefree or launch the benchmark until a public CUDA gate generated from the final pushed Framefree commit returns `torch_cuda=true`, `gpu_count>=1`, `pytest_returncode=0`, and the exact same commit SHA.
