# Frame-Free Bitmap Design

## Purpose

Speed up the Scenario V3 `warp_packet` network simulator by removing the full `frame_capacity` free-slot/reclaim scan from `emit()` while preserving Gen1 scientific behavior and canonical environment state exactly.

This milestone introduces an experimental scheduler mode named `framefree`. Gen1 remains the production default and oracle. `framefree` changes only frame recycling/allocation bookkeeping around `emit()`; it does not combine with the rejected `rxindex` experiment and does not modify any currently running MASAC/MATD3 BASE/PQL training job.

## Starting Point

Implementation work starts from commit `d1e7c6a32322a4c8e1c5d5aa0faa046b54f2b4bd` on branch `perf/framefree-bitmap-20261023`. This is the exact pre-`rxindex` point used so the new result is attributable to frame allocation alone.

Relevant baseline verification at this starting point:

- `tests/test_gpu_packets.py`
- `tests/test_propagation.py`
- `tests/test_exact_environment_resume.py`
- `tests/test_mac_scheduler_gen2.py`
- result: 61 tests passed, 733 subtests passed

## Evidence and Motivation

Task 4 profiling measured the remaining `emit()` frame recycling/allocation path as a major fixed-capacity scan:

- Scenario V3 `frame_capacity = 1632`.
- Approximately 4,603,051,104 frame-slot iterations were attributable to `emit()` recycling/free-slot scans in the measured active-streaming corpus.
- `emit()` already performs a necessary `frame_order` compaction pass to preserve the PHY retention set and interference summation order.
- After that compaction, Gen1 performs a second scan over all 1,632 frame slots to:
  1. convert `tx_active == 2 && tx_receive == 3` into `tx_active = 0`, and
  2. choose the smallest frame index with `tx_active == 0`.

The previous `rxindex` task intentionally deferred this work. Its production-shaped gate later showed only a small end-to-end benefit and was closed with `KEEP GEN1`. This task therefore returns to the independent `emit()` hotspot rather than combining optimizations.

## Immutable Scientific Semantics

`framefree` must not change any of the following:

- Gen1 event timestamps, priorities, insertion order, heap behavior, or fast-root behavior.
- CSMA/CA, DIFS/SIFS, contention windows, CCA, backoff draws/freezes, retries, ACK timeout behavior, or packet deadlines.
- DATA/ACK airtime or propagation timing.
- Path loss, LoS/NLoS, obstacle attenuation, SINR, interference, collisions, PHY thresholds, or receiver energy.
- The order in which `frame_order` is compacted.
- The order in which `busy()` and `evaluate()` accumulate interference over `frame_order`.
- The smallest-index free-frame choice made by Gen1.
- The canonical timing of `tx_active` state changes, including delayed cleanup of `tx_active == 2` receiver-retained frames.
- RNG state, RNG draw count, or RNG draw order.
- Packet/session/report generation, fragment windows, TTL, custody, relay hops, duplicate behavior, or report delivery timing.
- Reward, observations, centralized state, MASAC, MATD3, BASE, PQL, replay, AMP, compilation, optimizers, or checkpoints.
- Exact environment snapshot schema or canonical arrays.

Any speedup with canonical, scientific, ordering, reset, or resume divergence is a failed candidate.

## Current Gen1 Behavior That Must Be Reproduced

### Retention compaction

At the beginning of `emit()`, Gen1 walks `frame_order[0:frame_count]` in order. Frames whose latest arrival end is still inside the retention window are copied into the compacted prefix in the same order. Expired frames are removed from `frame_order` and receive one of two canonical state transitions:

- if `tx_receive` is 1 or 2, set `tx_active = 2`; the frame remains reserved for the serialized receiver inbox;
- otherwise set `tx_active = 0`; the frame is immediately reusable.

`framefree` keeps this entire loop and its ordering unchanged.

### Deferred receiver reclaim

A frame may have already left PHY retention while it is pending or active in the serialized receiver inbox. In that state Gen1 uses:

- `tx_active == 2`, and
- `tx_receive` equal to 1 or 2.

When receiver processing later completes, `tx_receive` becomes 3, but Gen1 does **not** immediately set `tx_active = 0`. The canonical transition from `tx_active == 2` to `tx_active == 0` occurs at the next call to `emit()` during the second full-frame scan.

`framefree` must preserve that timing exactly. It may remember the candidate in derived state at receiver completion, but it may not mutate canonical `tx_active` early.

### Smallest free frame

The Gen1 second scan walks frame indices from 0 to `frame_capacity - 1`, performs every reclaim transition, and records the first free frame index. Therefore `framefree` must both:

1. canonicalize **all** pending reclaims at each `emit()`, not only the selected frame; and
2. return the globally smallest reusable frame index.

## Chosen Architecture

### Experimental scheduler isolation

Add scheduler mode `framefree` beside the existing scheduler modes available at the starting commit.

- `gen1` remains unchanged and remains the default.
- `framefree` uses Gen1 packet-step setup, event heap, event ordering, fast-root logic, PHY equations, reports, streaming, diagnostics, and packet-step finish.
- `framefree` substitutes an experimental `emit_framefree()` only for DATA `TX_START` and ACK `ACK_START` event branches.
- Other event kinds continue through the existing Gen1 `event()` implementation.
- After Gen1 event handling, only receiver-state transitions that can create a deferred reclaim candidate update the derived reclaim bitmap.
- The experiment must not call `rxindex`, allocate receiver heaps, or use any `rxindex` code.

This keeps the change attributable to frame recycling/allocation and leaves an immediate rollback path to Gen1.

## Derived Bitmap State

For each environment, allocate two derived `uint64` bitmaps over frame indices.

### `free_words`

A bit is 1 exactly when the corresponding valid frame index is canonically free for allocation:

`tx_active == 0`

Bits outside `frame_capacity` in the last word are always zero.

### `reclaim_words`

A bit is 1 when receiver processing has made a frame eligible for the *next Gen1-style emit-time cleanup*:

`tx_active == 2 && tx_receive == 3`

This bitmap records work that Gen1 would discover during the next full-frame scan. Setting the bit does not change canonical state.

### Scenario V3 memory cost

With `frame_capacity = 1632`:

- words per bitmap row = `ceil(1632 / 64) = 26`;
- one bitmap at 2,048 environments = `2048 * 26 * 8 = 425,984` bytes;
- two bitmaps = `851,968` bytes = 0.8125 MiB.

This is derived acceleration state, not scientific state.

## Bitmap Operations

### Reset

For every reset environment row:

- initialize valid `free_words` bits to 1;
- mask tail bits beyond `frame_capacity` to 0;
- initialize all `reclaim_words` bits to 0.

Partial vector reset must modify only reset rows.

### Restore/rebuild

Derived bitmap state is excluded from exact environment snapshots. After restoring canonical state, rebuild both bitmaps from canonical arrays in one explicit reconstruction pass:

- set free bit iff `tx_active == 0`;
- set reclaim bit iff `tx_active == 2 && tx_receive == 3`.

Rebuilding must not mutate canonical arrays. Gen1 snapshots and `framefree` snapshots retain the same schema.

### Marking a reclaim candidate

After an event transition sets `tx_receive = 3`, mark the frame's reclaim bit only if canonical `tx_active == 2`.

The relevant Gen1 event families are:

- `DELIVER` when PHY evaluation fails;
- `RECEIVE` paths that finish/reject receiver service;
- `RECEIVE_DONE`.

The implementation should use one small helper that checks the canonical condition after the Gen1 transition instead of duplicating condition logic at each site.

### Retention expiration inside `emit_framefree()`

Keep Gen1's `frame_order` compaction loop exactly. For an expired frame:

- if receiver state requires retention (`tx_receive` 1 or 2), perform the existing `tx_active = 2` transition and keep its free bit cleared;
- otherwise perform the existing `tx_active = 0` transition and set its free bit.

No retained-frame order may change.

### Applying deferred reclaims

Before choosing a free frame, process all set bits in `reclaim_words` in ascending word order and ascending bit order. For every valid set bit:

- assert through test invariants that canonical state is `tx_active == 2 && tx_receive == 3`;
- perform the same canonical `tx_active = 0` transition Gen1 would perform in its second scan;
- set the corresponding free bit;
- clear the reclaim bit.

All deferred reclaims must be processed before allocation, even if a smaller free slot is already known, because Gen1 cleans every reclaimable frame on each `emit()` call.

Warp 1.18 does not expose a direct `ctz`/`ffs` primitive in the project runtime, so the first implementation uses deterministic word-order plus bit-order loops. This keeps the design portable and exact; a later low-level bit primitive is out of scope.

### Selecting the free frame

After all emit-time reclaim work is canonicalized:

1. scan `free_words` from lowest word to highest;
2. within the first nonzero word, scan bits from low to high;
3. return the first valid set bit;
4. clear that free bit before filling canonical frame fields.

This must return the exact same smallest frame index as Gen1.

If no free bit exists:

- set `errors[e] = 3`;
- return `-1`;

matching Gen1 behavior.

## DATA and ACK Integration

`framefree` must not duplicate the entire Gen1 event handler.

Implement a narrow experimental event wrapper:

- intercept DATA `TX_START` only to execute the same Gen1 branch with `emit_framefree()`;
- intercept `ACK_START` only to execute the same Gen1 branch with `emit_framefree()`;
- delegate every other event kind to Gen1 `event()`;
- after delegated receiver-related events, update only the derived reclaim bitmap if the canonical post-state qualifies.

The specialized DATA and ACK branches must remain line-for-line semantically equivalent to Gen1 apart from the selected emit function.

## Canonical State and Snapshot Contract

`free_words` and `reclaim_words` are derived acceleration state.

They must:

- not appear in `snapshot_state()["arrays"]`;
- not change snapshot version or schema;
- not be serialized into BASE/PQL checkpoints;
- be rebuilt after restore;
- be rebuilt after loading a Gen1 snapshot into a `framefree` environment;
- permit loading a `framefree` snapshot into Gen1 with exact canonical continuation.

Exact resume tests compare canonical arrays after subsequent steps, not merely final mission metrics.

## Correctness and TDD Gates

Implementation begins with failing tests before product code.

### Bitmap primitive tests

Prove:

- reset marks exactly valid frame indices free;
- tail bits outside `frame_capacity` remain zero;
- smallest-index selection crosses 64-bit word boundaries correctly;
- allocation clears the chosen bit;
- reclaim marking does not mutate canonical `tx_active` early;
- emit-time reclaim canonicalizes all queued candidates;
- mixed `tx_active` states produce the same chosen index as Gen1;
- zero free slots reproduce error 3.

### Retention and ordering tests

Construct cases containing:

- retained PHY frames;
- expired immediately reusable frames;
- expired receiver-retained frames (`tx_active == 2`);
- later receiver-completed reclaim frames;
- free slots both below and above reclaim candidates.

After `emit()`, compare Gen1 and `framefree` exactly for:

- selected frame index;
- `frame_order` and `frame_count`;
- all `tx_*` arrays;
- event arrays;
- packet/report state;
- diagnostics and errors.

### Scientific corpus parity

Run Gen1 versus `framefree` across existing packet/network corpora covering:

- direct delivery;
- DATA + ACK;
- contention and collision;
- hidden sender/interference;
- retry and ACK timeout;
- propagation delay;
- energy-denial paths;
- short-frame receiver FIFO after retention prune;
- relay/report streaming;
- report TTL/expiration;
- equal-time ordering cases.

Require exact canonical arrays where existing harnesses support them and exact scientific metrics otherwise.

### Reset and resume

Prove:

- full reset rebuilds derived state correctly;
- partial vector reset affects only selected rows;
- Gen1 live snapshot -> `framefree` restore -> continued steps are exact;
- `framefree` live snapshot -> Gen1 restore -> continued steps are exact;
- no derived bitmap arrays leak into the canonical snapshot schema.

### CUDA gate

Run focused CUDA exactness on T4 before any throughput benchmark. Any CPU/CUDA divergence blocks performance testing.

## Benchmark Gate

Only after all exactness gates pass, benchmark Gen1 versus `framefree` on Tesla T4.

Use the existing scheduler benchmark structure with:

- Scenario V3;
- 2,048 environments;
- 1 MB reports;
- both `active_streaming` and `low_contention` corpora;
- compile/JIT outside the measured region;
- identical seed and action corpus;
- at least three measured repeats per scheduler/corpus;
- scheduler order alternated or otherwise controlled so cache/order effects do not systematically favor one candidate;
- device-wide memory measurement plus derived-state allocation accounting.

Report at minimum:

- collector wall time;
- joint transitions/s;
- packet-step time/fraction;
- scientific metrics;
- simulator error count;
- environment/derived memory;
- device peak memory.

## Promotion Decision

`framefree` advances to a later production-shaped collector/learner gate only if all of the following hold:

1. exact scientific/canonical parity passes;
2. simulator error count is zero;
3. no OOM occurs;
4. median active-streaming throughput improves by at least 2.0% versus Gen1 across the repeated T4 measurements;
5. low-contention throughput does not regress by more than 2.0%;
6. measured derived-state memory is consistent with the approximately 0.8125 MiB design and does not introduce material hidden allocation.

If exactness passes but active-streaming improvement is below 2.0%, record `KEEP GEN1` and stop. Do not spend another production-shaped Kaggle probe on a sub-threshold candidate.

If exactness fails, reject the candidate regardless of speed.

Passing this benchmark does **not** itself promote `framefree` to production. It only authorizes a separate production-shaped gate.

## Rollback and Failure Handling

- Gen1 remains untouched and selectable at all times.
- Scheduler default remains Gen1.
- `framefree` is experimental until a separate production-promotion decision.
- No silent fallback may change frame index, retention order, event order, or canonical cleanup timing.
- Bitmap/canonical inconsistency is a correctness failure, not a condition to approximate around.
- A benchmark failure or small speedup closes the task with `KEEP GEN1`.

## Non-Goals

This milestone does not:

- optimize `next_receive()` or use `rxindex`;
- optimize packet-slot scans in `fill_session()`;
- optimize stale/no-op event pressure;
- change `frame_order` representation;
- change interference accumulation order;
- change report size, payload size, retry count, ranges, PHY settings, propagation, energy, rewards, or observations;
- change MASAC/MATD3 or BASE/PQL training behavior;
- change replay placement, batch size, environment count, AMP, compile, or optimizer settings;
- restart, stop, migrate, or modify the four current BASE/PQL training runs.

## Success Definition

The task succeeds only if the experimental `framefree` path removes Gen1's full 1,632-slot free/reclaim scan from `emit()`, preserves exact Gen1/UavNetSim-derived scientific and canonical behavior across direct, contention, retry, relay, reset, and resume workloads, and demonstrates a reproducible T4 active-streaming improvement of at least 2.0% without meaningful low-contention regression or memory growth.

Otherwise Gen1 remains the correct production implementation and the next independent hotspot is packet-slot scanning or stale/no-op event pressure.
