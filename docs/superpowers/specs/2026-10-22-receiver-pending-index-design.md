# Receiver Pending Index Design

## Purpose

Speed up the Scenario V3 `warp_packet` network simulator while preserving the scientific behavior of the pinned UavNetSim-based implementation exactly. Task 4 showed that the highest-value current hotspot is not macro-event dispatch but repeated fixed-capacity frame-table scanning, especially the `next_receive()` scan that searches all `frame_capacity=1632` frame slots whenever a receiver must select the next pending frame.

This milestone changes only how pending receive candidates are indexed on GPU. It does not change the network model, event timestamps, event priorities, packet/ACK behavior, PHY equations, RNG, report semantics, or MARL algorithms.

## Evidence and Motivation

On Tesla T4 with Scenario V3, 2,048 environments, four measured active-streaming steps, and 1 MB reports, Task 4 measured:

- 25,509,191 total root-event iterations.
- 5,637,819 estimated/observed `next_receive()` calls.
- `frame_capacity = 1632`.
- About 9,200,920,608 frame-slot iterations attributable to `next_receive()` alone.
- About 4,603,051,104 additional frame-slot iterations from `emit()` frame recycling/allocation scans.
- Baseline versus profiler scientific metrics were equal and simulator errors were zero.

The first optimization therefore targets only `next_receive()`. `emit()` free-frame/retention scanning is intentionally deferred to a later task so performance and correctness effects remain attributable to one change.

## Terminology

- **Gen1**: the existing exact event-heap scheduler and network behavior. It remains the oracle and production default.
- **Gen2**: the prior macro-event experiment. It is not promoted and is unrelated to this index design.
- **rxindex**: the new experimental scheduler mode that preserves Gen1 event semantics but replaces only receiver-pending frame selection with a derived GPU index.
- **Canonical state**: arrays in `WorldState`, including frame, event, packet, RNG, report, energy, diagnostic, observation, and reward state. Canonical state is authoritative for snapshots and scientific behavior.
- **Derived acceleration state**: the receiver pending index. It can be rebuilt exactly from canonical state and is not itself scientific state.

## Immutable UavNetSim Semantics

`rxindex` must not change any of the following:

- CSMA/CA behavior, DIFS/SIFS, contention windows, CCA decisions, backoff draws/freezes, or retry limits.
- DATA/ACK timestamps, ACK timeout behavior, packet deadlines, event priority, insertion order, or event heap ordering.
- Propagation delay, path loss, LoS/NLoS classification, obstacle attenuation, SINR, interference accumulation, collision outcomes, or PHY thresholds.
- TX power mapping, TX/RX/circuit/PA energy, battery state, or energy-denial behavior.
- RNG sequence or number/order of random draws.
- Packet/session generation, fragment windows, report custody, relay hops, TTL, duplicate handling, or report delivery timing.
- Reward, observation, centralized state, MASAC, MATD3, PQL, replay, AMP, `torch.compile`, optimizer behavior, or training checkpoints.
- Native UavNetSim parity expectations already covered by packet tests.

A speedup with any semantic divergence is a failed candidate.

## Chosen Architecture

### Experimental Scheduler Mode

Add internal scheduler mode `rxindex` beside `gen1` and `gen2`.

- `gen1` remains the default and retains the existing `event()` plus `next_receive()` scan path unchanged.
- `rxindex` uses the same packet-step preamble, event heap, fast-root logic, event ordering, PHY functions, report functions, and packet-step finish logic as Gen1.
- Only receive-side dispatch for `DELIVER`, `RECEIVE`, and `RECEIVE_DONE` is specialized so it can maintain and consume the derived receiver pending index.
- All other event kinds delegate to the existing Gen1 `event()` implementation.

This isolates the optimization and provides immediate rollback by selecting Gen1.

### Receiver Pending Heap

Maintain one fixed-capacity min-heap per `(environment, receiver)` containing frame indices whose canonical state satisfies:

- `tx_active != 0`, and
- `tx_destination == receiver`, and
- `tx_receive == 1` (successful PHY delivery, pending serialized receive service).

The heap key is exactly:

`(tx_receive_order, frame_index)`

This reproduces the current scan behavior. Existing `next_receive()` scans frame indices in ascending order and replaces the selected frame only when it finds a strictly smaller `tx_receive_order`; therefore equal orders are resolved by the smaller frame index. The heap must preserve that same tie break.

The heap stores frame indices only. It reads `tx_receive_order` from canonical `WorldState` during comparisons, avoiding duplicate scientific ordering state.

### Heap Memory

For `E` environments, `R = n + 1` receivers, and frame capacity `F`, allocate:

- `frames`: `int32[E, R, F]`
- `sizes`: `int32[E, R]`

For Scenario V3 at 2,048 environments, `R=7` and `F=1632`, the frame-index storage is approximately 93.6 MB plus negligible size metadata. At 4,096 environments it is approximately 187 MB. This memory is allocated only for `rxindex`; Gen1 memory is unchanged.

The T4 promotion gate must include measured total allocated memory and OOM/capacity checks.

## Exact Receive Lifecycle

### Successful `DELIVER`

Gen1 currently:

1. evaluates PHY;
2. sets `tx_receive = 1`;
3. sets `tx_receive_order = event_sequence`;
4. calls `next_receive()`;
5. if receiver idle, scan chooses the minimum pending frame, sets `tx_receive = 2`, marks `node_receiving = 1`, and pushes `RECEIVE` at the same `now`.

`rxindex` must perform the same scientific operations in the same order, with only this substitution:

1. after setting canonical pending state/order, push the frame index into the receiver heap;
2. if receiver is idle, pop the exact heap minimum;
3. set canonical `tx_receive = 2`, set `node_receiving = 1`, and call the same event-heap `push()` for `RECEIVE` at the same timestamp/priority/generation as Gen1.

Heap operations must never modify `event_sequence`, RNG state, energy, diagnostics, packet state, or frame state except for the same `tx_receive`/`node_receiving` updates Gen1 already performs.

### `RECEIVE` and `RECEIVE_DONE`

Whenever Gen1 clears `node_receiving` and calls `next_receive()`, `rxindex` clears the same canonical state and calls the indexed equivalent. The next pending frame must be exactly the same frame Gen1 would choose by full scan.

### Invalid/Inactive Receiver Paths

If a selected frame is rejected because the destination is inactive or TTL is invalid, the canonical frame transition remains unchanged and the next pending candidate is selected from the heap exactly where Gen1 currently calls `next_receive()`.

## Index Consistency

### Normal Execution

A frame is inserted exactly once when canonical state transitions from PHY-delivered (`tx_receive=0`) to pending (`tx_receive=1`). It is removed when selected for serialized receive service (`tx_receive=1 -> 2`). Frames in `tx_receive=2` are not stored in the heap. Frames in `tx_receive=3` are terminal and are not stored.

Pending frames retained outside `frame_order` by PHY retention remain valid index members, matching Gen1 because `next_receive()` considers any `tx_active != 0`, including `tx_active == 2`.

### Reset

`reset_network()` remains canonical. For `rxindex`, a separate acceleration reset kernel clears heap sizes only for environments selected by the reset mask. Heap payload contents need not be zeroed when size is zero.

Partial reset must leave non-reset environments' receiver indices unchanged.

### Snapshot and Restore

The receiver index is not stored in `WorldState` and therefore must not appear in snapshot version 1.

On `restore_state()` into an `rxindex` vector:

1. validate and restore canonical snapshot arrays exactly as today;
2. clear the derived heap sizes;
3. deterministically scan canonical frame state and insert every frame satisfying the pending predicate into its destination heap using the exact `(tx_receive_order, frame_index)` comparator;
4. do not modify canonical arrays while rebuilding.

This permits:

- `rxindex -> snapshot -> rxindex restore`, and
- `gen1 snapshot -> rxindex restore`

without changing checkpoint schema.

The rebuilt heap must not contain the currently serviced frame (`tx_receive == 2`) because its `RECEIVE` event is already represented in the canonical event heap.

## Component Boundaries

### `uav_network/communication/gpu/receiver_index.py` (new)

Owns derived receiver-index types and operations:

- fixed-capacity heap storage;
- exact comparator using canonical `tx_receive_order` then frame index;
- push/pop-min operations;
- reset-by-mask kernel;
- rebuild-from-canonical-state kernel;
- indexed `next_receive` helper that performs the same canonical state/event push as Gen1 after choosing the minimum.

It must not contain PHY, report, reward, or routing logic.

### `uav_network/communication/gpu/mac.py`

Keep Gen1 `event()` and `next_receive()` unchanged as the reference path.

Add the minimal `rxindex` event dispatch needed for `DELIVER`, `RECEIVE`, and `RECEIVE_DONE`. All non-receive-side event kinds delegate to Gen1 `event()`.

Add `run_event_loop_rxindex`, `packet_step_world_rxindex`, and `packet_step_rxindex` while reusing the exact Gen1 event heap and fast-root implementation.

### `uav_network/environment/gpu/vector.py`

- Accept internal scheduler mode `rxindex`.
- Allocate receiver-index arrays only for that mode.
- Launch `packet_step_rxindex` only for that mode.
- Clear derived index on masked reset.
- Rebuild derived index after canonical `restore_state()`.
- Exclude derived index from snapshot arrays by keeping it outside `WorldState`.
- Include derived index memory in `allocated_bytes` so benchmark memory reporting is honest.

### `uav_network/experiments/mac_scheduler_benchmark.py`

Allow `rxindex` as a benchmark scheduler and time `packet_step_rxindex` as the selected packet kernel. Existing Gen1/Gen2 measurements must remain unchanged.

### `phy.py`, `reports.py`, `streaming.py`, `rng.py`, reward/algorithm code

No semantic changes in Task 5.

## Exactness Contract

For identical config, seed, canonical starting state, and action sequence, Gen1 and `rxindex` must match at completed mission-step boundaries on every canonical simulator array.

Required exactness includes:

- event heap contents, event sequence, packet sequence, report sequence;
- all packet/frame/session arrays;
- `tx_receive`, `tx_receive_order`, `node_receiving`;
- MT19937 and geometry RNG state;
- PHY diagnostic counts/stats and transmission trace timestamps;
- battery, TX/RX/communication energy;
- report custody, hops, delivery state/times;
- reward, termination, observation, and centralized state.

The derived heap itself is not compared to Gen1 because Gen1 has no such state; instead tests validate that its minimum and reconstructed contents correspond exactly to canonical pending frames.

## Testing Strategy

Testing proceeds cheapest-first and any failure blocks CUDA/performance work.

1. **Heap unit tests on CPU Warp**
   - order by `tx_receive_order`;
   - equal-order tie chooses smaller frame index;
   - push/pop sequence preserves exact minimum;
   - independent receivers do not interfere;
   - maximum legal pending count fits fixed capacity.

2. **Rebuild tests**
   - rebuild inserts exactly canonical `tx_receive == 1` frames with `tx_active != 0`;
   - excludes `tx_receive == 0`, `2`, and `3`;
   - includes `tx_active == 2` pending frames;
   - rebuild does not mutate canonical arrays.

3. **Gen1 ↔ rxindex exact parity**
   - direct delivery;
   - two simultaneous/equal-order pending frames;
   - receiver busy with queued pending frames;
   - inactive receiver/TTL rejection;
   - cross-step DATA/ACK;
   - weak link and retry;
   - hidden sender/interference/collision;
   - energy denial;
   - 1 MB direct streaming;
   - 2-hop and 3-hop relay.

4. **Native UavNetSim oracle suite**
   Existing packet tests must remain green for Gen1. Where the current test harness supports scheduler selection, add `rxindex` comparisons without weakening the native oracle.

5. **Partial reset**
   Reset a subset of environments and prove the reset rows have empty derived heaps while untouched rows continue bit-exactly.

6. **Snapshot/restore**
   - snapshot schema keys remain identical to Gen1;
   - no receiver-index arrays appear in snapshot;
   - `rxindex -> restore -> continue` matches uninterrupted execution;
   - `gen1 snapshot -> rxindex restore -> continue` matches Gen1 continuation.

7. **Full local regression**
   Full project suite must pass.

8. **CUDA exactness on Tesla T4**
   Active 1 MB fixed-seed corpus must compare every canonical snapshot array Gen1 versus `rxindex` with exact tolerances used by current CUDA parity tests.

9. **T4 throughput benchmark**
   Scenario V3, 2,048 envs, low-contention and active-streaming corpora, repeated measurements. Report packet-step wall time, collector throughput, memory, errors, and scientific metric equality.

## Performance and Promotion Gates

Task 5 is successful only if correctness and speed both pass.

### Correctness Gates

- All local heap/rebuild/parity/resume tests pass.
- Existing native UavNetSim packet oracle tests remain green.
- Full project suite passes.
- T4 canonical-state exactness passes.
- Simulator errors and unexpected terminations remain zero in benchmark corpus.

### Performance Gates

- `rxindex` must be faster than Gen1 on repeated active-streaming 1 MB measurements; a one-off gain is insufficient.
- Low-contention performance must not regress materially.
- Added receiver-index memory must fit the T4 2,048-env benchmark without OOM and be reported explicitly.
- Promotion to production is not automatic even if Task 5 passes; production restart requires a later production-shaped collector/training gate because synthetic active-streaming throughput does not equal full training throughput.

If exactness passes but speed does not, keep Gen1 and record the negative result.

## Rollback and Failure Handling

- Gen1 remains unchanged and selectable at all times.
- `rxindex` is experimental and not the default.
- If derived index consistency is uncertain during development, fail the parity test; do not silently approximate ordering.
- No fallback is allowed that changes event ordering or skips canonical receive state transitions.
- A candidate with parity failure is rejected before T4 benchmark.
- A candidate with reproducibly slower T4 active-streaming throughput is not promoted.

## Non-Goals

Task 5 does not:

- optimize `emit()` free-frame scanning;
- change `frame_order` representation or interference summation order;
- remove stale LISTEN or cancelled ACK_TIMEOUT roots;
- change report size, fragment size, retry count, ranges, propagation model, PHY thresholds, or energy equations;
- simplify or approximate UavNetSim MAC/PHY behavior;
- change Scenario V3, MASAC, MATD3, PQL, reward, observations, or checkpoints;
- restart, modify, or migrate current production training runs.

## Future Follow-up if Task 5 Passes

If receiver indexing is exact and measurably faster, the next independent optimization candidate is `emit()` frame recycling/free-slot selection. That later work must preserve `frame_order` retention order and PHY interference summation order exactly; it is deliberately excluded from this milestone.

## Success Definition

Task 5 succeeds when an experimental `rxindex` path removes the full-frame scan from receiver pending selection, retains exact Gen1/UavNetSim canonical behavior across direct, contention, retry, relay, reset, and resume corpora, and demonstrates a reproducible Tesla T4 active-streaming speedup without unacceptable memory growth. Otherwise Gen1 remains the correct production implementation.
