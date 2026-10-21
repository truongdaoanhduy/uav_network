# MAC Scheduler Gen2 Design

## Purpose

Speed up the Scenario V3 `warp_packet` communication simulator without changing the scientific experiment. The current production profile remains the authority for mission geometry, sensing, networking, energy, reward, observation/state, MASAC/MATD3, PQL, checkpointing, and evaluation. MAC Scheduler Gen2 changes only how already-defined MAC events are executed on GPU.

The motivating production evidence is that collector time dominates wall time and the event-driven packet scheduler is the bottleneck. On Tesla T4, `packet_step` consumed about 98.7% of a profiled 2,048-environment step, and disabling only the MAC event loop reduced `packet_step` from roughly 14.48 s to 0.47 s. The design must therefore reduce event-loop execution overhead while preserving packet semantics exactly.

## Terminology

- **Scenario V3**: the current large-report research scenario. It is not being replaced.
- **Gen1**: the existing event-heap MAC scheduler in `uav_network/communication/gpu/mac.py` and `events.py`.
- **Gen2**: the proposed semantics-preserving macro-event scheduler.
- **Checkpoint version 3**: the full training-resume format; unrelated to Scenario V3 and Gen2.

## Immutable Scientific Scope

Gen2 must not change any of the following:

- Scenario profile semantics, including 6 UAVs, 35 targets, 25 obstacles, 1,800-step mission horizon, seed 44, and `warp_packet` backend for the current production study.
- Report size 1,000,000 bytes, buffer size 3,000,000 bytes, payload size 1,024 bytes, TTL, retry limits, bandwidth, PHY thresholds, peer/GCS ranges, propagation behavior, or obstacle attenuation.
- CSMA/CA ordering, DIFS/SIFS, contention-window behavior, CCA decisions, backoff draws/freezes, DATA/ACK timing, collisions, retries, packet deadlines, or FIFO ordering.
- RNG sequence and number/order of random draws for equivalent state/action inputs.
- Report custody, fragment/session semantics, relay hop accounting, duplicate handling, report delivery time, and report TTL behavior.
- TX/RX/circuit/PA energy accounting and battery state.
- Reward equations, observation/state definitions, MASAC/MATD3 equations, PQL behavior, replay semantics, AMP, `torch.compile`, or optimizer behavior.
- Checkpoint/resume exactness and CPU visualization behavior.

## Current Bottleneck

Gen1 launches one Warp thread per world for `packet_step`. Inside each thread, the MAC scheduler repeatedly pops the minimum event from a per-world heap, executes it, and potentially pushes new events back into the same heap. Profiling shows roughly 5,000–9,000 actual event-loop iterations per world per mission step in early Scenario V3 execution. Only around 4% are POLL events; LISTEN contributes roughly 21–41%; the majority are DATA/ACK/backoff/TX/RX/retry events. Therefore, optimizing only queue polling or unrelated scans is insufficient.

## Chosen Architecture: Exact Macro-Event Fusion

Gen2 keeps the same event heap and the same event records. It adds a fast execution path that can consume a deterministic chain of events without performing a full heap pop/push cycle between every event, but only when the scheduler can prove that no competing event can become earlier under the exact ordering key:

`(timestamp, priority, insertion_order)`.

The core rule is conservative:

> Gen2 may fuse only a chain whose next event is provably the same event Gen1 would pop next. If that proof is not available, Gen2 immediately returns to the ordinary Gen1 heap path before observable state changes diverge.

This means Gen2 is an execution optimization, not a different network model.

## Fusion Candidates

Initial work is limited to event chains that are local to one packet/node and whose scheduling dependencies are explicit. Candidate chains include:

1. `START -> MAC -> IDLE/BACKOFF progression` when no other event can preempt the next generated event.
2. `TX_START -> DATA_END` when the DATA_END event is provably earlier than every other queued event.
3. Receive/ACK continuations where the next timestamp/priority/order is already known and no competing receiver event can precede it.
4. Self-rescheduling LISTEN/POLL jumps only where existing Gen1 coalescing already establishes an exact boundary.

The first implementation must not fuse across:

- competing transmitters,
- equal-time events whose insertion order matters,
- collision windows with another active frame,
- receiver FIFO competition,
- retry/backoff RNG draws that would move relative to another event,
- report transfer/custody transitions unless exact next-event ordering is proven,
- any capacity/error boundary.

These restrictions are deliberate. Performance comes after parity.

## Component Boundaries

### `events.py`

Owns event ordering and heap mechanics. Gen2 may add read-only helpers that answer whether a proposed event is strictly earlier than the next competitor. These helpers must use the same comparison rule as `earlier()` and must not mutate event order or sequence counters.

### `mac.py`

Owns Gen1 and Gen2 scheduler dispatch. Gen1 remains the reference implementation and fallback. Gen2 should be isolated behind a small internal function rather than rewriting `event()` semantics.

### `phy.py`, `streaming.py`, `reports.py`, `rng.py`

No semantic rewrite is allowed in the first Gen2 milestone. Gen2 calls the same functions used by Gen1. Changes here are permitted only if needed for a pure helper with proven identical behavior and corresponding parity tests.

### Configuration

No new user-facing Scenario V3 parameter is required for the first milestone. Gen2 should be selectable only through an internal/runtime scheduler mode used for A/B verification until parity and performance gates pass. Gen1 remains the default for production until the rollout gate is approved.

## Exactness Contract

For identical config, seed, starting state, and action sequence, Gen1 and Gen2 must match on every completed mission step for all state that can affect future behavior or reported science. At minimum this includes:

- event heap contents and ordering fields,
- event/packet/report sequence counters,
- packet/session/frame state,
- RNG state,
- positions/velocities/battery/energy,
- report custody and delivered/known state,
- packet and diagnostic counters,
- transmission timestamps and trace order,
- reward and termination state,
- observations and centralized critic state.

Where the existing Native CPU network is the oracle, Gen2 must also retain the current Native ↔ Warp parity expectations.

## Testing Strategy

Testing is layered from cheapest to most authoritative.

1. **Event-order unit tests**: prove the Gen2 ordering predicate agrees with Gen1 heap ordering for earlier/later/equal timestamp, priority, and insertion-order cases.
2. **Gen1 ↔ Gen2 step parity tests**: execute the same prebuilt device state/action sequence through both schedulers and compare all future-relevant arrays byte-for-byte where dtype permits exact comparison.
3. **Existing packet oracle tests**: all existing `tests/test_gpu_packets.py` tests must remain green, including direct delivery, relay, collisions, weak links, retries, TTL, large streaming reports, equal-backoff ties, cross-step DATA/ACK, fractional poll intervals, six-UAV contention, and battery edge cases.
4. **Scenario V3 integration parity**: add a 1 MB multi-step Scenario V3 corpus using representative direct, relay, failed-PHY, and contention actions.
5. **Exact-resume parity**: a checkpoint captured after Gen2 execution must restore and continue identically to the uninterrupted Gen2 path; Gen2 must not change checkpoint schema.
6. **CUDA parity**: run Gen1 and Gen2 on Tesla T4 with the same fixed seeds/actions and compare the required state boundary.

Any parity failure blocks performance work from being promoted.

## Performance Measurement

Performance must be measured on the same Tesla T4 class and Scenario V3 parameters that exposed the bottleneck. Benchmarks must separately report:

- `packet_step` wall time,
- full collector wall time,
- joint transitions/s,
- event-loop iterations or macro-fusion counts,
- fallback count and reason,
- GPU memory allocated/reserved,
- error/capacity counters.

The benchmark corpus must include both low-contention and active-streaming states because early idle performance alone is not representative.

## Promotion Gates

Gen2 is eligible to replace Gen1 for a future production restart only if all gates pass:

1. **Scientific parity**: no known Gen1/Gen2 state or metric divergence on the required parity suite.
2. **Existing regression suite**: full project tests pass.
3. **CUDA exactness**: fixed-seed Tesla T4 parity passes.
4. **No new OOM/capacity failures** at 2,048 environments with the production replay/batch profile.
5. **Meaningful sustained speedup**: target at least 300 joint transitions/s on the representative Scenario V3 collector benchmark, or otherwise demonstrate through measured episode-length/throughput projections that the 20,000-episode target can fit the available Kaggle quota. A small 10–20% gain is not sufficient justification for a production restart.
6. **Rollback safety**: Gen1 remains available and selectable until a Gen2 production lineage has passed post-warmup monitoring.

Passing microbenchmarks alone is insufficient.

## Rollout Strategy

1. Develop Gen2 on an isolated branch/worktree from production SHA `6de210f44c6c7da70c94aaa9bd629be9e45efcd7`.
2. Do not modify or stop the four currently running Scenario V3 production jobs while Gen2 is experimental.
3. Run local CPU/Warp parity first.
4. Run bounded T4 CUDA parity and benchmark jobs on spare Kaggle accounts.
5. If all gates pass, freeze a new commit and run one bounded production-shaped smoke test before deciding whether to restart/continue the four scientific lineages on Gen2.
6. If performance remains insufficient or parity requires model simplification, reject Gen2 rather than silently changing Scenario V3 semantics.

## Observability

Gen2 should expose diagnostic counters for development and validation only:

- macro chains attempted,
- macro chains completed,
- events fused,
- fallbacks,
- fallback reason categories,
- maximum fused chain length.

These counters must be passive: enabling or disabling them must not affect event ordering, RNG, energy, or packet results.

## Failure Handling

- If a Gen2 proof condition fails, use Gen1 from the current heap state before consuming the uncertain event.
- If an existing capacity/error condition occurs, preserve the same error code and stopping behavior as Gen1.
- If a parity test exposes any divergence, minimize the failing event sequence and add it to the regression corpus before changing Gen2.
- Three unsuccessful optimization hypotheses without a parity-safe speedup trigger an architectural review instead of accumulating speculative patches.

## Non-Goals

This milestone does not:

- reduce report size or fragment count,
- change payload size,
- approximate CSMA/CA,
- remove ACK/retry/collision behavior,
- change the 1-second mission step,
- move to a simpler network backend,
- change MASAC/MATD3/PQL,
- alter reward weights,
- change the Scenario V3 mission definition,
- promise that Gen2 will be adopted if it cannot meet both parity and performance gates.

## Alternatives Rejected for This Milestone

- **Polling-only optimization**: measured POLL events are only about 4% of actual event-loop iterations; insufficient.
- **Small queue-scan optimization**: a tested `node_queue_count` substitution preserved parity but produced no measurable T4 speedup, so it was discarded.
- **Disable telemetry**: profiling showed essentially unchanged `packet_step` time; not the bottleneck.
- **Fully parallel packet scheduler**: potentially faster but much higher risk to exact SimPy ordering, RNG order, and collision semantics. Revisit only if conservative macro-event fusion cannot reach the performance gate.
- **Simplify the network model**: scientifically invalid for the current Scenario V3 comparison.

## Success Definition

The project succeeds only if Gen2 makes the V3 collector materially faster while producing the same scientific trajectory as Gen1 for the verified corpus. If exact parity and sufficient speed cannot both be demonstrated, the correct result is to keep Gen1 and report the feasibility limit rather than change the experiment silently.
