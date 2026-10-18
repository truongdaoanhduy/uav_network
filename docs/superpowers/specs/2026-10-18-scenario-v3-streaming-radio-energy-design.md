# Scenario V3: Streaming Reports and Radio-Energy-Aware Relay

## Intent

Create a publication-grade Scenario V3 for the 2D homogeneous UAV swarm in which each UAV must search for targets, create realistically sized reports, choose whether to send directly to GCS or relay through peers, choose transmit power, and jointly minimize mission delay, packet loss, propulsion energy, and communication energy. V3 must preserve MASAC/MATD3 algorithm definitions so any difference is attributable to environment/network formulation rather than algorithm-specific changes.

Scenario V2 remains immutable and its existing/running jobs remain valid baselines.

## Success criteria

1. Native and Warp backends implement identical report-transfer semantics for 1 MB reports without allocating per-session dense state proportional to all report fragments.
2. A 1 MB report is fragmented and streamed through a bounded per-UAV packet queue; queue refill happens automatically while a transfer is active rather than requiring repeated RL actions merely to enqueue the next chunk.
3. End-to-end custody moves only after the complete report has been received and reassembled.
4. Failed transmissions consume transmitter energy. Radio energy includes a configurable power-amplifier term and TX/RX circuitry energy; battery accounting is consistent between Native and Warp.
5. RL retains control of receiver choice and transmit-power action. The environment must not replace the learned power with the required power.
6. Actor observations continue to expose receiver link-state information, including required-power fraction, relay progress toward GCS, receiver buffer headroom, and admissibility.
7. Metrics make it possible to distinguish good power control from underpower/overpower: selected TX power, required power, margin dB, below-required attempts, retry cost, direct/relayed deliveries, hop count, relay progress, RF J/report, propulsion J/report, remaining battery, expired/dropped reports, PHY/packet success, and safety-shield activity.
8. V3 passes deterministic CPU tests, Native/Warp parity tests, direct/2-hop/3-hop 1 MB transfer tests, failure-energy tests, and GPU memory/capacity checks before long training.
9. Only after validation, fresh MASAC BASE/PQL and MATD3 BASE/PQL runs start from episode 0 on the same V3 commit/config/seed.

## Non-goals

- Do not change MASAC or MATD3 network architecture, hidden sizes, target-update rules, exploration mechanism, or optimization equations as part of V3 infrastructure work.
- Do not encode a rule that always picks the peer closest to GCS.
- Do not clamp learned TX power upward to required power.
- Do not increase battery or max mission duration solely to manufacture a higher success rate.
- Do not terminate or mutate currently running V2 jobs.

## Baseline invariants

Keep unless a feasibility study later proves the mission impossible:

- UAVs: 6
- Targets: 35
- Obstacles: 25
- Map: 1000 m square
- Episode horizon: 1800 s, dt=1 s
- Battery: 277,200 J/UAV
- Peer/GCS admission range: 500 m
- TX action range: 0.001--0.4 W, logarithmic mapping
- PHY rate: 2 Mbps
- Payload: 1024 B
- Max retries: 5
- LOS/NLOS urban channel, collision/interference, propagation delay
- Seed: 44; disjoint evaluation seed unchanged

## V3 workload

Create a new configuration instead of modifying V2:

- `network.report_bytes = 1_000_000`
- `network.buffer_bytes = 3_000_000`
- `network.payload_bytes = 1024`
- retain a bounded packet queue (initially 64)
- retain 300 s report TTL initially

This models roughly one megabyte of sensed target data per report while allowing each UAV to hold roughly three reports.

## Streaming fragmentation architecture

### Problem with the V2 representation

The Warp backend currently sizes session fragment arrays by:

`num_envs x session_capacity x fragments_per_report`.

At 1 MB / 1024 B, each report has about 977 fragments. With six UAVs, 35 targets and seven possible destinations, dense state per world would grow prohibitively large, especially at 4096 environments.

### Selected design: bounded streaming window

Represent transfer progress per session using O(1) metadata instead of O(fragment_count) arrays. Per active session maintain the minimum state needed to resume a report transfer, such as:

- session generation/id
- origin/source/target/destination
- report creation time and hop count
- next fragment/byte offset to schedule
- number or bytes of unique fragments successfully received
- bounded bookkeeping for only fragments currently queued/in-flight/retrying

Packets remain represented in the existing bounded per-world packet pool. A session feeds new fragments whenever the source queue has capacity. The number of materialized packet objects is therefore bounded by queue and packet-pool capacity, not report size.

### Automatic refill

When a packet completes, is acknowledged, expires, or exhausts retries, the MAC scheduler must re-evaluate the active session and feed subsequent unsent fragments while queue space is available. One RL action selects a destination and power for a transfer attempt/session; the simulator then serializes the report over the network rather than requiring one environment action per queue window.

A report changes custody only after all report bytes/fragments have been uniquely reassembled. Duplicate/retried packets must not double-count received bytes.

### Session changes

If the report disappears, expires, changes custody, or the action selects a different transfer before completion according to the agreed session semantics, stale packets must be invalidated by generation/session identifiers so late arrivals cannot complete the wrong report.

## Native/Warp parity

Implement the same streaming semantics in both backends. The Native implementation already lazily enqueues until `max_queue_packets`; it should be refactored to explicit session refill so it does not rely on a subsequent RL step for progress. Warp should mirror this behavior.

Parity tests must compare, for deterministic micro-scenarios:

- generated/received payload bytes
- complete custody transfer time
- retry and collision accounting
- packet/PHY success counts
- selected and required power telemetry
- RF energy debit at source and receiver
- delivery hop count and report delay

## Radio energy model

### Existing issue

V2 mostly charges radiated TX energy `P_rf * airtime`; with 1 KB reports this is negligible relative to propulsion. That prevents transmit-power choice from materially affecting battery use.

### V3 model

Introduce configurable parameters for a physically interpretable radio energy model:

- `pa_efficiency` in (0, 1]
- `tx_circuit_power_w >= 0`
- `rx_circuit_power_w >= 0`
- optionally an ACK/control circuit mode using the same model

For a DATA transmission of duration `t` and selected radiated power `P_rf`:

`E_tx = (tx_circuit_power_w + P_rf / pa_efficiency) * t`

For a receiver processing the frame for duration `t`:

`E_rx = rx_circuit_power_w * t`

TX energy is charged on every real transmission attempt, including failed PHY attempts. RX energy is charged for the defined receive/listen semantics consistently in both backends. GCS energy may remain separately reported and excluded from UAV battery objectives.

The V3 shared study defaults are `pa_efficiency=0.35`, `tx_circuit_power_w=0.1`, and `rx_circuit_power_w=0.1`. The model form is consistent with UAV-to-UAV energy-efficiency work that separates radiated power divided by PA efficiency from circuit/system power. For the numeric study assumptions, relay energy-efficiency literature reports PA efficiency 0.35 and practical circuit power in the tens-to-hundreds-of-mW range (Journal on Wireless Communications and Networking 2012:46, DOI 10.1186/1687-1499-2012-46); 0.1 W TX and RX are deliberately shared in-range values, not a calibrated hardware claim. Publication text must retain that distinction.

## Reward formulation

Keep mission rewards and learned power action. First evaluate V3 with the existing normalized communication-efficiency term after the workload and energy model are correct.

Do not immediately add algorithm-specific shaping. If short V3 training still shows systematic under-power or excessive over-power, add a shared, algorithm-independent margin-aware penalty in a later controlled revision/ablation. Desired behavior is a small positive link margin, not minimum possible power and not maximum power.

Propulsion weighting should also be handled as a shared ablation rather than silently changed in the baseline. A candidate energy-aware ablation may compare the existing `propulsion_energy_per_kj=0.01` against a stronger shared coefficient such as 0.05, but only after V3 infrastructure is validated.

## Observation and action contract

Do not change the semantic action contract:

- movement x/y
- destination: silent / peer / GCS
- continuous TX-power action

Retain link-state observation fields already implemented for each peer/GCS:

- normalized nominal required power
- progress toward GCS
- receiver buffer free fraction
- destination admissibility
- peer battery and relative motion/position

Any observation layout change required by radio-energy additions must be shared by MASAC and MATD3 and covered by checkpoint incompatibility/versioning tests.

## Telemetry additions

At minimum publish pooled and per-episode metrics for:

- DATA TX selected power mean/distribution
- nominal required power mean/distribution
- power margin dB mean/distribution
- below-required-power attempt fraction
- excessive-margin fraction using a configured diagnostic threshold
- RF TX energy and RX energy, per UAV and total
- RF J per successfully delivered report/MB
- propulsion J per delivered report
- total UAV J per delivered report
- attempts/retries/collisions per delivered report
- direct vs relayed GCS reports
- custody peer transfers and relay-toward-GCS fraction
- relay distance improvement
- delivery hop histogram
- report generation/discovery-to-delivery delay
- expired/dropped/discovered-undelivered
- battery remaining and exhausted UAVs
- shield intervention/infeasible/commanded-stall metrics

These diagnostics are for evaluation and debugging; they must not leak privileged information into actor observations unless explicitly part of the observation contract.

## Safety and motion

Keep the existing dynamic-window/safety-shield implementation and the already-fixed peer-safe obstacle tangent logic. V3 networking changes must not be bundled with new navigation behavior. Continue to record shield intervention because high intervention means the raw policy still relies heavily on the safety layer.

## Feasibility controller

Before long MARL training, create a deterministic heuristic/oracle evaluation that is allowed to use sensible search and relay behavior. Its purpose is not to become the deployed policy; it tests whether 6 UAVs, 35 targets, 25 obstacles, 1800 s, and 277.2 kJ/UAV can plausibly complete the V3 mission.

Only if this controller shows the mission is structurally infeasible should battery, target count, range or horizon be reconsidered.

## Validation plan

### Unit tests

- 1 MB fragment count and final partial-fragment correctness
- bounded queue never exceeds configured limit
- refill continues without an additional RL action
- retries do not duplicate reassembly bytes
- stale session packets cannot transfer custody
- failed TX consumes energy
- TX energy increases monotonically with radiated power for equal airtime
- RX energy is charged exactly once per modeled receive attempt/frame
- battery denial prevents unaffordable transmissions

### Integration tests

- one-hop direct GCS 1 MB delivery
- two-hop and three-hop relay delivery
- forced weak direct link where peer relay succeeds
- forced under-power produces lower success / retry behavior
- Native/Warp deterministic parity for transfer, energy and diagnostics
- TTL expiration during partial transfer
- source or relay battery depletion during partial transfer

### Performance gates

Benchmark environment memory and throughput at 1, 64, 512, then 4096 environments. Do not launch 25k training if V3 memory growth depends on full report fragment count or if packet/event capacities become infeasible on the target Kaggle GPU.

## Training rollout after validation

1. Preserve all V2 results as historical baseline.
2. Run short V3 smoke/training for MASAC and MATD3 to inspect search, delivery, under-power, excess margin, relay direction, RF/propulsion split, shield behavior and battery.
3. If shared environment behavior is healthy, launch fresh matched runs from episode zero:
   - MASAC BASE
   - MASAC PQL
   - MATD3 BASE
   - MATD3 PQL
4. Same source commit, config, seed, episode budget and evaluation protocol for all four.
5. Use final 30-episode disjoint CPU evaluation and videos for publication comparison.

## Compatibility

Scenario V3 must use a new config/group/tag and, where state layout changes, an explicit checkpoint/schema version. V2 checkpoints must not be silently resumed into incompatible V3 state. Existing V2 jobs and artifacts stay untouched.
