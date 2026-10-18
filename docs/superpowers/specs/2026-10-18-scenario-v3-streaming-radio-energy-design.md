# Scenario V3: Streaming Reports and Energy-Aware Relay Design

## Goal

Make the existing 2D UAV swarm scenario faithfully exercise the research objective: UAVs search for targets, create large reports, choose direct-GCS or peer relays, choose transmit power, and trade search/delivery quality against physical propulsion/radio energy without changing MASAC or MATD3 architectures.

## Constraints

- Preserve Scenario V2 and every currently running V2 job unchanged.
- Build V3 from commit `4c75c5cdae7f300c844ffe0bb9b20c89db4cd603` on a separate branch.
- Keep the common mission/training baseline: 6 UAVs, 35 targets, 25 obstacles, 1000 m map, 1800 steps, seed 44, 4096 training envs, 25,000 episode budget, 2 Mbps channel, 500 m peer/GCS range, 0.001-0.4 W log-mapped TX action, urban LOS/NLOS model, collision/interference, max retries 5, 277,200 J/UAV battery.
- Do not change MASAC/MATD3 network architectures or give either algorithm privileged rules.
- Receiver selection and TX power remain learned actions. Environment logic may mask physically impossible destinations but must not replace policy choice with nearest-GCS routing or automatic minimum power.
- Native UavNetSim and Warp packet backends must retain matched mission semantics.
- Test-first development is required for behavior changes.

## V3 Workload

V3 uses:

- `report_bytes: 1000000`
- `buffer_bytes: 3000000`
- `payload_bytes: 1024`
- `max_queue_packets: 64`
- `report_ttl_s: 300`
- `bit_rate_bps: 2000000`

A 1 MB report therefore contains roughly 977 DATA fragments. The implementation must not allocate dense `[environment, session, fragment]` state proportional to all fragments for every possible session. Queue capacity is a flow-control window, not the report size.

## Streaming Fragment Architecture

### Session state

Each active transfer session stores O(1) report progress rather than one state cell per fragment. At minimum it needs enough state to identify the transfer and determine which report byte/fragment is next to generate and how many fragments/bytes have arrived.

Packet slots continue to carry fragment indices/offsets so retries and out-of-order completion can be associated with the transfer. The bounded packet queue remains the materialized window of in-flight work.

### Refill behavior

Starting a transfer fills available packet slots up to `max_queue_packets`. Whenever a queued fragment reaches a terminal state (success, retry-limit drop, or packet lifetime expiry), the scheduler may materialize more not-yet-completed fragments for that same live session during the same network interval. A 1 MB transfer must therefore make progress continuously within a one-second environment step rather than being artificially limited to 64 KiB per RL action.

A complete report changes custody only after all report bytes/fragments for the session have arrived. A failed fragment remains eligible for a later packet generation while the report/session TTL is alive. Existing DATA/ACK CSMA/CA, collision, interference, retry and TTL rules remain authoritative.

### Memory requirement

Warp allocation must be O(envs * (sessions + packet_capacity + frame_capacity + event_capacity)), not O(envs * sessions * fragments). Increasing `report_bytes` must not linearly multiply per-session state allocation.

## Physical Radio Energy

Add configurable radio hardware accounting while preserving legacy behavior by default:

- `pa_efficiency` in `(0, 1]`, default `1.0`
- `tx_circuit_power_w >= 0`, default `0.0`
- `rx_circuit_power_w >= 0`, default `0.0`

For DATA or ACK airtime `t` and selected RF output `P_rf`:

`E_tx = (tx_circuit_power_w + P_rf / pa_efficiency) * t`

A receiver that listens to/receives a frame is charged:

`E_rx = rx_circuit_power_w * t`

Failed transmissions consume transmitter energy because energy is charged at emission. Receiver energy is charged for frames actually presented to that receiver by the PHY model, including unsuccessfully decoded frames when the receiver had to listen for the frame. GCS energy is tracked separately and does not reduce UAV battery. UAV TX/RX radio energy reduces the same battery used by propulsion.

Legacy configs with defaults reproduce the previous RF-output-only accounting.

Scenario V3 must explicitly select nonzero radio-hardware parameters only if they are documented as study assumptions in the config/design notes. They are not to be tuned per algorithm.

## Reward Semantics

Keep physical battery accounting separate from reward shaping.

- Existing `communication_energy` continues to penalize physical UAV radio joules.
- Existing normalized `communication_efficiency` continues to make radio efficiency learnable even when propulsion dominates physical energy.
- `propulsion_energy_per_kj` remains `0.01` in the first V3 baseline so workload/physics changes are isolated.
- Do not add a power-margin reward to the first V3 baseline. Power margin remains diagnostic. A later V3.1 ablation may add a symmetric underpower/excess-power shaping term only after a short V3 baseline demonstrates the same under/over-power pathologies.

## Observations and Routing

Keep the existing V2 link-state observation: per candidate receiver required-power fraction, progress toward GCS, receiver free-buffer fraction, battery/relative kinematics, and destination-admission mask. The policy therefore has enough information to learn direct-vs-relay and power control without a rule-based shortcut.

Continue to record:

- direct vs relayed GCS reports
- hop histogram / mean hops
- relay-toward-GCS fraction and GCS-distance improvement
- selected TX power, required TX power and dB margin
- below-required-power fraction
- LOS/NLOS, peer/GCS PHY success
- retries/collisions/interference/weak-signal failures
- report expiration/drop/discovered-undelivered
- propulsion energy, UAV TX energy, UAV RX energy, total UAV radio energy
- joules per delivered report and radio joules per delivered report

## Scenario V3 Config

Create `configs/comparison-25k-scenario-v3.yaml` rather than mutating V2. V3 inherits V2 mission/training values and changes only the report/buffer workload plus any radio-hardware assumptions required by this design. W&B group/tags must say `scenario-v3` so V2 and V3 cannot be confused.

## Verification

Required automated verification:

1. Legacy 1 KiB configs retain previous behavior when new radio-hardware parameters are at defaults.
2. A 1 MB report can be delivered direct-to-GCS with queue capacity 64 without queue-size-proportional truncation.
3. A 1 MB report can complete through two-hop and three-hop custody chains.
4. Native and Warp CPU backends agree on report completion, hop counts, packet generation/receipt semantics and energy accounting in deterministic fixtures.
5. Failed TX attempts consume TX energy; failed decodes consume RX listening energy where applicable.
6. GCS radio energy is reported but never drains UAV battery.
7. Warp state shapes for session progress do not contain a dimension equal to total report fragment count.
8. Existing V2 tests continue to pass unchanged.
9. A resource estimator/benchmark reports state memory for 1, 64, 512 and 4096 envs without constructing an unbounded dense fragment tensor. CUDA execution is validated separately when GPU is available.

## Training Gate

Do not launch fresh 25k V3 training until the full local suite, native/Warp parity fixtures, 1 MB direct/multihop tests, and GPU resource validation pass. First training should be a short smoke/diagnostic run for MASAC and MATD3 on the same V3 config. Only after those diagnostics should a power-margin reward ablation or full four-way 25k comparison be considered.
