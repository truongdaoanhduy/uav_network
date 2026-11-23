# HELLO over-the-air / 2.5D / AoI / Zeng2019 — scientific ablation

**Status:** implemented in Warp CPU simulator; no T4x2 execution or measured 20k-episode results. **Not** a replay/parity optimization to apply silently to an existing checkpoint.

**Parent:** `experiment/map1000-25d-zeng-energy-ablation-20261123` (9445e41).
**Opt-in config:** `configs/matd3-nooracle-map1000-25d-aoi-zeng2019-hello1hz-fresh2s-20261123.yaml`.
**Default config and earlier baseline branches stay unchanged.** New model changes the policy observation semantics (version 12 vs 11), routing mask, RF energy use and packet contention. Retrain from scratch (20,000 episodes, seed 44) and compare with 3–5 additional seeds.

## Exact protocol and non-oracle causal direction

1. Every active UAV independently advertises an **ACK-free broadcast** `HELLO` of **512 bits once per second**, staggered by node ID only for initial timing. It uses the **same event heap** as packet queues, transmitter-side CSMA carrier sensing, `DIFS + random [0,31] slot` backoff, a bounded 50 ms busy-channel deferral and the same RF-power/interference/pathloss/LOS check as data. It is **not** yet byte-compatible with the original upstream Python UavNetSim HELLO class, because Warp has a deterministic, packet-level radio replica; protocol equivalence is a separate test.
2. One 512-bit RF waveform is independently decoded by each potential recipient; no ACK is expected for broadcast. A failed decode, interference, NLOS or distance beyond peer range NEVER refreshes the receiver's cache. Interfering beacon frames share their airtime with actual DATA and ACK frames. Transmitter and receiver radio circuit/PA energy are debited from the same UAV battery used for flight.
3. The payload is copied at **source emission time**: source XY, planar velocity, altitude, vertical velocity, battery fraction, free-buffer fraction. On *successful* reception only, store the payload in `hello_*[world,receiving_UAV,source_UAV]` plus receiver timestamp. Store each source's in-flight payload at `[world, source_UAV]` to avoid a very large per-frame copy.
4. In the **Actor**, no peer positions/velocities/battery/link budget are read from the simulator world state. A remote peer's state, **normalized HELLO freshness (sixth peer feature)**, progress toward the GCS, free-buffer estimate, and destination mask are filled only from a successfully received, **age<=2 s** cached HELLO. All peer fields and route choices are zero before the first reception or after expiry. The fixed GCS coordinate remains a publicly known landmark; PHY still uses actual geometry, as it should.
5. Critic/scorer may retain privileged information in the *centralized state*, never copied into the decentralized Actor's neighbor fields. Own source has no oracle knowledge of remote radio admission. Data ACK/retransmission/custody/TTL remain the previous implementation; beam is a separate frame type 2 and does not inflate DATA PDR.
6. Added radio diagnostics `hello_tx_frames`, `hello_rx_frames`, `hello_decode_failures`, `hello_success_percent`, and `hello_enabled` to `network_info` / end-of-episode telemetry.

## Physics and scientific limits

- Same previous 2.5D: map 1000×1000m, 6 UAVs, altitude 35–130m, |v_z|<=4m/s, posterior-only 40×40 Bayesian belief/Pd-Pf, report TTL 300s, learned TX ceiling0.6W, energy budget277200J, Zeng2019 illustrative model. Default safety=`reject`; **CBF remains an independent opt-in control ablation**, not secretly enabled.
- HELLO TX power currently fixed to configured `tx_power_max_w` (0.6 W), not a learned beacon-power action. Beacon is a contention-aware ACK-free broadcast, not full per-destination data queue retry/ACK; use this as an explicit model assumption.
- Source-side scheduling avoids self-transmit overlap, but a pathological MAC boundary with more than one concurrent receiver and an ACK queue is still not fully proven equivalent to Python UavNetSim. CUDA byte-for-byte parity and actual RTX/T4 scheduling are NOT confirmed. Thus report simulations as **Warp PHY/MAC**, not as physically validated 802.11 hardware.
- In-current-turn test evidence: seed44 two-peer receive/expiry/cache, blocked NLOS, out-of-range, packet delivery GCS with concurrent HELLO, radio-energy debit, reset, deterministic repeat, independent config and backward-compat regression. CPU checks cannot prove CUDA memory limits, GPU throughput, 20k×5 seeds or training improvement.

## Fair comparison and go/no-go

Use identical target/obstacle placements and seeds for Zeng2019 **HELLO OFF** versus **HELLO ON**, train from scratch with different W&B names, and track (1) unique target confirmations/deliveries, (2) report delay and freshness, (3) false reports, (4) TX energy, (5) beacon receive percentage and stale masks, (6) wall-clock steps/sec, and (7) checkpoint held-out Val100 and final Test1000. Report median/IQR across independent seeds rather than a one-seed curve. Keep old baseline 59042708 strictly separate (its Bayesian prior, RF reward and energy are not comparable to this MDP).
