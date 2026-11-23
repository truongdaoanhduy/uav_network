# Final technical audit: no-oracle MATD3 cell-report Warp scenario

**Scope**: branch `fix/18d756ef-no-oracle-cell-reports-20261122`, configuration
`configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml`, 2026-11-22.
This is a code/test audit, **NOT** a successful E4096 T4×2 GPU training run.
The historical Kaggle/W&B run `18d756ef` was neither overwritten nor retrained.

## End-to-end component audit

| Component | Status / verified behavior |
|---|---|
| Scene generation | 1000m map, 25m cells (40×40), 6 UAV, 35 unique target cells, 25 obstacles; tested across varied seeds |
| 3D flight/energy | 1s step, speed/acceleration & battery accounting; `motion_safety=reject` is **not** an active CBF or DWA trajectory optimizer |
| Camera | FoV=90°, altitude-dependent Pd/Pf anchors; binary samples conditional on true visibility, belief integrates marginal partial-cell likelihoods |
| Bayesian belief | Independent local occupancy per UAV, initial prior 0.5, posterior updates; no oracle reset to 0.01 in this new mode |
| Candidate/confirmation | `yes`, altitude<=50m, posterior>=.99 creates observed-cell hypothesis even when the cell is actually empty |
| Decentralized Actor | No remote XYZ, battery, vertical velocity, queue or global report slot order available for free; observed local belief and report aggregates only |
| Centralized Critic | 830D; true target geometry removed. Version-7 report summary hashes every report into 35 fixed spatial buckets; bucket collisions lose detail |
| REPORT identity | Network report has original cell identity `pkt_cell`/`tx_cell`, rather than true target ID; physical queue, TTL, retries, ACK, custody still required |
| Peer RF path | No-beacon peer actions now attempt transmission without oracle prefilter on remote knowledge or distance; PHY/MAC decides outcome and accounts energy |
| GCS | Only actually delivered reports counted. False reports can reach GCS but do not earn true target credit |
| Reward | +20 for unique true discovery, +50 for unique true GCS delivery, -10 false local confirmation and -10 false GCS delivery; true/false separated by evaluator-only ground truth |
| Termination | All 35 true targets delivered, all UAV batteries depleted, or 1500 steps; done worlds must be selectively reset before subsequent action |
| Training settings | From pinned 18d756ef config: E4096, U4, B8192, replay 1M, CPU75%/GPU25%, shared Actor BN/normal Critic, hidden256×512, seed44 |
| Validation/final test | Config declares 100 periodic eval episodes and 4000 final test episodes, but neither has been run on the newly corrected MDP |

## Confirmed bugs, reproductions and fixes

1. **Free remote altitude/vertical velocity in Actor**: `tests/test_18d756ef_final_scenario_audit_20261122.py::test_peer_altitude_and_vertical_velocity_not_free_observation` failed; fixed in `gpu/xyz_observations.py` for new mode only.
2. **Arbitrary report-slot allocation affected Critic**: previously a valid delivery stored in slot63 of a 64-slot store changed none of the critic's report features; moving the same cell into slot0 changed multiple features. Fixed with `gpu/cell_report_state.py` spatial bucket projection over all S slots. Caution: collisions still lose resolution.
3. **True report latency contaminated by false reports**: comparison metric had used the raw network average `report_e2e_delay_s` despite dividing energy/reward by true deliveries. Added `true_report_e2e_delay_s`, kept raw network metrics separately, updated paper comparison path and W&B tracker.
4. **Unobservable MAC preflight**: source attempted no packet and spent zero TX energy when the selected peer was out of range or already had the report, despite no modeled beacon. No-beacon mode now allows an actual RF attempt, and PHY/MAC charges TX and determines success. Historical mode remains unchanged.
5. **Checkpoint semantic compatibility**: bumped `observation_version` from 6 to **7** and Warp snapshot from 3 to **4**, plus distinct W&B group and output directory. Tensor dimensions remain the same but policies from old versions are incompatible.

The user-identified hidden true-target-ID confirmation was fixed in preceding
commit `f452e3b`; the final audit validates it is still absent from report
generation, PHY admission, actor and critic observations.

## Tests and stress evidence

- Reproductions above were run RED before fixes; post-fix targeted tests GREEN.
- **Final release regression on v7/snapshot4 and optimized no-beacon MAC:** **187 passed, 10 subtests passed**. Covers Warp XYZ sensor/motion, radio packets, original model parity, false-report delivery, truthful metrics, tracked training, and all five final adversarial audit cases.
- Supplemental M2/CBF/safety tests: **100 passed / 1 skipped / 2 subtests**. Passing Native M2 tests does not prove the 2D DS map is in the corrected Warp scenario.
- **Final v7 / optimized-MAC stress:** 24 Warp CPU worlds, each with 6 UAV, 35 target and 25 obstacles, stepped for a complete 1500-tick horizon with selective reset on `done`. **24/24 episodes terminated correctly** (9 battery exhausted, 15 time limit), no report-pool exhaustion or device capacity errors; ~28.8s CPU wall time. These were random actions, not a trained MATD3 policy. The earlier `error 92` was only the stress harness stepping an already terminated world.
- A 256-world partial-FoV sensor calibration at 50m with a uniformly distributed target in a partially visible cell gave q≈0.376, marginal model P(yes|occupied)≈0.363 and sampled yes≈0.340. This is consistent within finite-sample noise. **Do not** blindly replace conditional Pd with marginal p1 when sampling a known-visible true target: that would discount FoV twice.
- CPU stress became materially slower when MAC stopped skipping out-of-range peer sends: ~17.2s with the old shortcut versus ~48.4s with unoptimized physical peer probes over 24 worlds×1500 steps. The final optimized no-beacon MAC completed the same 24-world×1500-step random-action test in **~28.8s** by removing unused preflight link checks. It still consumes real channel airtime on failed peer attempts. CUDA/T4 throughput is untested.

## Remaining research/engineering limitations (NOT claimed fixed)

- **No full DS in this Warp path:** Bayesian posterior and `altitude<=50m` confirmation; no DS m(F)>=.99, no per-UAV DS source fusion, no GCS belief-map merging/broadcast/corrections. Those features are in the separate Native M2 prototype.
- **Bounded hypotheses:** 140 report slots per world versus 1600 grid cells, with an explicit error if exhausted. Investigate long-run FP-heavy settings; do not silently ignore reports.
- **Lossy Critic compression:** 35 hashed spatial buckets cannot encode all 140 reports uniquely. Broader state encoding would alter the pinned critic input size and should be an explicit ablation.
- **No active peer discovery beacon:** Actor lacks true peer position/route progress; failed peer transmissions cost energy. A real beacon protocol with measured airtime is a new research feature.
- **No Bayesian report retraction:** Once confirmed, a later negative posterior does not retract an already delivered cell report. This can leave stale false alarms at the GCS.
- **Temporal sensor correlation:** repeated measurements are not decorrelated; posterior .99 is model-based, not proof of an empirical 1% false-report rate.
- **Coverage counts touched grid cells:** any nonzero visible fraction q marks the cell visited; this is NOT continuous FoV area coverage.
- **Safety control is motion reject**, not DWA/CBF as the active production no-oracle config. Independent flight-controller analysis is needed to avoid corner saturation and wasted energy.
- **CUDA, E4096 throughput, replay CPU/GPU split performance, checkpoint restore across long Kaggle sessions, model convergence, 100-scenario validation and 4000-scenario final test remain unverified for this new MDP.**
