# Corrected sensor-to-report full flow for MATD3 baseline `18d756ef`

**Date:** 2026-11-22 · **Branch:** `fix/18d756ef-no-oracle-cell-reports-20261122`

The previous run `MATD3-EVAL100-ACTORBN-PLAINCRITIC-E4096-U4-B8192-H256x512-S44-18d756ef` was trained using an historical simulator where the camera used truth to produce measurements **and then** used truth a second time to selectively create target-ID reports and reset falsely confirmed cells to 0.01. This was an oracle-aided perception/report pipeline. It is not scientifically interchangeable with a system which reports its own possibly false detections. This change **does not modify the old run, its model weights, frozen Kaggle notebook, or W&B metrics**.

## Activation and provenance

- Historical reference config, unmodified: `configs/reference-18d756ef-wandb-resolved-20261121.yaml`.
- **New experiment config:** `configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml` with `sensor.report_cell_hypotheses: true` and a distinct run group/output directory.
- Opt-in flag defaults `false` so historical Warp XYZ and legacy XY kernel behavior and old test contracts remain accessible.
- Model training hyperparameters retained from W&B: 6 UAV, 35 targets, 25 obstacles, 1000 m map, 25 m cells (40×40), dt=1 s, horizon 1500, seed44, 4096 vector worlds, four updates per vector step, minibatch8192, replay1M CPU75%/GPU25%, shared Actor BN/plain Critic, hidden [256,512], warmup131072, checkpoint256, validation100 fixed seeds/checkpoint and test4000 separate seeds **in the config**. These are *planned parameters*, not evidence a new GPU run was performed.
- Actor vector remains 228-dim, critic 830-dim for this W&B config, **but semantics have changed**. `observation_version=6`, Warp snapshot `version=3`. Do not warm-start/load the old `18d756ef` policy into the new task even though tensor shapes match.

## Corrected full flow (ground-truth separation)

1. **World reset:** Generate target XY ground truth for the **simulator only**; reject any target placement that would put two different targets in the same 25m map cell. Set local Bayesian occupancy `0.5` as per historical Bayes baseline; true target-ID and cell-hypothesis namespaces are separate. The world holds target count T=35, hypothesized report storage slots S=max(64,4T)=140. The `report_cell` array maps an internal slot to a globally unique observed cell ID; unallocated slots are -1.
2. **Policy action:** MATD3 shared decentralized Actor sees its own position, velocity, battery, local target-belief patch and coarse uncertainty, GCS vector, local report counts, local observable obstacles, and tentative routing destinations. No true target positions; no global report-slot index; no UAV peer position/velocity/battery/buffer state masquerading as free beacon traffic. Critic sees a centralized joint state but no true target coordinates or global obstacle ground-truth geometry in the new mode. A future explicitly modeled and airtime-accounted beacon protocol could reintroduce justified peer telemetry.
3. **3D motion and camera:** `WarpVectorEnv.step_async` applies XYZ bounded acceleration/motion with collision rejection and energy costs, then `sense_bayesian()` samples camera `yes/no` with original FoV/occlusion and altitude-dependent P_D(h),P_F(h). Hidden truth may be consulted **only to simulate a physically plausible camera sample**, and to compute separate evaluation/reward diagnostics; not for confirmation or report creation.
4. **Bayesian belief:** Each UAV independently calculates posterior from its previous local belief and sampled binary measurement. A **false-positive at an empty cell remains a legitimate observation**; no oracle change `posterior -> verified_empty_belief` can occur. Local camera `yes` + altitude<=50m + posterior>=0.99 yields a **candidate (cell ID)**, regardless of actual target presence. Multiple UAV readings of same cell deduplicate by cell ID in `confirm_observed_cells` after the parallel sampling kernel. A dedicated sparse candidate queue avoids scanning all 1,600 cells in the confirmation kernel. Exceeding the finite hypothesis pool is an explicit error, never silently a missing false alarm.
5. **Actor candidate ordering:** `xyz_observations.candidates` ranks previously observed, elevated-posterior local cells by Bayesian confidence, ties by proximity/cell-ID; no longer treats a one-frame `yes` nearest cell as automatically most important. This changes the observation semantic version.
6. **Radio report generation:** `sense_reports_at` sees sensor-derived visible **hypothesis slots**. It creates a report for the observed cell, even for empty ground-truth cells. Slot numbers are internal memory keys. `pkt_cell` and `tx_cell` hold actual serialized, observable 2D cell identity on the packet path and validate cell identity at receipt. FIFO report TTL, physical radio channel, MAC retries, relay custody, GCS receive/ACK behavior are retained. A report's *truth* is not tested before transmission or receiver acceptance.
7. **GCS and scoring:** `s.delivered[e,slot]` records all physical REPORT deliveries including false reports. A separate scoring layer `environment/gpu/scoring_truth.py` maps each reported **cell** to a target's cell for `targets_found` and `targets_delivered` metrics and reward. The scoring truth is consulted only in reward/bookkeeping after a report exists or is received, never in policy observation or the reporting network admission path. A target can receive credit at most once because true target cells are unique; all reported false cells remain in `false_confirmed_cells` and `false_reports_delivered` counters.
8. **Reward:** True local target recognition still earns +20; true GCS target delivery +50. A locally confirmed false cell and a false report delivered to GCS each incur -10 via the pre-existing dropped-report weight. Motion, step, radio/energy and uncertainty shaping components remain as previously configured. Ground truth is allowed in **RL reward computation**, but not as a policy feature or a gating condition for an on-air REPORT. Episodic termination is **all 35 true targets delivered**, all batteries depleted, or 1500 steps. It is not report slot pool exhaustion; the latter is a separate simulation error.
9. **Metrics / leakage:** `world_info` reports truth-rated target discovery/delivery plus false alerts, as well as the raw count of all on-air GCS reports. True report delays and hops have their own accumulators and exclude false reports, while raw network traffic metrics include all packets. `tracking.progress_metrics` cannot label the new local hypotheses proxy as actual target recall, so it no longer logs the old `progress/search_rate_percent` and `progress/delivery_rate_percent` keys. The completed episode metrics still carry actual truth-rated search/delivery. W&B per-episode fields include `outcome/false_confirmed_cells`, `outcome/false_reports_delivered`, `outcome/total_hypotheses`.
10. **Learning/evaluation:** MATD3 core architecture and optimizer implementation remain unchanged. New training starts from scratch on the new MDP. Validation should use exactly the same new cell-report simulator and held-out seeds. Never reuse the old 18d756ef 37.165% discovery, 33.288% GCS delivery final-test values as valid performance under the corrected scenario.

## Exact replacement for the requested code snippet

**Historical branch** (`cell_reports=0`, for reproduction only):

```python
if yes and altitude <= fine_altitude and posterior >= confirmation_threshold:
    for target in true_targets:          # historical oracle gate
        if target_is_in_observed_cell:
            visible[target_id] = 1
```

**Corrected branch** (`cell_reports=1`, used in the new profile):

```python
# Camera sampling already occurred; the following uses NO true target array:
if yes and altitude <= fine_altitude and posterior >= confirmation_threshold:
    confirmed_candidates[uav].append(observed_cell_id)

# Serial per world, deterministic cell ID mapping:
slot = find_or_allocate_report_slot(observed_cell_id)
visible_cell_report[uav,slot] = True
# Real network creates/relays REPORT(payload.cell_id=observed_cell_id)
# Only ground-truth scorer later decides TP/FP, never report creation.
```

## Validation evidence

- Test file: `tests/test_18d756ef_no_oracle_cell_reports_20261122.py` (17+ tests), including deliberately constructed empty-cell false report, true/false identical transmission trigger, fake report over actual Warp MAC/PHY/GCS, true vs false reward and terminal status, shared cell-report slot identity, privacy metamorphic tests, masked reset, distinct snapshot/observation version, candidate ranking, true-only delivery delay accounting and W&B false-alarm metrics.
- Broad old-scene smoke/regression: historical `tests/test_xyz_warp.py`, `test_xyz_radio.py`, `test_gpu_packets.py` and exact M2 parity tests. See actual test logs, do not substitute a planned run for a pass.
- 2-world, 6-UAV/35-target/25-obstacle **Warp CPU smoke** passed reset and 500 random-action steps. The hardware running these tests has **no accessible NVIDIA CUDA driver**, so CUDA T4×2/E4096 compilation, memory and throughput are unverified. The original project branch's full E4096 one-session runner/evaluator remains frozen separately and has **not been fully ported** to this corrected branch. The new profile is validated and ready for further development, not a claim that a new 20,000-episode GPU train/validation/final-test session completed.

## Remaining issues and research decisions

- **Motion saturation / low battery:** The historical policy caused 84.66% battery-exhausted training episodes and many unsafe/rejected flight commands. Replacing the oracle changes the task; a new policy must be retrained and audited. This patch does not pretend to fix a learned behavior by changing a plotted value. Implementing and validating a physically correct CBF/safety controller or action regularization requires a distinct ablation.
- **Correlated sensing:** The unchanged original binary sensor likelihood assumes conditional independence of successive observations for Bayes updating. Repeated near-identical observations may over-concentrate belief. Separate calibration tests with correlated measurements, false reports, precision-recall and reliability diagrams are needed before paper claims.
- **Hypothesis pool:** finite S=140 can signal a clear error when more distinct cell hypotheses are generated in an episode. Do not silently suppress excess false alarms or claim a 1,600-cell fully unbounded report system.
- **Passive peer telemetry:** no explicit beacon packet is sent; this fix conservatively removes free access to remote peer state and chooses tentative routing actions, meaning performance could decline. A network-accounted beacon protocol can be a subsequent controlled extension.
- **Off-policy checkpoint/source:** even identical [228,830] tensor shapes do not establish equivalent MDP semantics. Keep `observation_version=6`, snapshot version3 and distinct tracking run/group.
- **Evaluation baselines:** repeat with at least multiple random seeds and held-out worlds, including a no-sensor-oracle baseline and original-run reproduction, and report the changed task conditions alongside any performance differences.
