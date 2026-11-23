# Native M2 – 1000 m – Implementation, tests and science checks

Date: 2026-11-21 | Branch `feature/correlated-cooperative-m2-map-20261121`

## User-selected scope

6 UAV; 35 target cells; 25 cylindrical obstacles; **1000 m × 1000 m only**; grid 25 m, hence 40×40 = 1600 ground cells. Simulation step Δt=1 second, max 1500 steps on the final mission config. M2 implementation is an **opt-in Native CPU reference**. The CUDA Warp 4096-environment backend and convergence training of MATD3/MASAC are deferred until the user provides the Kaggle UI notebook. Legacy RL action dimension 4 remains unchanged when `world.m2_enabled=False`; M2 uses 5 continuous dimensions.

## Decisions preserved

- 1B: sensor/altitude/visible-area conditioned evidence; paper fixed 0.8 is not presented as the final calibrated method.
- 2B: DS creates a **candidate**, not a confirmed true target.
- 3C + Q3A: cooperative sequential likelihood test, nominal α=β=0.05 **per candidate**; no mandatory two-UAV quorum.
- Q1B: correlated errors **between UAVs and across time**, modeled by a shared latent cell bias.
- 4C-1: immutable observed events with ID/provenance, carried through real Native UavNetSim.
- Q4B: versioned correction and retraction, GCS has an explicit unified `consensus_status` including `disputed`.
- 5D: reward target, verification, correct reporting, 2D map improvement, 3D obstacle improvement and energy.
- M2: target belief, obstacle belief, coverage, freshness, candidate/reports, 3D obstacle geometry; each UAV and GCS keeps distinct maps.
- GCS downlink constrained by physical radio reach; UAV multihop forwarding selected by Actor action.
- Actor controls altitude; **no forced descent to 50 m**.

## Flow with equations

1. Generate scene truth `T_c, O_c`, 3D obstacle geometry `(x,y,r,h)`, UAV poses, GCS pose; initialize **all maps** to `(F,E,U)=(0,0,1)`.
2. Actor receives *local/noisy/received* observation and chooses `a_i=(u_x,u_y,u_z,u_kind,u_tx,destination)`; `destination=0` means silence. M2 `u_kind` chooses evidence / geometry / any / map_delta / report / correction; UavNetSim checks actual PHY/MAC connectivity. The underlying MATD3/MASAC action interface now supports 5 dimensions only in M2 opt-in config.
3. Acceleration `a_i = max_accel * u_xyz`, desired velocity `v' = limit(v+a Δt)`, safety `CBF`, position `p'=p+v' Δt`.
4. Flight energy `E_flight=(20+120+0.5||v||²+2||a||²)Δt`; radio TX/RX energy comes from UavNetSim; **the same UAV battery object** pays both costs.
5. FoV footprint `r(h)=h tan(FoV/2)`, `q=area(cell∩FoV)/area(cell)`; occluded/fully unseen target cells are not sampled. Camera scans the full eligible footprint by default, **no hard-coded four-cell cap**; transport packet budget is separate.
6. Interpolate `Pd(h)` and `Pf(h)` over `[0,50,100,150]` m. Occupancy effective likelihood: `p0=1-(1-Pf)^q`, `p1=q Pd+(1-q)p0`. This is a declared partial-FoV model approximation; not Zhang's DS equation.
7. Correlated sensor latent `B_c∈{-1,+1}` fixed per cell and shared by UAVs within the episode; `P(z=1|H_j,B_c=b)=p_j + b ρ min(p_j,1-p_j)`; `z=1[Uniform(0,1)<P]`. This preserves marginal `p_j` but induces cross-UAV/time correlations. `ρ` is **not** the Pearson correlation coefficient.
8. Evidence event `{event_id, origin, tick, cell, layer, observed_z, p1, p0, ρ}`. No simulator true-target ID.
9. Each node computes `L_j=P(z|H_j,history)` through the Bayes posterior of the nuisance sign under each hypothesis. Conditional log-likelihood `ℓ=log(L_1/L_0)`. `ℓ>0` is evidence for occupancy even when `z=0`; `ℓ<0` is evidence for empty even when `z=1`.
10. Reconstruct DS exactly from accumulated **signed** LLR magnitudes `U=sum(max(ℓ,0))`, `V=sum(max(-ℓ,0))`. Unnormalized masses: `F=(1-exp(-U))*exp(-V)`, `E=exp(-U)*(1-exp(-V))`, `Unknown=exp(-U-V)`; evaluate in **log-domain** and normalize using logsumexp. This replaces numerically unstable iterative normalization under long contradictory streams.
11. DS `m_F>=0.95` starts candidate. Cooperative SPRT **starts from zero only on new verification measurements**, but conditions likelihood on prior history; `S_n=sum(new ℓ_k)`. Nominal boundaries `±log(19)≈±2.944`. Status pending/verified/rejected and revisions based on received event set; out-of-order arrivals trigger deterministic replay.
12. Source-packet store batches evidence, geometry detections, reports, map delta and corrections into bounded packets; real pinned **Native UavNetSim** handles radio energy, CSMA/CA, congestion, ACK/PHY, packet loss, multi-hop. Packet not received ≠ map updated. Repeats are discarded by event ID. A sender's observed contact time is only updated after successful transfer. `MAP_DELTA` contains GCS snapshot *and original evidence events*, which the receiver recomputes; snapshots are not fused as independent evidence.
13. Obstacle 3D camera generates **noisy** center XY/radius/height cylinder detections without ground-truth obstacle IDs; node performs uncertainty-gated track association and maps these tracks into voxel z-layers. GCS only sees measurements delivered through packets. 3D map is complete for the simulator's **cylindrical obstacle family**, not arbitrary 3D triangle meshes/photogrammetry.
14. A verified target report has `(source,cell,version,status,tick)`; GCS de-duplicates versions, resolves conflicting source claims to an explicit status, preserves correction history. **Correct GCS delivery requires an actual verified report arriving**, not merely DS map evidence crossing a threshold. Simulator truth is only used in sensor generation/evaluation/reward, not in Actor/Critic observation.
15. Reward (current *prototype* weights): `+.15` per new candidate cell, `±1` per verification verdict depending on simulator truth, `+50` per first correct verified target cell delivered, `-10` per false verified report, plus Brier-map progress, `+2*ΔIoU_3D`, minus flight/RF/step penalties. Reward parameters are not tuned for convergence yet; stage-specific baselines needed.
16. Next observation, centralized observation-only Critic, replay, MATD3/MASAC update. No learned-routing/altitude convergence claim before actual Kaggle run.

## Metrics

**Primary:** GCS verified-report recall (TP distinct delivered /35), false report proportion (N/A if no reports), target-map AUPRC, target-map Brier/ECE, obstacle 3D voxel IoU, GCS coverage, normalized correct-delivery AUC, total energy and energy per correct delivery.

**Diagnostic:** Candidate recall distinct from report recall, verification false-positive/negative rate by sensor height and cross-UAV history, report correction/false-alert exposure, packet PDR, useful evidence bytes, duplicate payload fraction, hop count, end-to-end latency, mean/P95 AoI, number of GCS 3D geometry tracks, safety shield interventions, Actor/critic losses. Use `mission time` inside episode and `total_env_transitions` for multi-run training curves.

## Current test and scientific limitations

- The **feature-focused suite** covers Native correlated sensing, long contradictory DS streams, positive-observation negative LLR, 3D geometry, single shared battery, full FoV altitude scaling, real Native UavNetSim packet delivery with 5D Actor packet category, GCS map delta, versioned correction, report-vs-candidate distinction, and MATD3/MASAC CPU gradient update.
- There is **no proof** Wald nominal 5% guarantees persist after adaptive DS candidate selection, correlated sensing and report revisions; run Monte Carlo calibration by seed and separate alpha/beta/FPR tables before publication.
- GCS sends map delta to in-range UAV in a basic scheduling policy; UAV selects relay, receiver and packet type. GCS itself is not an RL policy.
- GCS still cannot guarantee all UAV eventually receive data with network partitions/TTL; source-provenance dedup prevents double evidence, but unavoidable duplicate **transmissions** can occur on unknown alternate paths.
- The 3D geometry observation model is a parametric noisy range sensor, not validated on optical sensor/camera datasets. Camera image encoder, free-form 3D meshes and dense LiDAR SLAM are NOT implemented.
- Full Warp E4096 environment and MATD3/MASAC convergence are explicitly **deferred**. No Kaggle notebook has been modified or launched.
- The legacy repository has a known set of historical checkpoint/tracking/structure-related test failures. Compare the final full-suite **test names** to `/home/aduy/Documents/NCKH/uav_network_study_1200/outputs/baseline_rootcause_20261120/test_status.json` before calling any failure a regression.
- The new standalone Kaggle API token appended after JSON in `key.txt` was accepted by authenticated `kaggle quota` using an isolated Kaggle 2.2.5 CLI. Credential-file modification was blocked by the runtime safety layer; old Kaggle keys are **not** reported as deleted and the key file requires user-mediated cleanup before code can `json.loads` it.

## Repro commands

```bash
cd /home/aduy/Documents/NCKH/uav_network_coopmap_20261121
python -m pytest -q tests/test_m2_hardening_20261121.py tests/test_m2_mission_native.py tests/test_m2_uavnetsim_integration.py tests/test_correlated_m2_protocol.py tests/test_ds_evidence_map.py tests/test_ds_evidence_warp.py tests/test_network_telemetry.py tests/test_gpu_packets.py tests/test_matd3.py tests/test_masac.py
python scripts/research/run_cooperative_m2_native_demo_20261121.py --steps 30 --seed 44 --output /home/aduy/Downloads/UAV_M2_HARDENED_1000M_20261121
```

**Demo with ground-truth evaluation:** `m2_demo_metrics.json`, `m2_native_1000m_timeline.csv`, `m2_maps_and_truth_evaluation_only.npz`, three PNG figures in `~/Downloads/UAV_M2_HARDENED_1000M_20261121/`. Never put evaluator-only `.npz` truth arrays in Actor observation/replay.

## Quality hardening after final full-suite regression check

- M2 configuration is opt-in in `runtime.m2_enabled` rather than `world`: the latter caused a new `scenario_v3` identity/hash failure. The revised placement restores the exact frozen historic scenario hash and does not change legacy action dimensions.
- Explicit test verifies geometry behind a closer taller cylinder is not detected by the noisy geometric sensor; the implemented top-center ray occlusion is an approximation and not complete partial-surface ray tracing.
- Explicit early guard in `environment/vector.py::make_vector` rejects M2 in *legacy* Native/Warp trainers, because the E4096 integration is intentionally deferred. This prevents silently training the old 4D environment under an M2 configuration.
- Targeted post-guard regression, including `test_scenario_v3.py`: **134 passed, 23 subtests passed**.
- A full-suite run before moving the M2 flag out of frozen `world` yielded `1811 passed, 28 failed, 105 skipped, 1983 subtests passed` in ~477s; 27 failures exactly matched preexisting baseline, the sole new failure `tests/test_scenario_v3.py::ScenarioV3Tests::test_v3_freezes_large_report_radio_study_and_v2_stays_legacy` was corrected and its direct unit test is now green. A second full-suite run after this fix was started; refer to `/tmp/m2_fullsuite_final_clean_20261121.log` for the final summary once finished.

## FINAL COMPLETE FULL-SUITE REGRESSION, 2026-11-21

Following the runtime-only `m2_enabled` flag fix (which eliminated the single new `scenario_v3` test failure), the complete repository test suite `python -m pytest -q tests` finished in 430.46 seconds:

**1814 passed, 27 failed, 105 skipped, 8 warnings, 1984 subtests passed.**

Comparison of every `FAILED`/`SUBFAILED` test identifier against `/home/aduy/Documents/NCKH/uav_network_study_1200/outputs/baseline_rootcause_20261120/test_status.json` returned **exact match of all 27 previous failures; ZERO additional failing test identifiers**. The historical failures include source structure/hash constraints and checkpoint, logging, W&B/tracking, probe and training test issues; these preexisting unrelated failures were not suppressed or incorrectly marked resolved. The M2-focused suite including the historic `scenario_v3` test yielded **134 passed, 23 subtests passed**, and the extracted source ZIP was retested independently with **54 passed**.

User credential note: the newly appended standalone Kaggle token (KGAT_ form, not printed) authenticated via isolated Kaggle 2.2.5 `quota`. Changing/removing other credential entries in `key.txt` was blocked by tool safety; do not claim the old keys were deleted. The owner must clean the JSON credential document manually.
