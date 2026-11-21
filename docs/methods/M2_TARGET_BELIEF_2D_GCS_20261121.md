# M2 2D Target Belief Map – GCS-first multi-hop synchronization (2026-11-21)

**Scope selected by researcher:** 1000 m × 1000 m; 40 × 40 ground target cells with 25 m resolution; 6 UAV; 35 stationary targets; 25 physical cylindrical obstacles for *flight safety and optical occlusion only*. UAV flight remains 3D with Actor-controlled altitude. **There is no reconstructed 3D obstacle geometry, voxel map, SLAM, separate obstacle occupancy map or obstacle-sensing event stream.** Both local UAV and GCS maps hold only three 2D DS target belief masses `F`, `E`, `U` and metadata (coverage, observation timestamp, report status).

## Source paper and precise boundary

Boquan Zhang et al., “Enhancing Multi-UAV Reconnaissance and Search Through Double Critic DDPG With Belief Probability Maps,” *IEEE Transactions on Intelligent Vehicles* 9(2), 2024, DOI: 10.1109/TIV.2024.3352581. Source PDF provided by the user and verified at original pages 4–7.

- Eq. (3)–(5): `Θ={F,E}`, each target cell begins with `m(F)=m(E)=0, m(U)=1` and satisfies `m(F)+m(E)+m(U)=1`.
- Eq. (6)–(13): Dempster's mass update for positive or negative evidence (paper experiments use `mf=me=.8`).
- Eq. (14)–(17): the author's global count fusion takes `MAX_i(N_i+)`, `MAX_i(N_i-)` under **assumed real-time communication**; the paper does *not* implement real UavNetSim, GCS relay, ACK, map delta or per-packet loss.
- Eq. (18): minimize `Σ_c m_c(U)` and maximize `Σ_c 1[m_c(F)≥τ]`. In paper threshold alone declares target. Here `m(F)≥.95` creates a *candidate*, and cooperative correlated-likelihood SPRT generates verified/rejected/retracted report revisions. Ground-truth-correct GCS delivery is evaluated as a **third mission objective**, not misattributed to Zhang et al.
- The paper flies at fixed altitude and senses one projection cell per step. Here UAVs select altitude 0–150 m and scan multiple ground cells inside FoV, with `Pd(h),Pf(h)` and partial-FoV coverage.

## Ground truth / sensing / map / network flow

0. Environment: static positions for `T_c∈{0,1}` in 1600 target cells. Simulator also generates obstacle geometry `(x,y,r,height)`, visible to the physics collision and occlusion functions, **never sent as a map to UAV or GCS**. No truth occupancy array in Actor/Critic state.
1. Actor `(u_x,u_y,u_z,u_kind,u_TX,destination)` chooses XYZ acceleration and height adjustment, one communication category and TX power, plus neighbor/GCS destination. Five continuous dimensions and a categorical next-hop destination; `u_kind` is currently quantized by thresholds and needs a proper categorical-policy ablation during later Kaggle training.
2. Motion: `a_xyz=a_max u_xyz`, `v' = clip_speed(v+a_xyz Δt)`, `p'=p+v' Δt`, CBF safety, per-step shared-battery `E_flight=(20+120+0.5||v||²+2||a||²)Δt` plus RF energy consumed in UavNetSim.
3. FoV footprint: `r(h)=h tan(FoV/2)`. For ground cell `c`, `q_c = intersection_area(c,FoV)/cell_area`; occluded/`q=0` cells have **no observation**, not a negative observation. Altitude-dependent `Pd/Pf` interpolated from calibrated anchors. Effective cell measurement likelihood: `p0=1-(1-Pf)^q`, `p1=q Pd+(1-q)p0` (explicit simulator approximation, *not* from Zhang).
4. Correlated binary sensor: latent shared `B_c∈{−1,+1}` per cell; conditional Bernoulli `θ_j(B_c)=p_j+B_c ρ min(p_j,1−p_j)` for `H_j∈{empty,target}`; `z=1[r<θ]`, `r~Uniform(0,1)`. A single cell's latent sign is shared by all UAVs and times, while random draws are conditional-independent.
5. Event ID `(episode/source/tick/cell)` with target layer only, original `p1,p0,ρ,z` and no target truth. Each node stores source-provenanced events and replay-deduplicates by event ID. Conditional log-LR: `ℓ_k=log P(z_k|H1,history)/P(z_k|H0,history)`. `ℓ>0` adds target evidence and `ℓ<0` empty evidence; *sign may differ from raw positive/negative sensor reading*.
6. Let `A=Σ max(ℓ,0)` and `B=Σ max(−ℓ,0)` per cell, `D=e^−A+e^−B−e^−A−B`. The stable 2D DS F/E/U masses are `mF=(1−e^−A)e^−B/D`, `mE=(1−e^−B)e^−A/D`, `mU=e^−A−B/D`, evaluated in log-domain. This is *an explicit correlated-likelihood extension* of paper's fixed-mass DS, not a verbatim reproduction.
7. Candidate `mF≥.95`; verification uses only post-candidate conditional LLRs (while conditioning latent noise on prior evidence): `S_n=Σ_{verification} ℓ_k`, upper/lower bounds `log((1−β)/α)` and `log(β/(1−α))`, for α=β=.05 approximately ±2.944. Correct target reports require a **verified notice delivered as packet** to GCS; candidate mass alone is not successful delivery.
8. `UAV → UAV → ... → GCS`: Native UavNetSim CSMA/CA with radio reach, TTL, bounded packet buffers, power, ACK, packet fragmentation, loss and delay. Receiving an evidence packet causes GCS to replay unique events into the target-only 2D F/E/U map; no direct memory copies. GCS can receive versioned corrections/retractions and preserves conflicting source claims for consensus resolution.
9. `GCS → in-range UAV → peer relays`: a GCS-produced `MAP_DELTA` packages source evidence events plus latest fused 2D target F/E/U snapshot per included cell; fragments stay within 1024-byte message budget. Its snapshot is *synchronization metadata*, **not independent evidence**. Receiving UAV recomputes fusion from unique original evidence IDs, so it combines the received information with its own observations without double counting. Multi-cell deltas and periodic resend after TTL protect against permanently missing old updates. GCS downlink is not an impossible global broadcast; the next UAV must physically receive it first.
10. Reward and state: `r ≈ +.15 new candidate + verified bonus (+1 or −1 by evaluator truth) +50 first correctly delivered verified target −10 false verified report + 4Δmean_U_GCS + Δmean_U_local − flight/RF/step costs`. Reward truth use is legal (simulator), but no target truth enters the policy observation. Evaluator computes Brier, AUPRC, total unknown, mean unknown, target coverage, report correctness, GCS delivery AUC and energy; **no obstacle 3D/voxel metrics**.
11. After `T` steps evaluate a GCS 2D target belief map against the simulator's true 2D target occupancy, for **offline evaluation only**. This is *target-belief mapping*, **not** 2D obstacle SLAM or 3D reconstruction.

## Closest related work, role, and degree of overlap

| Source | Shared goal | Different scenario / limitation | Public code found? |
|---|---|---|---|
| Zhang et al., IEEE TIV 2024 (DOI above) | 2D DS target belief F/E/U, explore/search | Full communication assumption; fixed altitude; threshold = detection; no packet-level GCS | No verified official source URL found |
| Xue et al., *Electronics* 15(9), 1875, Apr 2026, DOI 10.3390/electronics15091875 | DS + multi-UAV target search + spatial communication blockage + AoI + MARL | Explicit freshness-weighted belief fusion, PPO/MAPPO/QMIX; no proof that they implement same per-packet verified-report/downlink protocol | No verified public code; paper data-availability says inquiries via corresponding author |
| Dou et al., *Drones* 9(12), 855, 2025, DOI 10.3390/drones9120855 | Constrained communications, target existence probability, uncertain sensor Pd/Pf, information fusion | Prioritized target assignment, event-triggered cluster fusion, not our correlated DS + SPRT pipeline | Authors explicitly state simulation code cannot currently be publicly released |
| He et al., *Ad Hoc Networks* 164, 103602, Nov 2024, DOI 10.1016/j.adhoc.2024.103602 | MU/RU multi-hop toward GCS, joint path/TX/energy MARL | Coverage and network rate optimization, no three-channel DS target map or correlated verification | No verified public code URL found |
| Yanmaz, Bettstetter, Güven, *IEEE TVT* 2024 DOI 10.1109/TVT.2024.3363840 | Multi-target search, network connectivity and reports to GCS | Connectivity and adaptive route planning, not source-provenanced DS target-belief delta | No verified public code URL found |

A complementary *Remote Sensing* 2023 paper “Multi-UAV Mapping and Target Finding in Large, Complex, Partially Observable Environments” includes a **real GitHub link** `https://github.com/ViWalkerDev/NanoMap/tree/mdpi-planning-and-control-2023`, but that code performs **3D occupancy mapping** and therefore **should not be taken as a direct baseline for this target-only 2D map**.

## Scientific caveats before publication

- A. The 2026 AoI-DS paper materially overlaps DS+communication-aware multi-UAV search. Do not claim those ingredients are novel. Proposed distinction to test is **provenance-safe correlated evidence, cooperative verification and verified report delivery + bidirectional GCS-to-UAV 2D map synchronization over actual packet constraints**, not a generic “first DS multi-UAV search system”.
- B. Nominal SPRT α/β values need Monte-Carlo calibration across seeds, adaptive candidate selection and spatial/time-correlated sensors, as formal 5% error guarantees do not automatically transfer.
- C. Actor's packet-type output is currently a *thresholded continuous* action, not a differentiable categorical branch in MATD3/MASAC. Correctness of packet category implementation has unit+radio tests; learnability, convergence and policy efficiency require later Kaggle ablation.
- D. The safety CBF shield uses simulator-level obstacle geometry, an **oracle safety shield assumption**. It is not legitimate to claim obstacle avoidance was learned solely from local target-belief maps. Flight may remain 3D; eliminating obstacle mapping does not eliminate safety physics.
- E. GCS can only send to a reachable UAV. Timely full-network synchronization is not guaranteed with partitions, finite buffers and TTL. The stream has bounded packet sizes, multi-cell deltas, duplicate event IDs and periodic resend; measure GCS/UAV map divergence and AoI rather than assuming identical maps at each step.
- F. A 30-step scripted diagnostic is **not an RL-trained result** and should not be reported as a MATD3/MASAC comparison. CUDA Warp E4096 and learned training convergence are explicitly deferred.
- G. A 1000×1000m world, 25m/cell contains 1600 cells with 35 positive targets (≈2.19%); ordinary classification accuracy is misleading; target AUPRC and verified GCS recall matter more.

## Tests and execution

```bash
cd /home/aduy/Documents/NCKH/uav_network_m2_belief2d_20261121
python -m pytest -q tests/test_m2_target_belief2d_20261121.py tests/test_m2_hardening_20261121.py tests/test_m2_mission_native.py tests/test_m2_uavnetsim_integration.py tests/test_correlated_m2_protocol.py
python scripts/research/run_cooperative_m2_native_demo_20261121.py --steps 30 --seed 44
```

All output artifacts under `~/Downloads/UAV_M2_TARGET_BELIEF_2D_1000M_20261121/`. The `.npz` ground-truth map is for evaluation only and must not be fed into Actor/Critic/communication. `key.txt` must not be included in source snapshots. As observed when inspecting metadata only, account1 is enabled and its token matches the trailing duplicate token, but the trailing standalone text makes the complete file invalid JSON; user should remove the redundant last line manually without exposing credentials in a notebook.

## Last 2D safety-net refinements

- `GCS.consensus_status(cell)` returns **`disputed`** when source claims disagree, even if the GCS DS map has a strong local verified estimate. A local fused verdict is not sufficient justification to silently erase an independent contradictory report. Versioned correction/retraction can resolve the dispute.
- `GCS → UAV` prioritizes correction and verified report packets before ordinary bulk target map delta; when no report is eligible it sends the multi-cell `MAP_DELTA`. Original-author filtering checks immutable message provenance only, not the recipient's private buffers; a GCS must not send a source UAV its own already-authored report.
- The old 3D-geometry-specific unit tests were replaced with 2D source-provenance, radio reach, real GCS→peer→peer relay, target-only map metrics, 1024-byte multi-cell delta and expiry/retry tests. **No test was marked skip to hide a regression.**
