# M2 paper-grounded DS detector — current flow (2026-11-21)

## Source, scope and explicit differences

Primary PDF supplied by researcher: Boquan Zhang et al., **Enhancing Multi-UAV Reconnaissance and Search Through Double Critic DDPG With Belief Probability Maps**, *IEEE Transactions on Intelligent Vehicles*, 9(2), 3827–3842 (2024), DOI **10.1109/TIV.2024.3352581**.

- Zhang Eq. (3)–(5): frame `Theta={F,E}` and per-cell masses `m(F),m(E),m(U)`, initially `(0,0,1)`.
- Zhang Eq. (9): positive detection BPA `m_b(F)=mf`, `m_b(E)=0`, `m_b(U)=1-mf`.
- Zhang Eq. (10): negative detection BPA `m_b(E)=me`, `m_b(F)=0`, `m_b(U)=1-me`.
- Zhang Eq. (6)–(13): normalized DS fusion. In the paper's **Model Performance Analysis**, `mf=me=0.8`, target threshold `tau=0.95`. Table III explains a 2-count excess of positive evidence is sufficient.
- Zhang Eq. (14)–(17): each source UAV maintains `N_i(+)`, `N_i(-)`; global counts are **MAX across sources** (`N+(c)=max_i N_i+(c)` and `N-(c)=max_i N_i-(c)`, *not* sums across UAVs). Source counts are preserved in M2 by immutable event IDs and `origin`; radio duplicate/retransmit never increments a source count twice.
- Zhang Eq. (18): a target is **identified directly** if `m(F)>=tau`; **no SPRT** is required to validate that mass threshold. In the paper the UAVs are assumed to communicate in real time; packet-level network, GCS report delivery and map downlink are **M2 extensions**, not claimed as being present in the source paper.

**Current M2 research baseline: `confirmation_mode='ds_threshold'`, `evidence_mode='zhang_max_counts'`, `paper_mf=paper_me=0.8`, `ds_threshold=0.95`, `correlation_strength=0`.** `CooperativeM2Env` uses these defaults. The old correlated-sensor/SPRT implementation is retained *only* as explicit `(confirmation_mode='sprt', evidence_mode='correlated_llr')` ablation (nondefault), and has no claimed 5% end-to-end false-positive guarantee. Unsupported mode cross-combinations are rejected at construction.

Define `f=(1-mf)^{N+(c)}`, `e=(1-me)^{N-(c)}`; stable DS fusion equivalent to paper Eq. (11)–(17):

```
D = f + e - f*e
m(F) = (1-f)*e/D
m(E) = (1-e)*f/D
m(U) = f*e/D
```

All mass totals are computed via stable log-domain `stable_mass_from_llr_totals`, which prevents underflow/overflow on long contradictory sequences. Two positive sightings by **the same UAV** and zero negatives yield `(mF,mE,mU)=(0.96,0,0.04)` and direct `verified` detection. One positive sighting each from two UAVs yields only `mF=0.8` because of paper Eq. (16) MAX fusion. Positive/negative evidence can decrease `mF`; if `mF` falls under threshold after a transmitted report, M2 sends a versioned `retracted` **network consistency update**. This retraction protocol is an M2 transport extension, **not** an SPRT statistical guarantee and **not** a result claimed by Zhang et al.

The source paper assumes a same-cell, simple sensor and realtime inter-UAV communication at fixed altitude. In contrast M2 retains a **3D flight**, altitude/partial-FOV-dependent Bernoulli sensor, packet loss and ACK, GCS-centered fusion, MARL actions, radio energy, and physical-obstacle collision/occlusion. These are modifications, not exact replications. Baseline sensor samples are independent (`rho=0`). Historical `.55` correlated sensor experiments **must** opt into the separate SPRT profile or explicitly be labeled as a new stress-test model, not 'original Zhang'. The paper's `mf=.8` is a DS BPA parameter and should not be confused with physical sensor `Pd(h)` or with 80% true detection probability at every altitude.

The frozen original W&B reference `18d756ef` and its `configs/m2-target2d-exact-18d756ef-20261121.yaml` retain their **original legacy** `sensor.confirmation_threshold=.99` for a config-parity test; **this legacy field does not control the new M2 DS threshold**. `CooperativeM2Env(ds_threshold=.95)` is authoritative for M2; the paper-source fixed BPA and mode are constructor arguments, separately recorded in M2 `info()`, native demo and pilot manifest.

## Complete algorithmic sequence of ONE Native M2 timestep

0. `reset(seed)`: draw world of 1000 m × 1000 m, 6 3D UAVs, 35 stationary ground targets in distinct 25 m target cells, 25 physical 3D obstacles and a GCS; initialize 40×40=1600 target cell masses `(0,0,1)` on **each of 6 UAVs and GCS**; create independent binary sensor (`rho=0`) and Native UavNetSim network. Reset physical states, battery, source counts, event IDs, map status, packet buffers and network clocks; run an initial local sensor sweep.
1. `actor`: each UAV reads a **local 255D observation**: normalized self/GCS/energy/time, local 5×5 DS target patch, report statuses, local outbound packet counts, last *received* radio contacts and 7 destination-mask bits; no true target occupancy, remote secret buffers or global obstacle map. Decentralized MATD3 Actor chooses 5 continuous actions `[accel_x,accel_y,accel_z,packet_category,TX_power]`, one discrete destination index 0 (silent), 1..5 (ordered other UAVs), 6 (GCS).
2. `3D motion`: integrate bounded acceleration (2m/s²) and speed (20m/s), enforce world/height constraints and selected safety (`reject` in the separate prepared M2 YAML; `cbf` in Native pilot), charge propulsion energy `E=(20+120+0.5*v²+2*a²)dt`. Physical obstacles are for safety and camera occlusion, **NOT an obstacle target map**.
3. `sensor`: at every flying UAV, FoV90° ground projection and obstacle LOS determine candidate 25 m cells. Physical Bernoulli observation has altitude-interpolated Pd/Pf and partial-cell coverage; occluded/unseen cells contribute NO negative observations. Default `rho=0` means independent conditional draws. Ground truth enters only sensor simulator/evaluator/reward, never Actor, Critic, report payload or DS map.
4. `event`: assign immutable `(origin UAV, cell, tick, event_id, positive/negative, sensor likelihood metadata)`; increment source-specific positive/negative count for each unique event. For every changed cell compute Zhang `N+=MAX_i N_i+`, `N-=MAX_i N_i-`, DS `F/E/U` using mf=me=.8; duplicates and packet replays do not count twice. A standalone local map can update immediately on *own* observations; other nodes update only after full packet reception.
5. `decision`: if local `mF>=.95`, **immediately** set `verified` and emit a versioned REPORT. No `candidate -> pending -> SPRT -> verified` step. If later `mF` drops below .95, emit a new revision `retracted` as a network message to cancel stale reports; this state revision is not a new statistical detector.
6. `uplink`: Native UavNetSim with CSMA/CA, 1024-byte message cap, bounded FIFO, per-destination/channel constraints, ACK, retry and TTL. The policy's destination / packet category / transmit power control which packet is attempted (evidence, report, correction, map delta) and who receives it. A source can route UAV→UAV→…→GCS, with physical range/SINR and packet loss. The source timestamp does NOT determine the TTL of a newly minted delayed-relay REPORT: its actual creation time does.
7. `GCS`: on radio-packet completion, decode & deduplicate `event_id` or `DecisionNotice(source,cell,revision,status)`. GCS reconstructs its own `F/E/U` using source maxima. A **REPORT packet received at GCS** is necessary for verified delivery metrics: GCS map mass alone, even `mF>=.95`, is NOT delivery. Late older revisions cannot undo newer retractions. Source disagreement can be `disputed`.
8. `downlink`: GCS prepares bounded target-only `MAP_DELTA` packets containing unique source events and snapshots. GCS sends to **one physically reachable UAV** round-robin; other UAVs receive through learned peer relays. Snapshot is metadata, **never independent evidence to fuse again**. UAVs recompute their own target masses from original `origin/event_id` evidence and update local 2D maps only after real packet delivery.
9. `reward`: existing M2 research reward is retained, **not from Zhang**: +.15 first new local threshold detection, +1 correct/−1 false new verified local cell, +50 first newly correctly verified delivered report at GCS/−10 false report, +4 decrease in mean GCS DS uncertainty and +1 local uncertainty decrease, minus propulsion/radio energy and step cost. Observation has no ground truth; reward/evaluator alone use truth. Reward labels `candidate/verification` remain for metric ABI compatibility but refer to the immediate DS threshold in the paper mode.
10. `terminal/log`: terminate at 1500 steps or battery depletion; record GCS correct verified delivery recall, normalized delivery AUC, false reports, GCS target AUPRC/Brier/uncertainty/coverage, energy, MAC transmissions/PDR, losses; final evaluation knows simulator truth, UAV Actor and Critic do not.
11. `MARL training`: decentralized shared MATD3 Actor, centralized observation-only TwinCritic. During training, `s_t,o_t,a_t,r_t,s_{t+1},o_{t+1},done` go into replay; sample batches; critic minimizes TD target with min of twin target Qs and target policy noise; actor updated every 2 critic updates with deterministic policy gradient; target networks soft-update. The old reference run `18d756ef` specifies Actor BatchNorm and plain Critic, H256×512, E4096, U4, B8192 and 75% CPU/25% GPU replay. **This is NOT running for full M2**: existing `warp_packet` rollout and `experiments/training.py` explicitly reject `m2_enabled`. The separate one-world Native pilot has GPU learner, H256×256, B128, replay20k and 8-cell scan cap; it is NOT scientifically equivalent to the E4096 reference.
12. `validation/test`: evaluate fixed held-out seeds, deterministic policy, mission-level detection/verified delivery/false alarm/energy/map quality and communication; never present random-policy smoke-test results as RL convergence.

## Source paths and regression tests

- `uav_network/communication/m2_protocol.py` — per-source MAX counts, fixed BPA, exact normalized masses, threshold reports and network correction; legacy correlated SPRT opt-in preserved.
- `uav_network/environment/m2_mission.py` — M2 paper DS defaults, local/global observation, reward, radio/GCS and 3D motion.
- `uav_network/communication/m2_network.py` — physical packet distribution, per-event dedup, decision notice creation time & TTL fix.
- `scripts/research/run_cooperative_m2_native_demo_20261121.py` / `scripts/research/verify_m2_native_1500step_flow_20261121.py` — paper baseline diagnostic (not RL).
- `scripts/research/train_m2_matd3_native_gpu_20261121.py` — paper-mode Native pilot with checkpoint format `m2-target-belief-2d-matd3-v2`, observation schema `m2-local-5x5-f-e-u-zhang-ds-v2`. Old v1 SPRT pilot snapshots intentionally rejected on semantic mismatch.
- `tests/test_m2_zhang_ds_confirmation_20261121.py` — test direct .8/.96 mass, MAX not SUM, no SPRT, retraction/versioning, delayed radio report created time, event dedup and local Actor ABI.
- Keep the original reference run/model frozen: *Do not overwrite original checkpoint weights or compare new task metrics to old task as if they were the same objective.*

## Limits on scientific claims

This change is a **Zhang-faithful target belief/threshold submodule**, **not** an exact reproduction of the full Zhang pipeline. A fixed DS BPA .8 is a paper simulation setting; it is not automatically calibrated to M2 altitude-varying `Pd/Pf`. The source MAX fusion deliberately avoids summing corroborating readings from independent UAVs, as the paper chooses; compare with an alternative fusion ablation if higher recall is desired. Reversible REPORT versions and radio-level GCS/downlink are M2 engineering extensions, not author-proven DS results. Need multi-seed calibration, threshold sensitivity (`tau=.9,.95,.99`), false alert rate, target delivery recall, and matched-budget training to justify a new contribution.

## Verification, 2026-11-21

`PYTHONPATH=. pytest -q tests/test_m2_zhang_ds_confirmation_20261121.py tests/test_m2_target_belief2d_20261121.py tests/test_m2_hardening_20261121.py tests/test_m2_logic_audit_20261121.py tests/test_m2_mission_native.py tests/test_correlated_m2_protocol.py tests/test_m2_uavnetsim_integration.py tests/test_m2_matd3_training_20261121.py tests/test_m2_exact_18d756ef_parity_20261121.py tests/test_m2_ds_warp_projector_20261121.py --disable-warnings --tb=short` → **105 passed** in 16.91s.

`python scripts/research/run_cooperative_m2_native_demo_20261121.py --steps 35 --seed 44 --output /tmp/m2_zhang_ds_demo_new_20261121` → PASS; GCS evidence events=612, radio packets=244, GCS coverage=0.11125 and true reported target count=1, scripted policy only. This is not a 1500step stress test nor an RL convergence result.
