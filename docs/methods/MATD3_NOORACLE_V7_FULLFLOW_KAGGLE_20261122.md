# MATD3 v7 Search→REPORT→GCS — final source/runner review and Kaggle handoff

Date: 2026-11-22. This is the **no-oracle Bayesian cell-report** MATD3 task,
not the separate native M2 Dempster–Shafer mapping task.

## Reproducing the frozen `18d756ef` training layout

Reference run: `MATD3-EVAL100-ACTORBN-PLAINCRITIC-E4096-U4-B8192-H256x512-S44-18d756ef`.
The original 2026-11-21 source is frozen at
`/home/aduy/Downloads/M2_EXACT_REFERENCE_18d756ef_20261121/verified_source/`.
This new experiment restores the **same** fullflow modules, source layout and
selection protocol on the corrected v7 simulator, without loading the old model:

- MATD3, 6 UAV, 35 targets in unique 25m cells, 25 obstacles, 1000m map,
  1500 steps max/episode, seed44, 20,000 complete episodes.
- E4096 simultaneous Warp CUDA environments; four critic/actor update attempts
  per vector step; replay1,000,000 (75% CPU / 25% GPU); batch8192.
- Shared Actor BatchNorm hidden layers [256,512], plain Critic, no CBF;
  `world.motion_safety=reject`. GPU0/GPU1 placement inherits the original
  launcher; only real CUDA T4×2 testing can verify it under quota.
- Immutable Actor snapshot at t=0 and every 100 completed vector steps, plus
  actual final and real training-best candidate. Deterministic Validation on
  **the SAME 100 held-out seeds** for each exact 100-vector-step checkpoint
  and final policy. Select best **from validation**, then **one held-out
  4,000-scenario final test**, never select from Test.
- Final report under `/kaggle/working/one-session-train-eval-test-report.json`;
  required W&B summary booleans `fullflow/train_complete`,
  `fullflow/validation_complete`, `fullflow/test_complete`, `fullflow/complete`.
  If all 20k episodes have not finished, the one-session evaluator **refuses**
  to mark the run successful or launch final testing.

## Bugs caught by the full runner integration (and fixed)

1. Original deferred GPU evaluator continues stepping a batch containing
   already terminated worlds with **all-zero no-op** actions. Old `validate_actions`
   rejected these with `s.errors=92`, even for allowed no-ops. In **cell-report
   mode only**, it now permits zero actions on terminated worlds. `begin_step`,
   `sense_reports_at`, `packet_step`, and `end_step` skip those worlds, preserving
   terminal state and preventing extra energy/rewards. Nonzero actions on a
   finished world still error. Historical mode retains strict behavior.
2. Restored `MissionCauseMonitor` assumed `known.shape=(E,6,35)` but new REPORT
   namespace has 140 slots; it logged `action_safety_logging_failed` instead
   of actual forensic mission outcomes. Now a **truth-scoring-only observer**
   maps reported cell IDs to target cells, computes TP/FP/true delivery, and
   writes metrics without exposing truth to Actor or GCS admission.
3. Restored `vector_step_mission_progress` treated the v7 Actor's aggregate
   report columns and hashed Critic columns as true target identities. Replaced
   these metrics in v7 with vectorized truth-rated post-step, pre-reset
   search/delivery/coverage that uses the simulator state **only in analytics**.
4. Restored training entrypoint allocated an output directory before rejecting
   `runtime.m2_enabled=True` (unsupported native M2-to-Warp port). It now fails
   early without mutating files. No DS mapping/GCS fusion was added.
5. Kaggle bundle launches `uav_network.cloud.remote` for real training, then
   `uav_network.experiments.one_session` for two-GPU validation and final test;
   reserves 5,400s for evaluation inside an 11h session. Preflight fails closed
   if requested episode budget is unfinished. Source code is embedded in the
   notebook, pinned UavNetSim source checked by SHA256.

## Real checks performed

- Focused regression including M2 non-portability guard, Warp, packet radio,
  v7 no-oracle scientific telemetry, training-entry, source bootstrap:
  **161 pytest tests passed +14 parameterized subtests** after the changes.
- Real short **MATD3 training with 66 optimizer updates**, 4 episodes / 2 worlds
  / 20 steps, CPU Warp; two distinct immutable Actor snapshots, checkpoint
  manifest and compressed bundle saved successfully.
- Deterministic grouped validation smoke: **two policies × 100 identical
  validation seeds each**, `evaluate_group` on CPU, all 200 rows complete.
- Independent `gpu_validate_one` Actor-only held-out seed evaluation smoke:
  two scenarios completed with `warp_vector_cpu_smoke` backend.
- Prior branch no-oracle sensor, GCS report, false-report accounting and MAC
  regressions: **187 pass** at pinned source commit `8211e26`.

## External launch state

On the user's connected Ubuntu 24.04, Warp 1.18.0 reports **no accessible CUDA
GPU driver**, so a 20,000-episode E4096/T4×2 run cannot execute locally.
The Kaggle CLI/API presently fails before authenticated quota checks due to
incompatible versions of `kaggle` and `kagglesdk` (not proof of exhausted
quota). The builder `scripts/build_matd3_nooracle_one_session_ui.py` creates a
self-contained `ipynb` for **Kaggle UI** that reads notebook-scoped
`WANDB_API_KEY` and `HF_TOKEN` secrets at runtime; no secret is embedded.
Before execution select **GPU T4×2, Internet On, Save & Run All**.
Do not claim a remote job was submitted unless Kaggle supplies a kernel URL
and a real run/version receipt. Do not claim a final test is complete until
`fullflow/complete=true` is observed on W&B for the new unique run ID.

## Known scientific limitations (outside this user's requested scope)

- DS occupancy evidence fusion/GCS feedback and 3D obstacle reconstruction
  remain **out of scope**. Bayesian search/report only.
- False positive report retraction, temporal sensor correlation calibration,
  report slot 140/1600 bound, lossy 830D Critic hash embedding and a physically
  integrated peer beacon are future research. These are not presented as fixed.
- Learned motion reject/corner saturation and battery exhaustion require new
  multi-seed ablations; no new long-run success metric is invented.
