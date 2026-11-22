# MATD3 NoOracle V9 training speed investigation and bitwise-equivalent acceleration

**Scope: TRAINING ONLY, not deferred GPU evaluation.** Historical Kaggle run
`e079eace` is left unchanged. Baseline `18d756ef` and NoOracle V9 do not
share identical scientific target-report semantics, so throughput differences
cannot be ascribed solely to the user-selected `num_envs=4096`.

## Evidence from real W&B

- Baseline `18d756ef`: `train/joint_transitions_per_s = 4599.7581`.
- Running V9 `e079eace`: `train/joint_transitions_per_s = 1981.9121`.
- Same 4096 environment rows, same 20k episode budget; V9 roughly
  `1981.9/4599.8 = 43.1%` of baseline throughput, i.e. 2.32x slower.
- V9 cumulative timers: `timing/collector_wait_seconds=9073.0`,
  `timing/learner_seconds=3473.9`, `timing/replay_sample_host_seconds=2069.15`,
  `timing/optimizer_update_host_seconds=1403.05`.
  The learner timer includes replay sampling + optimizer; **do not add the
  individual learner subcomponents again**. Collector and learner can overlap,
  so timer ratios are not necessarily exclusive GPU percentages.
- Original NoOracle source SHA in `e079eace` W&B: 1346ad47, source scenario
  `sensor.report_cell_hypotheses=True`, `sensor.verified_empty_belief=0`.
  Original uses 35 target ID slots (`18d756ef`); V9 uses up to 140 observed
  cell hypothesis slots. Map size 1000m, cell 25m, 40x40 belief grid, 6 UAV,
  35 targets, 25 obstacles. The NoOracle scene generator enforces unique
  target cells.

## Where processing happens (NOT one Python UavNetSim per world)

`uav_network/environment/gpu/vector.py` launches Warp kernels directly on
GPU0 for 4096 worlds; there are no 4096 Python env processes. `sense_bayesian`
handles confidence and truth-generated sensor noise; `confirm_observed_cells`
handles cell-level evidence; `packet_step` implements embedded Warp PHY/MAC
(UavNetSim-like CSMA/CA events and link/interference calculations). `observe`
materializes local decentralized Actor observations. MATD3 and 75% CPU /
25% CUDA replay run with the separate learner GPU.

A controlled CPU32×180 random-action counterfactual on pinned V9 source
measured 3.609s with packet network enabled and 2.184s with networking disabled;
this **does not identify the exact exclusive packet kernel percentage**, as
network toggling changes resulting simulation dynamics. Kernel phase profiling
at CPU32×220 indicated `sense_bayesian`, `observe` and `packet_step` are the
largest per-step kernels. Exact ranking and magnitude on T4 are unverified.

## Fix implemented on a NEW branch (source V9 not modified)

1. Keep known-safe sparse early termination for **append-only internal report
   slot** scans in Warp MAC, report queue and confirmation kernels. Previously
   landed on the original separate sparsefast branch. Do NOT early-break Actor
   observation or critic state on first empty slot: that fails intentional
   slot-permutation invariance tests.
2. Rebuild an **ephemeral cell-to-target lookup** once per NoOracle scene reset.
   Since the scene generator guarantees one target per cell, the Bayesian
   camera can look up the single matching target in O(1) rather than scanning
   35 targets for every observed cell. The same range, obstacle occlusion,
   probability and per-cell counter logic remain unchanged.
3. The lookup is on `WarpVectorEnv` **outside** `WorldState` and out of
   `snapshot_state()` and critic observation: no new oracle leakage, no
   checkpoint/forensic payload bloat. Exact resume recomputes the derived
   lookup from restored truth-side world geometry, not from the Actor.
4. NoOracle Actor observation now counts all 140 report slots just once per
   UAV and reuses the exact resulting pending/fresh report eligibility for
   source-local destination masks. This preserves permutation invariance and
   NoOracle concealment of peer private state. The original `destination_allowed`
   reference path remains for historical/legacy scenarios.
5. The original public 4-parameter `sense_bayesian` kernel remains available
   for existing scientific/geometry tests. `WarpVectorEnv` invokes a distinct
   internally indexed kernel; scientific outputs are not switched over in
   historical saved model code.

## Verification and results (not a T4 speedup claim)

- Historical `1346ad47` vs accelerated source, 12 CPU worlds ×180 steps,
  same generated actions and seeds: **all 9 trajectory SHA256 checkpoints
  matched** for common scientific arrays (belief, coverage, reward, packet,
  delivered, report, Actor observation, critic state, energy, etc.).
- Reset and partial reset rebuild the exact lookup; old snapshot format and
  exact environment restoration tested; no GPU memory/CPU checkpoint bloat.
- CPU32×220 paired profile runs (same config, same action generator seed):

  | Pair | Original (trans/s) | Accelerated (trans/s) | Gain |
  | --- | ---: | ---: | ---: |
  | 1 | 2197.47 | 2622.38 | +19.3% |
  | 2 | 2141.01 | 2296.87 | +7.3% |
  | 3 | 1907.82 | 2272.88 | +19.1% |

  *Paired trial speedup ranged 7.3–19.3%; sample medians yield +7.3%.*
  CPU profile only, **NOT a validated CUDA T4 or end-to-end 20k-episode speedup**.
- Replay CPU→GPU sample transfer (75% CPU, 25% GPU), root-audit and
  non-blocking H2D pinned staging remain additional measured hypotheses.
  Do NOT claim they are fixed or change replay sampling/reward logic without
  controlled CUDA profiling and real learner metric parity.

## Reproduce / evaluate on Kaggle GPU T4x2

Run the adjacent GPU comparative benchmark notebook, once available, on the
same T4 accelerator, with warmup compilation excluded and the original
4096 envs. Compare actual per-kernel CUDA activity, wall time and output
checksums; check host↔device synchronization separately. A CUDA graph might
help launch overhead only if Nsight shows that launch dominates. NVIDIA Warp
profiler guidance: https://nvidia.github.io/warp/v1.18/user_guide/faq.html
and https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/profiling.html .

Do not touch active `e079eace`, shrink map/target count, turn off RF network,
change reward or episode horizon, leak true target cells into Actor input,
or silently claim baseline-level performance.

## Deliverables: two isolated Kaggle UI notebooks

1. `notebooks/MATD3-V9-TRAINING-CPUCUDA-AB-4096-WARP-KERNEL-PROFILER-T4x2.ipynb`
   is the **recommended first run**. It embeds the original source bundle from
   the pinned historical V9 notebook, then overlays only SHA256-checked module
   files to generate the accelerated version. Requires Kaggle T4×2 and Internet
   for package installation, no W&B/HF secrets. Executes three sequential
   A/B Warp CUDA probes at **4096 worlds** on the SAME physical T4 GPU0,
   after excluding JIT warmup; collects time and CUDA activity for
   `sense_bayesian`, `observe`, `packet_step`, etc. Verifies same-seed 4-world
   CUDA trajectories between source variants. Saves a JSON report to
   `/kaggle/working/v9_training_ab_cpu_cuda/v9_gpu_kernel_ab.json`.
   No MATD3 training/replay, no account mutation; it is environment-only.
2. `notebooks/MATD3-NOORACLE-V9-TRAINACCEL-CELLLOOKUP-OBSFAST-E4096-U4-B8192-H256x512-S44-KAGGLE-UI.ipynb`
   is the **full fresh training run** in a new W&B run, with original
   E4096/U4/B8192/20k-episodes/seed44/6UAV/35targets/25obstacles,
   GPU0 Warp collector, GPU1 MATD3 shared ActorBN and plain Critic, 75/25
   hybrid replay, fixed checkpoint Eval100, best actor, Test4000, W&B and HF.
   It **will** retrain MATD3 from random initialization if run. The builder
   uses original reference notebook's three-cell format and embeds the entire
   SHA256-verified source archive; no source checkout is required in Kaggle.

   Note: the full training notebook keeps the conservative **2 checkpoints ×
   100 worlds / GPU batch** V9 Eval setting to avoid changing too many factors
   in the training throughput A/B. This means its *post-training* Eval can
   still be slow. Use the separate Eval/Test-only benchmark notebook from
   branch `experiment/v9-evaltest-baselinebatch-20261122` for the Eval batching
   experiment, and keep these experiments isolated.

**Interpretation:** First compare the GPU profiler report's per-kernel CUDA
activity and total simulated-world steps/s, and verify parity. Only after
both the runtime and scientific match on T4 should a full 20k training job be
used to claim end-to-end speedup. If CUDA parity fails or GPU throughput is
worse, stop; retain pinned V9 semantics and do not merge this branch.
