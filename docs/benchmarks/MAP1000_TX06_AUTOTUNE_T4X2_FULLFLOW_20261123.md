# MAP1000 / TX0.6 — T4×2 conservative hardware autotune and full-flow audit

**2026-11-23** · Branch `perf/map1000-tx06-macblockdim-20261123`. Source worktree `/home/aduy/Documents/NCKH/uav_network_map1000_tx06_macblockdim_20261123`. Original production branch `experiment/map1000-tx06-posterioronly-fastgrid-20261123` at commit `49bc095420a6537147eeff4c7769598682df8ff9` is not modified.

## Core requirement and constraints

Keep exactly the MAP1000×1000m/40×40 2D Bayesian belief (25m/cell), six UAV, 35 targets, 25 obstacles, altitude z up to 150m, independent `Pd/Pf` by altitude, **positive sensor reading AND posterior≥0.99** directly confirmed (no independent LR/altitude confirmation gate), learned TX power capped0.6W, complete Warp UavNetSim network PHY/MAC, same NoOracle cell-ID report custody, all false hypotheses, collisions, expiry, retry, battery, observation and reward semantics. MATD3 H256×256 ActorBN, plain Critic, seed44, 4096envs, 4updates/vector step, B8192, 20,000 training episodes, horizon1500, hybrid replay75%CPU/25%GPU (capacity1M). Save Actor initial, every100 vector steps and final; GPU Validation100 on every saved Actor, select best by validation, final independent heldout Test1000. Maintain W&B/HF deep forensic logs, independent seeds, and immutable evidence.

**Do not disable UavNetSim, cut the episode budget or pretend to improve training science quality by omitting false confirmations.** Comparing new Posterior-only vs 18d756ef (truth-target-ID reports) is not a controlled throughput benchmark.

## Scientific audit performed on real local CPU Warp

- `scripts/audit_map1000_tx06_mission_full_horizon_20261123.py` executed real complete **16 worlds × 1500 steps**, seeds400–415, same scripted 3D actions as original MAP1000/TX0.6 reference, state/model unchanged, complete network and Bayesian sensor.
- Exact **per-world mission metric parity with original MAP1000 reference** including every world seed, steps, discovered/delivered counts, total reported hypotheses, false-confirmed cells, false GCS reports, termination reason, confirmation source, 140 report-slot capacity and max altitude150m. `any_internal_error=False`; totals **535 true targets discovered, 442 delivered, 393 false-confirmed cells**, max77 hypotheses. This is an environment/scenario audit, **not trained-policy performance**. Evidence in `/home/aduy/Downloads/MAP1000_TX06_AUTOTUNE_16WORLD_FULL_HORIZON_20261123.json`.
- `scripts/audit_map1000_tx06_cpu_fullflow_20261123.py` and regression runs: **real CPU MATD3 train6 episodes -> 7 immutable Actor snapshots -> Validation2 independent seeds each -> Best selection -> heldout Test2**; this bounded functional smoke does not prove 20k E4096 T4 speed or learned effectiveness. Separate embedded-source test extracts EXACT source archive from the newly built Kaggle notebook and really performs CPU MATD3 train2 episodes -> actual saved checkpoint -> heldout evaluation2.
- New profiler runs CPU `UAV_WARP_MAC_BLOCK_DIM` values0/8/32/128 with **matching SHA256 of all simulator arrays**. This is a **CUDA-only** launch option (CPU ignores it), so CPU speed at these values is not a GPU speed benchmark.
- New replay A/B B8192 (same full 145MB minibatch shape, CPU75/25 layout, capacity16384) `.sample()` staging OFF/ON on CPU returned **bitwise-identical entire minibatch, RNG and `seen` bitmap**. CPU timer readings are not CUDA pinned transfer evidence.
- The notebook autotune controller is executed in CPU tests using synthetic GPU metrics for fast, too-slow, SHA mismatch and baseline-throws scenarios. It always rejects state mismatches and gains of <=5%, falls back to stock scheduling, and independently decides pinned staging based on identical minibatch/RNG.
- Prior attempts with live-frame index, frame emit sparse reclaim and 64-fragment shortening produced full-horizon bitwise parity but **no stable speed gain on CPU**, so they are **NOT in either new sealed notebook**.

## What was actually optimized and how decision is made

**CUDA scheduling:** optional environment variable `UAV_WARP_MAC_BLOCK_DIM` in `uav_network/environment/gpu/vector.py`, permitting 0(stock) /32/64/128 (also validated other power-of-2 values), applied only to CUDA `packet_step` and only NoOracle cell-report mode. The simulator state, PHY/MAC kernels, scheduling order, queues, packet processing, rewards and episode/action config are unchanged. Benchmark profiler `scripts/profile_nooracle_warp_phases_20261122.py` logs `packet_step` kernel ms and throughput and can hash all scientific arrays for <=32 worlds.

**Hybrid Replay:** original `HybridMeasuredReplay` unchanged; existing source already has optional bounded pinned staging via `UAV_HYBRID_PINNED_STAGING=1`. `scripts/benchmark_hybrid_replay_cuda_ab_20261123.py` measures B8192 real state shapes with CPU75%/GPU25%, hashes all tensors + RNG/seen state; comparison OFF/ON runs on the same GPU1. Pinned host buffer staging is NOT enabled unless a faster and semantically identical T4 result is observed. This does **not** convert 1M replay into an infeasible full-GPU buffer.

**Conservative autotune:** `scripts/build_map1000_tx06_autotune_fullflow_ui_20261123.py` creates the 3-cell SHA256-pinned full-training notebook. It tests each MAC blockdim with E16 scientific-state SHA parity (GPU0), paired 2×48-step E4096 environment kernel profiles; selects a matching setting only for >5% median improvement over stock. Then tests replay OFF/ON paired with B8192 & matching minibatch+RNG/seen; selects pinned only for >5% speed benefit. If probe fails, defaults to MAC blockdim0 and unpinned replay, independently. Writes `/kaggle/working/MAP1000_TX06_AUTOTUNE_RESULT.json`. **Autotune completes BEFORE remote training/Val/Test, not after.** Retains all production requirements without skipping any policy evaluation or changing science.

**Pilot:** `scripts/build_map1000_tx06_t4_mac_replay_ab_ui_20261123.py` creates separate T4×2 benchmark-only notebook (3cells, no secrets, no 20k production job). It runs more CUDA blockdims 0/8/16/32/64/128 on GPU0 and paired replay on GPU1, with SHA-checked source. Output `/kaggle/working/MAP1000_TX06_T4_MAC_REPLAY_AB_RESULTS.json`.

## Scientific/performance interpretation and blockers

- Old known W&B results: original `18d756ef` 4,599.758 transitions/s, NoOracle `602a1e82` 2,493.174 transitions/s; these run different scenarios, so **cannot be cited as speedup of this new source**.
- Congestion evidence: direct HF `episodes.jsonl` archive25 was baseline25 packet/episode vs NoOracle1210.5 packets/episode, collisions0.5 vs1492.1; see `docs/benchmarks/MAP1000_TX06_MATD3_FULLFLOW_ROOT_CAUSE_20261123.md` for sample limits, underlying source and academic context. No MAC/PHY shortcuts are authorized just to make measured throughput resemble baseline.
- The connected Ubuntu laptop lacks the NVIDIA CUDA driver (`nvidia-smi` unavailable), so **no real Kaggle T4×2 run, GPU speedup or final 20k episode evaluation was executed in this chat**. Current tests prove CPU pipeline and immutable source/scenario integrity only. The two Kaggle notebooks contain the full GPU A/B sequence for running on actual T4×2.
- Pinned staging A/B uses 16,384 capacity for a bounded hardware pilot, but the **production training replay capacity remains1M**; pilot data throughput may not predict full 1M replay performance. Env profiler uses random actions for apples-to-apples kernel comparison; trained actor traffic may differ. Threshold5% reduces premature tuning but is not a guarantee of full-run speedup or policy quality.
- Kaggle fullflow still requires Accelerator T4×2, Internet ON and Secrets `WANDB_API_KEY` / `HF_TOKEN`, as the original notebook. The standalone pilot requires neither Secret.
- All source and `.ipynb` are checksum-pinned, no source checkout occurs inside Kaggle; do not reuse/overwrite old run IDs/checkpoints. Outputs are separately named.

## Public academic references

- WarpDrive (JMLR 2022): https://www.jmlr.org/papers/v23/22-0185.html
- EnvPool (NeurIPS 2022): https://papers.neurips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html
- BenchMARL (JMLR 2024): https://www.jmlr.org/papers/v25/23-1612.html
- FastTD3 reference source: https://github.com/younggyoseo/FastTD3
- PyTorch pinned host memory tutorial: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

## Outputs

- Fullflow notebook: `notebooks/MATD3-MAP1000-TX06-AUTOTUNE-T4X2-TRAIN20000-VAL100-TEST1000-20261123.ipynb`
- Standalone A/B pilot notebook: `notebooks/MAP1000-TX06-T4X2-MACBLOCK-REPLAYPIN-AB-PILOT-20261123.ipynb`
- Full CPU scenario report: `/home/aduy/Downloads/MAP1000_TX06_AUTOTUNE_16WORLD_FULL_HORIZON_20261123.json`
- CPU B8192 replay parity: `/home/aduy/Downloads/MAP1000_TX06_AUTOTUNE_CPU_REPLAY_B8192_PARITY_20261123.json`
- Regression report: `/home/aduy/Downloads/MAP1000_TX06_AUTOTUNE_GREEN_REGRESSIONS_20261123.log`
