# MATD3 NoOracle V9/V7/original speed investigation — 2026-11-22

Source: original run 18d756ef commit 0804dd75; V7 run 504d1b89 commit fbba543; V9 run e079eace commit 1346ad47.

W&B sampled history global-step slope (environment transitions per second): original 4612, V7 2082, V9 2022. These are unpaired time windows and do not establish comparable GPU utilization or policy performance.

V9 cumulative timers in actual W&B: collector wait ~2147 seconds, learner ~819 seconds, hybrid replay sampling ~496 seconds, optimizer updating ~322 seconds. Per-step collector wait includes the entire Warp environment, not only communication.

V9 active HF archive exists under duy95/uav-search-diagnostics/runs/e079eace/diagnostic-archives, and recovered V7 17 files at /home/aduy/Downloads/MATD3-V7-504d1b89-RAW-KAGGLE-LOGS-RECOVERED-20261122.tar.gz.

Source comparison: legacy original target report slots = 35, NoOracle observed-cell hypothesis report slots = 140. Extra loops exist in MAC, report buffers, observation, reward, and Bayesian candidate matching. The runtime simulator uses custom Warp PHY/MAC kernels; upstream Python UavNetSim is a reference, not a Python process executed for each step.

Branch perf/nooracle-v9-sparse-report-iteration-20261122 modifies internal report-slot loops to break at the first unallocated slot. In normal world state, report slots are append-only, and all reset to -1 together; maximum capacity stays 140. Critical: Actor observation and centralized critic must scan all slots to preserve report-slot permutation invariance. An attempted optimization to Actor/Critic produced 2 test failures and was removed.

CPU 32-env 160-step sample: original ~2.976s, optimized ~2.496s for near-empty report slots; kernel share original Bayes ~35.2%, observation ~28.2%, packet_step ~22.5% of total wall time. CPU 32-env 120-step with 8 preallocated slots: original 1787.7 transitions/s and optimized 1756.0; no robust gain shown there. Trajectory hashes were byte-identical in both microtests. These CPU results are not CUDA T4 benchmarks.

Run scripts/profile_nooracle_warp_phases_20261122.py with --device cuda --envs 4096 --steps 80 --warmup 6 on the same GPU to profile individual CUDA kernels. It uses Warp CUDA activity timing and synchronization. --network-disabled gives a counterfactual only and must NOT be used for production training.

Research: NVIDIA Warp profiling https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/profiling.html ; NVIDIA Isaac Lab Warp https://docs.nvidia.com/isaaclab/experimental-features/warp-environments ; BenchMARL JMLR2024 https://www.jmlr.org/papers/v25/23-1612.html ; FastTD3 https://arxiv.org/abs/2505.22642 ; upstream UavNetSim https://github.com/Zihao-Felix-Zhou/UavNetSim . CUDA graphs and fusion are candidates, not validated changes.

Safety: do not turn off actual network, shrink report capacity, leak hidden true targets, reduce report semantics, or modify user running Kaggle job merely to speed the benchmark.Additional counterfactual CPU test in the same unpatched V9 source: with NoOracle cells enabled (140 slots) 32×160 steps took 3.090 s, while flipping to legacy oracle-style target reports (35 slots) took 4.320 s because legacy observation sorting is more expensive on CPU. Therefore slot count ALONE cannot explain the 2.28x W&B GPU throughput gap. This reinforces the need for actual T4 CUDA per-kernel profiling before attributing a root cause.
