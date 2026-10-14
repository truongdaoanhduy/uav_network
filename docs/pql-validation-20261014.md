# PQL-inspired two-T4 verification, 2026-10-14

Code measured: e680276b4a4b6c081b3a684132848c7e49104765.
Kaggle: https://www.kaggle.com/code/haibro1234/uavnet-benchmark-20261014-141837-fb9fbe .
This is a short pipeline/VRAM benchmark, not a convergence experiment or a speedup comparison against the fixed runner.

## Exact case
- 2 × Tesla T4, each 14.56 GiB; Torch2.11.0+cu128, CUDA12.8, Python3.13.15.
- 6 UAV, 25 targets, 15 obstacles, configured episode horizon1000.
- 4096 GPU worlds, batch8192, replay500000 joint rows, hidden256×256.
- 40 actual vector collections =163840 joint transitions.
- Warmup32; first eligible update is collection32, yielding9 eligible collections and72 critic updates at ratio8.
- MASAC72 policy/alpha updates; MATD336 policy updates following policy_delay2.
- Collector CUDA0; replay, critic and policy CUDA1; zero CPU environment workers.
- Same seed44 and unchanged algorithm/world/reward/network/safety configuration.

## Measured results
| Algorithm | Seconds | Joint transitions/s | Critic / policy updates | GPU1 Torch allocated peak GiB | GPU1 Torch reserved peak GiB |
|---|---:|---:|---:|---:|---:|
| MASAC | 41.36 | 3961 | 72 / 72 | 8.41 | 11.37 |
| MATD3 | 17.86 | 9172 | 72 / 36 | 5.62 | 5.96 |

Replay allocated 4,920,500,000 bytes (~4.58 GiB including age/seen metadata).
Warp environment allocation report 1,843,609,600 bytes (~1.72 GiB) on GPU0.
GPU0 Torch memory counters do not include all Warp-owned allocations; do not interpret them as total physical GPU usage.
GPU1 rows above include the full-capacity replay allocation, learners and active batch tensors; physical CUDA context/workspace costs are not fully described by Torch counters.

Elapsed/throughput exclude initial construction and final W&B shutdown. They include32 warmup collections and all subsequent learning/drain time.
They are short-window figures, not guaranteed steady-state training throughput.
Both cases passed with finite executed gradients, target CUDA topology confirmed and no OOM.
The entire run's sample-use/new-row metric is3.6 because warmup contributes inserted rows but no retrospective updates; steady configured target is16.
Replay did not yet wrap in40 collections. Unsampled-eviction regression uses a small real ring; no claim of a measured long-run eviction rate.

## Tests and integrity
- Local full suite278 tests OK;46 CUDA skips on the CPU host.
- T4 baseline CUDA tests10/10, mission regressions8/8, PQL training/smoke9/9 including a fresh CUDA benchmark subprocess.
- Old update equations frozen at afeaa3d match split wrappers exactly for losses, weights, target networks, Adam and RNG.
- Independent review's three material findings fixed: final resume artifact selection, consistent progress/fractional credit validation, and rejecting a benchmark with no completed learner updates.
- First T4 attempt f25892f stopped at peak-counter reset before CUDA allocator initialization. Explicit cuda.init plus a fresh-process regression corrected it; the rerun above passed.
- Async schedules are not bitwise deterministic. Synchronous debug repeatability is tested.
- Saved policy/replay/optimizer state resumes with fresh simulator episodes and unused seeds; it is not a mid-episode exact simulation restore.

## Online records
- MASAC: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/af2b713c .
- MATD3: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/1cce9677 .
- Machine-readable results: [pql-t4-20261014.json](benchmarks/pql-t4-20261014.json).

Learning loss/alpha values in the short report follow configured periodic logging: this window records the initial checked update, not a loss trajectory or final72-update estimate.
No long production training was launched. No outcome/convergence improvement has been established; use controlled curves against transitions AND wall time before paper claims.
