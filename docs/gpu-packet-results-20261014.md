# GPU packet environment results — 2026-10-14

## Scope

This report closes Task 4 of `docs/superpowers/plans/2026-10-14-gpu-packet-env.md`: real CUDA parity, capacity sweep, telemetry, and evidence for the `warp_packet` environment. The implementation uses NVIDIA Warp 1.18 kernels and PyTorch tensors on the same CUDA device; it does **not** replace or modify the pinned upstream UavNetSim oracle. Native UavNetSim remains the behavioral reference used by parity tests.

The successful cloud run is pinned to commit `93feee740d20c8c34f93b6e60fba0cb832dbf68a`:

- Kaggle: https://www.kaggle.com/code/duytrngoanh/uavnet-gpu-capacity-20261014-085653
- Hardware: 2 × Tesla T4, 15360 MiB each; benchmark cases use CUDA device 0.
- Python 3.13.15, PyTorch 2.11.0+cu128, NVIDIA Warp 1.18.0.

## Verification

The successful T4 job passed:

| Gate | Result |
|---|---:|
| `test_gpu*.py` | 89/89 pass |
| legacy `test_cuda.py` | 10/10 pass |
| mission regressions | 8/8 pass |
| local full suite after final clock fix | 254 tests, 0 failures, 44 CUDA skips |

The first T4 attempt exposed one real CUDA-only regression in the non-binary `dt=0.7` packet clock: a fused multiply-add could produce `12600000.0 us` where native Python ordering produced `12599999.999999998 us`. The fix materializes `steps * dt` as FP64 `mission_time` in the preceding kernel, then adds `dt`, preserving native operation ordering. The failing evidence job is retained at https://www.kaggle.com/code/duytrngoanh/uavnet-gpu-capacity-20261014-084800.

## Capacity sweep

All cases use the same 6-UAV/10-target/8-obstacle scenario, batch size 512, replay capacity 100000, learning starts 4000, and one optimizer update per vector step. Each case runs a real learner and real `warp_packet` environment; telemetry below is from the measurement window after warm-up.

| Envs | Algo | CPU env workers | Env VRAM MiB | GPU peak MiB | GPU util mean | CPU util mean | Host RAM peak GiB | Joint transitions/s | Sample uses/new transition |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | MASAC | 0 | 50.3 | 1201 | 77.5% | 28.0% | 3.90 | 537.9 | 4 |
| 128 | MATD3 | 0 | 50.3 | 1019 | 88.2% | 28.1% | 3.82 | 549.8 | 4 |
| 256 | MASAC | 0 | 100.7 | 1263 | 80.8% | 28.1% | 3.88 | 832.6 | 2 |
| 256 | MATD3 | 0 | 100.7 | 1085 | 91.0% | 28.1% | 3.83 | 881.8 | 2 |
| 512 | MASAC | 0 | 201.4 | 1359 | 80.2% | 28.1% | 3.88 | 1695.4 | 1 |
| 512 | MATD3 | 0 | 201.4 | 1181 | 91.5% | 28.2% | 3.84 | 1835.6 | 1 |
| 1024 | MASAC | 0 | 402.7 | 1551 | 81.9% | 28.1% | 3.90 | 3496.7 | 0.5 |
| 1024 | MATD3 | 0 | 402.7 | 1371 | 92.5% | 28.2% | 3.83 | 3604.6 | 0.5 |
| 2048 | MASAC | 0 | 805.4 | 1935 | 84.1% | 28.2% | 3.91 | 6716.3 | 0.25 |
| 2048 | MATD3 | 0 | 805.4 | 1775 | 91.9% | 28.1% | 3.84 | 7158.7 | 0.25 |
| 4096 | MASAC | 0 | 1610.9 | 2735 | 82.9% | 28.1% | 3.90 | 13268.5 | 0.125 |
| 4096 | MATD3 | 0 | 1610.9 | 2577 | 93.8% | 28.3% | 3.84 | 14184.8 | 0.125 |

`stop_reason` is `null` and `skipped_counts` is empty: 128, 256, 512, 1024, 2048, and 4096 environments all completed for both algorithms. No CUDA OOM occurred.

## What changed relative to the CPU-worker bottleneck

The former process-vectorized backend created one CPU worker per environment. In the earlier 128/256-env capacity run, CPU utilization was already about 93–95% at 128 envs, and 256 envs reached the 25.53/30 GiB RAM guard before GPU memory became limiting. The new backend reports `num_cpu_env_workers=0` in every measured case. At 4096 envs, host RAM remains about 3.8–3.9 GiB while measured GPU utilization is 82.9% for MASAC and 93.8% for MATD3. The old CPU process/RAM ceiling is therefore removed for this tested range.

The environment state itself scales approximately linearly on GPU: about 50.3 MiB at 128 envs and 1610.9 MiB at 4096 envs. End-to-end peak device memory remains only 2735 MiB for MASAC and 2577 MiB for MATD3 at 4096 envs on the tested T4, leaving substantial VRAM headroom.

## Training-rate caveat

This sweep intentionally kept `batch_size=512` and `updates_per_step=1` fixed to isolate environment scaling. Consequently, learner sample usage per newly collected transition falls from 4.0 at 128 envs to 0.125 at 4096 envs. These throughput results do **not** imply that a 4096-env long training run should keep one update per vector step. For equal replay-sample usage per new transition at 4096 envs with batch 512, eight optimizer updates per vector step would give a nominal ratio of 1.0. That learning-intensity choice should be validated separately; it was not changed in this capacity benchmark.

## Implementation notes

NVIDIA Warp is used here as the GPU kernel framework. The design keeps environment arrays and learner-facing tensors resident on CUDA and uses Warp/PyTorch interoperability rather than a CPU process per environment. This is conceptually aligned with GPU-resident MARL systems such as WarpDrive/VMAS/JaxMARL, but this project's packet/MAC/ACK/retry/TTL/custody semantics are project-specific and were independently implemented and tested against pinned native UavNetSim behavior.

Machine-readable evidence is in `docs/gpu-packet-results-20261014-evidence.json`.
