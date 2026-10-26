# Task 3 — matched BASE AMP / torch.compile benchmark

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Matched real-CUDA BASE benchmark for MASAC and MATD3 at the production G3 shape. Four cells changed only AMP/compile; fused Adam stayed OFF. No 20k training, resume, restart, or mutation of existing G3 production lineages.

Pinned source for all four kernels: `a26a67fd1ae6823ad4c33ca7eb0515fe4c1a821b`.
Benchmark harness source change: `e718305` (`--benchmark-steps`, `--benchmark-learner-updates`). Before launch the full offline suite passed: 806 passed, 95 skipped, 8 warnings, 1352 subtests, exit 0. Harness budget used for every cell: 32 collector steps and 20 measured learner updates.

Common shape: BASE, E4096, B8192, replay 1,000,000, hybrid CPU75/GPU25, 1 KiB report/payload, G3, seed44, 2xTesla T4, fused optimizers OFF.

## Kernels

| Cell | Account | Kernel |
| --- | --- | --- |
| off/off | account_07 | `kjhihgiugy/ben-g3-both-base-h75-10261751-c0faa9` |
| FP16/off | account_13 | `bjkjbhm/ben-g3-both-base-h75-10261750-576c72` |
| off/compile | account_14 | `weofjwelf/ben-g3-both-base-h75-10261751-214e43` |
| FP16/compile | account_24 | `bnvhvgv/ben-g3-both-base-h75-10261750-8d5f24` |

All four kernels reached Kaggle `complete`; all eight MASAC/MATD3 benchmark results are `passed`.

## MASAC results

| Cell | learner updates/s | vs off/off | collector transitions/s | warmup s | GPU1 peak allocated GiB | GPU1 peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off/off | 1.4644 | baseline | 4325.98 | 1.935 | 3.752 | 4.020 |
| FP16/off | 1.5141 | +3.39% | 4347.24 | 2.010 | 3.726 | 3.975 |
| off/compile | 1.4499 | -0.99% | 4336.29 | 8.111 | 3.667 | 3.785 |
| FP16/compile | 2.2682 | **+54.89%** | 4341.63 | 7.840 | **3.591** | **3.715** |

FP16/compile execution check: status passed, autocast FP16 marker=1, 32 critic +32 actor updates, 3 compiled graphs, 88 generated kernels, 0 graph breaks and 0 frame fallback.

## MATD3 results

| Cell | learner updates/s | vs off/off | collector transitions/s | warmup s | GPU1 peak allocated GiB | GPU1 peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off/off | 6.4011 | baseline | 4336.89 | 0.829 | 3.769 | 3.812 |
| FP16/off | 7.1200 | +11.23% | 4356.41 | 0.870 | 3.743 | 3.969 |
| off/compile | 6.4070 | +0.09% | 4352.63 | 6.860 | 3.701 | 3.844 |
| FP16/compile | 7.4457 | **+16.32%** | 4345.08 | 7.027 | **3.618** | 3.764 |

FP16/compile execution check: status passed, autocast FP16 marker=1, 32 critic +16 actor updates, 3 compiled graphs, 88 generated kernels, 0 graph breaks and 0 frame fallback.

## Interpretation

- Collector throughput is essentially unchanged across cells (all within about 0.5%), as expected because AMP/compile targets the learner path.
- FP16 alone gives a modest learner gain: +3.39% MASAC and +11.23% MATD3 in this matched run.
- Compile alone does not improve warmed learner throughput here and adds roughly 6 s of startup/warmup.
- FP16+compile is the best measured cell for both learners: +54.89% MASAC and +16.32% MATD3 versus off/off, while reducing GPU1 peak allocated memory by about 4.30% and 3.99% respectively.
- The compile warmup cost is one-time per process; long production runs can amortize it.
- Generic BASE benchmark `cuda_pipeline_verified=false` / `dual_gpu_verified=false` is expected for this hybrid layout because that helper requires replay device `cuda:1`; the actual devices are collector cuda:0, learner cuda:1, replay storage CPU with a GPU hot tier. Dedicated execution checks for accelerated cells passed.
- This is one matched run per cell, not a repeated statistical benchmark. It supports retaining FP16+compile as the candidate production setting; it does not establish confidence intervals or long-horizon learning-quality equivalence.

## Additional CUDA gates

Every cell: `test_cuda.py` 11/11, `test_cuda_scenario_v2.py` 4/4, mission regressions 8/8.

## Output hashes

- off/off MASAC `6b8c2328cc2b5636ab3c68c6851e10878460083cd09f906cca1570da46e81dff`; MATD3 `e57129ad1649082aa0c85cdfd1bee63774d240c9b3d6d76f64d1d520cae08e1d`; job `eef0e236ea545df925b9d041fb2229d488db00ce28f4ba3112148cb9d6f702e6`.
- FP16/off MASAC `b9d6311916d31b0913c504f5b09f246dcbec3fee439c8b61aa8ebe04ccd40494`; MATD3 `797163ac910dc1bc457677338e4efa2ba9993283fc81bfd9ee524ff0315e9aae`; job `ea7f9513e8bac1082a4feb07401a145139aa170e78d728e32e2ac08c034818f2`.
- off/compile MASAC `3bf852e9d1e53d86368bcd306817d00a07b39015e740c0824589ef752518c7d4`; MATD3 `c0325789e3c1a91b814fc0bcc45df70860b43f51b0111614ffa51a6617119b67`; job `d8f40108bf155c87b5ad74f93f1d6151b6260ff95c2c588e3d9802b94a3962af`.
- FP16/compile MASAC `a520c770987c87a1161878c57490b8deece8fa0cd2c4ce50d0aa6a1007c1f192`; MATD3 `4e157ecb2ecd1334e17131f71729eaecfc90c0af634da3fe5843ad864a07b8c3`; job `f3d67c7c59e5fb1ed5de57653f23916d1320387fba0bd608b2773e97691dcfcb`.
- `hardware.json` is identical across downloaded cells: `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`.

## Next task

Task 4 is the same four-cell comparison for the async PQL path before making a final BASE+PQL production recommendation.
