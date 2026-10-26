# Task 4 — matched async-PQL AMP / torch.compile benchmark

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Matched real-CUDA async-PQL benchmark for MASAC and MATD3 at the production G3 shape. Four cells changed only AMP/compile; fused Adam stayed OFF. No production 20k training, exact resume, restart, or mutation of existing G3 lineages.

Pinned source for all four kernels: `d12dae57b0ce3848e58de39f96b0c0335f21dac0`.
Common shape: PQL async, E4096, B8192, replay 1,000,000, hybrid CPU75/GPU25, 1 KiB report/payload, G3, seed44, 2xTesla T4, fused optimizers OFF.

The BASE-only manifest fields `benchmark_steps=2` / `benchmark_learner_updates=2` do not control this path. `pql_benchmarks()` selected the real PQL benchmark budget; every inner report records `steps=40`, 72 critic updates, and policy updates according to each algorithm's delay.

## Kernels and W&B runs

| Cell | Kernel | MASAC run | MATD3 run |
| --- | --- | --- | --- |
| off/off | `kjhihgiugy/ben-g3-both-pql-h75-10261802-c0b8a9` | `33cd579c` | `325ee75f` |
| FP16/off | `bjkjbhm/ben-g3-both-pql-h75-10261802-9f8f8a` | `47439c16` | `e64df3ad` |
| off/compile | `weofjwelf/ben-g3-both-pql-h75-10261802-0a7c81` | `c67f57fc` | `7640c24b` |
| FP16/compile | `bnvhvgv/ben-g3-both-pql-h75-10261802-10a522` | `289c297d` | `b627d81b` |

All four kernels reached Kaggle `complete`; all eight algorithm benchmark results are `passed`; all eight inner reports have `cuda_pipeline_verified=true` with collector cuda:0 and critic/policy cuda:1.

## MASAC PQL results

| Cell | transitions/s | critic updates/s | vs off/off | elapsed s | GPU1 alloc GiB | GPU1 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off/off | 1847.19 | 0.8118 | baseline | 88.697 | 4.458 | 5.000 |
| FP16/off | 2103.01 | 0.9242 | +13.85% | 77.908 | 4.274 | 4.645 |
| off/compile | 2088.43 | 0.9178 | +13.06% | 78.451 | 4.358 | 4.629 |
| FP16/compile | **2459.05** | **1.0806** | **+33.12%** | **66.627** | **3.961** | **4.248** |

Actor updates/s follows the same ratio because MASAC performed 72 critic and 72 actor updates. FP16/compile reduced GPU1 peak allocated memory by about 11.15% and reserved memory by about 15.04% versus off/off.

The FP16 cells report autocast marker=1, no AMP step skips, finite scaler=65536. The FP16/compile execution check passed with 3 compiled graphs, 88 generated kernels, 0 graph breaks and 0 frame fallback.

## MATD3 PQL results

| Cell | transitions/s | critic updates/s | vs off/off | elapsed s | GPU1 alloc GiB | GPU1 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off/off | 3332.40 | 1.4644 | baseline | 49.166 | 3.983 | 4.660 |
| FP16/off | **3473.82** | **1.5266** | **+4.24%** | **47.164** | 3.958 | 4.705 |
| off/compile | 3041.76 | 1.3367 | -8.72% | 53.864 | 3.916 | 4.551 |
| FP16/compile | 3121.69 | 1.3718 | -6.32% | 52.484 | **3.843** | **4.268** |

MATD3 kept its 2:1 critic:actor update ratio (72 critic, 36 actor), so actor updates/s changed by the same percentages. FP16/compile saves memory but is slower in this single matched async run. FP16/off is the fastest measured MATD3 PQL cell.

The accelerated execution checks all passed. FP16/compile has autocast marker=1, 3 compiled graphs, 88 generated kernels, 0 graph breaks/fallback, and no AMP skips.

## Interpretation

- MASAC PQL strongly favors FP16+compile in this run: +33.12% throughput and materially lower GPU1 memory.
- MATD3 PQL does **not** show the same result: compile is slower here; FP16-only is +4.24% over off/off while FP16+compile is -6.32%.
- Therefore a single global claim that compile improves all PQL learners is not supported.
- This is one matched run per cell. Async PQL timing can vary with collector/learner scheduling, so the MATD3 compile regression must be repeated before changing production policy per algorithm.
- Existing canonical profile remains unchanged during this evidence task; no production job is restarted or resumed with new flags.

## Additional CUDA gates

Every cell passed: CUDA 11/11, scenario CUDA 4/4, mission regressions 8/8, PQL smoke 11/11.

## Output hashes

- off/off: MASAC `b3d3b1a09ab99854b82fc788bbeeb48da230c6b0fb44750341e5c7ba95e9d912`; MATD3 `fecdff490cf425795b91ebe12c6f32910900004506ea7100cfda60b999e7eb7a`; job `74b193166fbafb858ab696a4f638cd160cfee1b037ddab9deff4c0a3138060d3`.
- FP16/off: MASAC `753f6917bfc887754f05a40ee18bb469f6e1bae518c9d402585d1c87228d0348`; MATD3 `e6fd5e9482f28139ac5dea8809d9b3f2e04c6da804f15d54bc9bb27b5636da35`; job `f14271baad4e1ddeb33de0f492542cea233ed12aa780f2e609ddc02d32eff87c`.
- off/compile: MASAC `f17ec81b4789fdeb35d4e7f6dd1871767287459f3420c681458b07299e339098`; MATD3 `b98de20b269899690134ddc192d51ba8b246ff220f558c99cc76f8434b68c9e4`; job `ee5006b2d3d7555705c993a8e9014072d2a335b99168838503a572978f079c41`.
- FP16/compile: MASAC `50d804daf44d3d7107f6515866a0d6378c157a0c393e84ccbd4239d3d1c8a05f`; MATD3 `94498b2bbf949e8e2a495302c445a93bc5fc29dd2f97f6718995c9551f339e27`; job `b1b146fd3dd9ef9692c03dfbb1be3aa7d6da59b867722b33b37e4e8341238cfd`.
- `hardware.json` identical across cells: `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`.

## Next task

Task 5 repeats the relevant MATD3 PQL comparison (FP16/off vs FP16/compile) to determine whether the measured compile slowdown is reproducible before setting a per-algorithm production policy.
