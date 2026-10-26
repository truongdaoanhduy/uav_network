# Task 5 — repeated MATD3 PQL compile check

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Question

Task 4 found MATD3 async-PQL faster with FP16 only than FP16 + `torch.compile`. Task 5 tests whether that result repeats before changing production policy.

Common shape: MATD3 PQL async, E4096, B8192, replay 1,000,000, hybrid CPU75/GPU25, 1 KiB, G3, seed44, 2xTesla T4, AMP FP16 ON, fused Adam OFF. The only comparison variable is `runtime.compile_models`.

Task 4 supplies replicate 1. Task 5 adds two new replicates per mode, so each mode has n=3.

Task 5 source: `89097f05bced2d633c13f84731987a620b1c7b3e`.

## New replicate kernels

| Mode | Replicate | Kernel | W&B run |
| --- | --- | --- | --- |
| FP16/off | r2 | `kjhihgiugy/ben-g3-matd3-pql-h75-10261816-12f5ae` | `5a289110` |
| FP16/off | r3 | `bjkjbhm/ben-g3-matd3-pql-h75-10261817-158aab` | `645edba5` |
| FP16/compile | r2 | `weofjwelf/ben-g3-matd3-pql-h75-10261817-7c0d33` | `405ecee3` |
| FP16/compile | r3 | `bnvhvgv/ben-g3-matd3-pql-h75-10261817-94417f` | `192e5488` |

All four new kernels reached `complete`; each benchmark report is `passed` and `cuda_pipeline_verified=true`.

## Replicate results

### FP16, compile OFF

| Replicate | transitions/s | critic updates/s | actor updates/s | elapsed s | GPU1 alloc GiB | GPU1 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task4 r1 | 3473.82 | 1.5266 | 0.7633 | 47.164 | 3.958 | 4.705 |
| Task5 r2 | 3383.90 | 1.4871 | 0.7435 | 48.418 | 3.973 | 4.705 |
| Task5 r3 | 3535.33 | 1.5536 | 0.7768 | 46.344 | 3.967 | 4.705 |
| **mean** | **3464.35** | **1.5224** | **0.7612** | **47.308** | **3.966** | **4.705** |
| median | 3473.82 | 1.5266 | 0.7633 | 47.164 | 3.967 | 4.705 |

Critic throughput stdev: 0.0335 updates/s.

### FP16, compile ON

| Replicate | transitions/s | critic updates/s | actor updates/s | elapsed s | GPU1 alloc GiB | GPU1 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task4 r1 | 3121.69 | 1.3718 | 0.6859 | 52.484 | 3.843 | 4.268 |
| Task5 r2 | 3124.26 | 1.3730 | 0.6865 | 52.441 | 3.843 | 4.268 |
| Task5 r3 | 2950.37 | 1.2965 | 0.6483 | 55.532 | 3.843 | 4.332 |
| **mean** | **3065.44** | **1.3471** | **0.6736** | **53.486** | **3.843** | **4.289** |
| median | 3121.69 | 1.3718 | 0.6859 | 52.484 | 3.843 | 4.268 |

Critic throughput stdev: 0.0438 updates/s.

## Result

Using the n=3 means, enabling compile on top of FP16 changes MATD3 PQL:

- transitions/s: 3464.35 -> 3065.44, **-11.51%**
- critic updates/s: 1.5224 -> 1.3471, **-11.51%**
- actor updates/s: 0.7612 -> 0.6736, **-11.51%**
- elapsed benchmark time: 47.308 -> 53.486 s, **+13.06%**
- GPU1 peak allocated: 3.966 -> 3.843 GiB, **-3.11%**
- GPU1 peak reserved: 4.705 -> 4.289 GiB, **-8.84%**

Most importantly, the fastest compile run (3124.26 transitions/s) is still slower than the slowest compile-off run (3383.90 transitions/s). The slowdown therefore repeated across all three observations rather than depending on one outlier.

All new execution checks passed. FP16 marker=1 in every case; compile-on runs produced 3 graphs / 88 kernels with 0 graph breaks and 0 frame fallback; all AMP optimizer groups had zero skipped steps. Each new kernel also passed CUDA 11/11, scenario CUDA 4/4, mission regressions 8/8, and PQL smoke 11/11.

## Production implication

For MATD3 PQL on this T4/G3 shape, **FP16 with compile OFF** is the measured production candidate. Compile saves some VRAM but costs about 11.5% throughput in the repeated sample.

Combined with Tasks 3–4, the current measured candidate policy is:

- MASAC BASE: FP16 + compile ON
- MATD3 BASE: FP16 + compile ON
- MASAC PQL: FP16 + compile ON
- MATD3 PQL: FP16 + compile OFF
- fused Adam: remain OFF; it has not been benchmarked in this sequence.

BASE and MASAC-PQL acceleration choices are still based on one matched run per cell; only the MATD3-PQL compile decision has n=3 confirmation.

## New output hashes

- FP16/off r2: benchmark `b8800a64527dc7d0169c25274926f48f1201fdc2ca62ec1cf400c13706478891`; execution `01587c8d3a7e7dd3edb45a193265c4d9f6ef86b6451f0747b2b820e11fa26a47`; job `8cd24e1e12bd8ef8cfa023d2c0705032698222d7025d7bdfe20dff8116a6d825`.
- FP16/off r3: benchmark `d328b0b2db4d984cd31397c548e0eded59104aa7b9525ea2223c8dff6644e6a9`; execution `11b82e4aa3685c11a3b2802f4a887a6edcc39495895ac5484bac4a8bd22e73b5`; job `4f83a6ef68ff4fd967175434a6e5e86fc9ed8409d5a3bede10ab5de0a488559e`.
- FP16/compile r2: benchmark `bcb7442589110fa94ba7fffd8d18f477c411fe6af7e938477a5ebbb87c9b0204`; execution `1373b859bf805378712f48f24ea523cfb4d14cb086952fcda6df56735e884b13`; job `8d16ad535f2465dcdfb01cc2d7f4f97c90ca60b3071a37bdee6779379c11acb5`.
- FP16/compile r3: benchmark `52ada99c802a1179f4614a400bbe9e505abe00e4838d7ebf0a46959d65072912`; execution `74e370404e1f09d069d4f7f01a61111be8f3c37a55be10fc522f485f7c3c4d2c`; job `622823840d07f4f98d07ec9394cab80f7a074bf60f21698d9f88a32187495c16`.

## Next task

Task 6 encodes the measured acceleration policy into the production launch/config path with tests, so MATD3 PQL does not accidentally inherit compile ON while the other measured modes retain it.
