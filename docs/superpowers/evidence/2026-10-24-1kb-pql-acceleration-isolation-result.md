# 1 KiB PQL acceleration isolation — 2026-10-24

## Scope

Kaggle T4 x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` v9, runner commit `1b2ddc917796fbf8953bb3e959e58d60b91b282e`. Primary 1 KiB shape: env4096, batch8192, replay1M, CPU75/GPU25. Two repetitions per case. Tracking disabled; no production checkpoint or production lineage modified.

## Results

### MASAC PQL

| Case | Median transitions/s | Speedup vs baseline | Median critic active | Median policy |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2223.38 | 1.000x | 37.82 s | 34.90 s |
| AMP FP16 only | **2388.55** | **1.074x** | 35.18 s | 34.16 s |
| compile only | 1964.83 | **0.884x** | 51.51 s | 36.86 s |
| fused Adam only | 2248.61 | 1.011x | 39.40 s | 35.68 s |

### MATD3 PQL

| Case | Median transitions/s | Speedup vs baseline | Median critic active | Median policy |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3482.79 | 1.000x | 9.38 s | 2.66 s |
| AMP FP16 only | 3543.92 | 1.018x | 9.11 s | 3.09 s |
| compile only | 3199.16 | **0.919x** | 14.53 s | 6.01 s |
| fused Adam only | 3473.25 | 0.997x | 9.30 s | 2.56 s |

## Root-cause ruling

`torch.compile` is the clear harmful component on the current asynchronous PQL path: both algorithms lose pipeline throughput and spend substantially more time in critic/policy work. Fused Adam is effectively neutral at pipeline level. AMP-only is a modest positive for MASAC PQL and a very small positive for MATD3 PQL, but neither justifies changing an in-flight scientific cohort.

The prior all-three result is therefore explained: MASAC PQL's AMP benefit is largely cancelled by compile overhead, while MATD3 PQL becomes slower. Do not globally promote compile/fused for PQL. Keep the current four production lineages unaccelerated and internally consistent.
