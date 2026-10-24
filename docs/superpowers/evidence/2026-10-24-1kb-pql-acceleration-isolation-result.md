# 1 KiB PQL one-variable acceleration isolation — 2026-10-24

Kaggle T4 x2 diagnostic: `bjkjbhm/uav-1kb-learner-accel-20261024`, using runner commit `1b2ddc9`. No production checkpoint or live production lineage was modified.

Primary shape: report/payload 1024 B, env4096, batch8192, replay1M, CPU75/GPU25 hybrid replay, PQL enabled. Two interleaved repetitions per case.

## Median throughput

| Algorithm | Baseline | AMP only | compile only | fused only |
| --- | ---: | ---: | ---: | ---: |
| MASAC PQL | 2223.38 | 2388.55 (**1.074x**) | 1964.83 (**0.884x**) | 2248.61 (**1.011x**) |
| MATD3 PQL | 3482.79 | 3543.92 (**1.018x**) | 3199.16 (**0.919x**) | 3473.25 (**0.997x**) |

## Ruling

- `torch.compile` is rejected for PQL on this workload: it slowed both algorithms materially.
- fused Adam alone is not promoted: effect was approximately neutral.
- AMP FP16 is the only single variable that improved both PQL paths, with a meaningful +7.43% on MASAC PQL and a small +1.76% on MATD3 PQL.
- Current production 1 KiB lineages remain unchanged because changing precision/runtime flags mid-lineage would violate checkpoint/runtime invariants.
- The only remaining justified combination is **AMP + fused Adam, compile off**, tested against AMP-only and baseline on the same spare T4x2 account.
