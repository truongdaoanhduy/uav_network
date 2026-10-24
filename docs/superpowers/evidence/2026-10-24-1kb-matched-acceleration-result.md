# 1 KiB matched acceleration benchmark — 2026-10-24

## Scope

Kaggle T4 x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` v8, runner commit `b42a9a749770acbf518bac84389ef5a8041768e9`. No production checkpoint or production lineage was modified. All cases use the primary 1 KiB config with env4096, batch8192, replay1M and hybrid CPU75/GPU25 replay.

The accelerated case is AMP FP16 + `torch.compile` + fused Adam. Two repetitions were run for baseline and accelerated cases.

## Result

| Path | Baseline median | Accelerated median | Speedup |
| --- | ---: | ---: | ---: |
| MATD3 BASE learner update | 158.82 ms | 139.29 ms | **1.140x** |
| MASAC PQL pipeline | 2407.09 transitions/s | 2453.36 transitions/s | **1.019x** |
| MATD3 PQL pipeline | 3509.38 transitions/s | 3272.39 transitions/s | **0.932x** |

MASAC PQL has substantial two-run variation and the +1.9% median gain is too small to call a robust promotion. MATD3 PQL is consistently slower with the combined acceleration flags. MATD3 BASE benefits moderately, but the current unaccelerated MATD3 production run is already fast and healthy.

## Ruling

Do not enable AMP+compile+fused globally across all four algorithms/BASE-PQL paths. Preserve the current unaccelerated 1 KiB production cohort for scientific consistency. The previous MASAC BASE result (1.600x learner speedup) remains valid for that path, and MATD3 BASE shows a smaller 1.140x learner speedup, but PQL requires path-specific diagnosis before any promotion.

Next diagnostic: isolate AMP-only, compile-only and fused-only on MASAC PQL and MATD3 PQL with the same env4096/batch8192/replay1M pipeline. The aim is root-cause identification, not a production change.
