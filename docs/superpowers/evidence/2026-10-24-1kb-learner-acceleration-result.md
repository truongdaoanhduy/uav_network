# 1 KiB MASAC learner acceleration benchmark — 2026-10-24

## Scope and provenance

- Kaggle kernel: `bjkjbhm/uav-1kb-learner-accel-20261024`, version 7, T4 x2.
- Runner commit: `89440b01862cece2bfc1ee9c3615d0e2e269061b`.
- Production code under test is byte-equivalent to parent `fbbb5569ce6520004bf406de5829fd2ac4696875` except the added benchmark runner; the JSON's `source_commit` field intentionally identifies that production-code parent.
- Config shape: report/payload 1024 B, 4096 envs, batch 8192, replay 1,000,000, CPU replay fraction 0.75, MASAC BASE learner on GPU1.
- Replay was filled to exactly 1,000,000 rows once before all cases: 9,831,000,000 CPU bytes + 3,274,000,000 hot-GPU bytes = 13,105,000,000 bytes.
- Timed region is production-shaped `replay.sample(8192) + learner.update(..., collect_metrics=False)`. Two interleaved repetitions per case; 3 warmup updates before each timed block of 20 updates.
- This is a throughput/correctness benchmark, not a convergence or policy-quality result. Existing production 1 KiB lineages were not modified.

## Results

| Case | Median update | Speedup vs baseline | Learner-time reduction | Median warmup |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 566.02 ms | 1.000x | 0.0% | 2.26 s |
| AMP FP16 only | 534.07 ms | 1.060x | 5.65% | 1.75 s |
| `torch.compile` only | 534.69 ms | 1.059x | 5.54% | 14.80 s |
| fused Adam only | 577.57 ms | 0.980x | -2.04% | 1.72 s |
| AMP + compile + fused | **353.78 ms** | **1.600x** | **37.50%** | 9.84 s |

The combined configuration is strongly non-additive: fused Adam alone is slightly slower, while the full combination is substantially faster than either AMP-only or compile-only.

## Correctness / execution evidence

- All rows retained finite learner parameters.
- AMP cases: critic, actor, and alpha each applied 23/23 scaler-mediated steps with 0 skips in both repetitions.
- compile-only: 3 unique compiled graphs and 78 generated Inductor kernels per repetition.
- all-three: 3 unique compiled graphs and 88 generated Inductor kernels per repetition.
- fused cases: every Adam parameter group reported `fused=True`.
- Full-combination GPU1 peak allocated memory was ~5.55 GB versus ~6.62 GB baseline in these runs.

## Decision

The performance gate passes for the **combined** AMP FP16 + `torch.compile` + fused Adam configuration. Do not promote fused Adam by itself. Do not mutate or resume an existing non-AMP production lineage with AMP/compile/fused enabled because precision/runtime checkpoint invariants differ. Use the combined flags only for a fresh matched cohort or a future experiment where all compared algorithms/cases start from episode 0 with the same runtime acceleration policy.
