# 1 KiB PQL AMP + fused combination check — 2026-10-24

Kaggle T4 x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` version 11 completed from runner commit `c8ca18a58272a2c72def14cfd6d9b4f8651113af`. No production checkpoint or live lineage was modified. `torch.compile` remained disabled in all cases.

| Algorithm | Baseline | AMP only | AMP + fused |
| --- | ---: | ---: | ---: |
| MASAC PQL | 2219.19 | 2439.91 (**1.099x**) | 2356.10 (**1.062x**, 0.966x vs AMP-only) |
| MATD3 PQL | 3509.02 | 3457.48 (**0.985x**) | 3500.10 (**0.997x**, 1.012x vs AMP-only) |

## Ruling

- Do not add fused Adam to AMP for MASAC PQL: it reduced median throughput versus AMP-only.
- MATD3 PQL remains effectively baseline-speed; neither AMP-only nor AMP+fused shows a robust promotion signal in this matched run.
- `torch.compile` was already rejected for PQL by the previous one-variable isolation.
- Therefore there is no single accelerated PQL policy to promote globally. Preserve the current four non-AMP production lineages.
- If a future fresh MASAC-only cohort is launched, AMP-only is the only PQL acceleration still supported by repeated measurements; it must start from episode 0 and use the same precision policy across the compared MASAC cases if numerical fairness is required.
