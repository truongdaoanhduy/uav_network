# 1 KiB production cohort launch and live throughput — 2026-10-24

## Decision carried forward

Primary future training uses `configs/comparison-20k-hybrid75-1kb.yaml`:
- report_bytes = 1024
- payload_bytes = 1024
- buffer_bytes = 3072
- num_envs = 4096
- episodes = 20000
- max_steps = 1800
- replay capacity = 1,000,000
- batch = 8192
- replay CPU fraction = 0.75
- backend = warp_packet

Existing 1 MB lineages remain immutable historical/stress-test evidence and are not resumed into 1 KiB runs.

## Fresh 1 KiB lineages

All four runs started from episode 0 on source commit `0c9fbc5b93acfd3008f9802950570849bc42997c`.

| Case | Account | Kaggle GPU kernel | W&B run | Controller | Health |
| --- | --- | --- | --- | --- | --- |
| MASAC BASE | account_01 | `duytrngoanh/uavnet-train-20261024-073119-a53a63` | `7f985331` | `duytrngoanh/uav-flow-22c9acce21eef2-0000` | running, unattended_ready=true |
| MASAC PQL | account_03 | `vuliu123456/uavnet-train-20261024-074226-5b2381` | `e0bfbcaf` | `vuliu123456/uav-flow-7dbb76c688e6f5-0000` | running, unattended_ready=true |
| MATD3 BASE | account_06 | `libiu123/uavnet-train-20261024-074821-e57c46` | `767cb8da` | `libiu123/uav-flow-f3d6a5b7d68d52-0000` | running, unattended_ready=true |
| MATD3 PQL | account_09 | `nhtbrg/uavnet-train-20261024-080451-eda89d` | `30a40bf7` | `nhtbrg/uav-flow-2d26ab05b1b8af-0000` | running, unattended_ready=true |

Controller health was re-checked after adoption for all four lineages.

## Ambiguous-submit cleanup

A Kaggle/external-service 502 caused the MATD3 BASE submit response to be ambiguous. Reconciliation found two live identical 1 KiB MATD3 BASE roots on account_06. The older run `767cb8da` had progressed farther and was retained. The accidental newer duplicate `a379f8bd` and its controller `libiu123/uav-flow-10ea62cf6a6c06-0000` were deleted. The retained GPU root and controller remained running after cleanup.

## 62-second live throughput sample

A fresh W&B delta sample over 62.025 s measured:

| Case | Delta global_step | Live global steps/s |
| --- | ---: | ---: |
| MASAC BASE | 49,152 | 792.45 |
| MASAC PQL | 69,632 | 1,122.64 |
| MATD3 BASE | 208,896 | 3,367.93 |
| MATD3 PQL | 151,552 | 2,443.40 |

All four W&B runs were `running` and had `report_bytes=1024`, `payload_bytes=1024`, `num_envs=4096`, and target episodes=20000.

## Comparison with canonical 1 MB roots

Historical root average rates:
- MASAC BASE `63faeffb`: 191.70 steps/s
- MASAC PQL `fc14508c`: 193.76 steps/s
- MATD3 BASE `2dcbf36c`: 104.34 steps/s
- MATD3 PQL `278634f6`: 95.58 steps/s

Using the 62-second live 1 KiB rates, observed speedups versus the corresponding 1 MB root average are approximately:
- MASAC BASE: 4.13x
- MASAC PQL: 5.79x
- MATD3 BASE: 32.28x
- MATD3 PQL: 25.56x

These are early live measurements, not final run-wide averages.

## Bottleneck shift

Fresh cumulative timing summaries after the 1 KiB change:
- MASAC BASE: learner 2223.96 s vs collector wait 30.32 s
- MATD3 BASE: learner 999.95 s vs collector wait 83.17 s
- MASAC PQL: collector 557.38 s, collector wait 1114.48 s, critic 1630.87 s, policy 1373.09 s
- MATD3 PQL: collector 187.58 s, collector wait 32.36 s, critic 117.42 s, policy 34.47 s

The dominant 1 MB packet/collector bottleneck is no longer the main BASE bottleneck. For BASE, learner time now dominates. PQL is more mixed/overlapped, but the 1 MB collector domination is gone.

## Rough 20k ETA from historical 1 KiB episode/step relation

Using old 25k 1 KiB runs to scale an approximate 20k global-step budget, then dividing by the current 62-second live rate:
- MASAC BASE: ~11.87 h
- MASAC PQL: ~8.77 h
- MATD3 BASE: ~3.30 h
- MATD3 PQL: ~4.53 h

These are rough early estimates only. Rates can change after additional learning/network activity and episode waves. The important conclusion is that 1 KiB removes the catastrophic 1 MB packet-level cost. MASAC BASE is now the case most likely to need either continuation or learner-side acceleration.

## Next optimization task

Do not mutate the four running production lineages. The next controlled optimization target is learner-side acceleration (AMP/compile/fused optimizers) on a spare account, because the 1 KiB BASE timings now show the learner rather than the packet collector as the dominant cost.
