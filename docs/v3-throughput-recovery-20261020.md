# V3 1800-step Warp packet throughput recovery — 2026-10-20

## Problem observed in the first production handoff

The first fresh 1800-step MASAC BASE production run used source `aec3724`, 4096 Warp packet environments, target 20,000 completed episodes, seed 44, and two visible T4 GPUs. It never completed an episode. Its final W&B state is `crashed` at global step 380,928 / vector step 93 with `train/episodes_completed=0`.

The live rate fell to about 31.61 joint transitions/s as report traffic increased. System telemetry showed GPU 0 (Warp collector) at 100% utilization while GPU 1 (learner/replay) was mostly idle between collections, so the bottleneck was the deterministic packet scheduler rather than the learner.

## Scheduler-only optimization

Two commits optimize internal Warp event scheduling without changing the learning algorithms, reward, packet timestamps, MAC/PHY parameters, retry/TTL rules, report size, queue limits, or the 1800-step scenario:

- `40d53e3` — replace pop+push heap churn for self-rescheduling idle events and keep stationary link gain cache valid inside one packet step.
- `8856c02` — batch logically repeated idle POLL/LISTEN reschedules while advancing the logical event sequence and diagnostic counters exactly as the skipped individual reschedules would have done.

The production learner/training code and configs are unchanged by these commits.

## T4 A/B evidence

A matched 4096-environment T4 benchmark compared base `b33f1ab` with candidate `8856c02` using the exact V3 1800-step config and one active 1 MB report per world. Three timed active packet steps produced:

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Mean packet-step wall time | 71.237 s | 13.668 s |
| Vector steps/minute | 0.842 | 4.390 |
| Speedup | — | **5.212×** |

The two executions had identical aggregate logical `event_sequence`, packet counters, and all network diagnostic counters. The benchmark ran on a Tesla T4 with 4096 worlds and allocated the same 2,353,823,744 bytes of Warp state in both cases.

Evidence: `outputs/session-relay-20261020/kaggle-packet-scheduler-t4-final/output/result.json`.

## Regression gates

Focused packet/physics/vector/training-audit/exact-resume verification after the optimization: **74 passed, 12 skipped, 16 subtests**.

Fresh full repository suite at final source: **527 passed, 70 skipped, 1185 subtests passed**, exit 0.

## Replacement production handoff

Because the old run had zero completed episodes and crashed, no learner/world state is resumed from it. A fresh run was submitted from source `8856c02` with the same scientific profile:

- config `configs/comparison-20k-steps1800-scenario-v3.yaml`
- MASAC BASE; seed 44; evaluation seed 2000000044
- 4096 environments; 20,000 completed-episode target; max_steps 1800
- 35 targets; 25 obstacles; 1 MB reports; 3 MB buffers
- fresh learner / no warm-start artifact
- two visible T4 GPUs requested
- auto-continuation and final CPU visualization enabled
- Kaggle: `duytrngoanh/uavnet-validate-20261020-103631-39e2ac`
- W&B reserved run ID: `0a296462`

At the latest recorded check the Kaggle job is `QUEUED`; therefore no W&B trainer metrics exist yet and no claim is made that the optimized production trainer has started or that the 20,000-episode budget is complete. The queued job does not consume GPU quota until allocation.
