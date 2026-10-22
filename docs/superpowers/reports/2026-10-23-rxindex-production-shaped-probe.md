# Receiver Pending Index — Production-Shaped Decision

## Scope

This closes the production-shaped gate for the experimental `rxindex` packet scheduler. The task asks whether the small Task-5 packet-scheduler microbenchmark gain survives inside a real MASAC BASE collector/learner workload strongly enough to justify a later production-promotion gate.

`rxindex` remains experimental throughout this task. Gen1 is unchanged and remains the production/oracle scheduler. No production training run was stopped, restarted, migrated, or modified.

## Frozen candidate and hardware

- Experimental branch: `perf/receiver-pending-index-20261022`
- Candidate/report SHA: `f87c2d3469ce2864007f5a4214fa7e1d8c8eb217`
- Scenario profile: `configs/comparison-20k-steps1800-scenario-v3.yaml`
- Kaggle account: `account_27`
- Kernel: `ioeufwejewlfl/uav-rx-minprod-20261022-190444`
- Hardware: 2× Tesla T4
- GPU0: collector/environment
- GPU1: learner + replay

The earlier full production-shaped attempt was not usable: the 3-hour job stopped during Gen1 warmup at 100/200 vector steps and never produced `gen1.json`, `rxindex.json`, or an aggregate A/B result. Per the approved fallback, this gate therefore used the minimal production-shaped probe rather than launching another full 3-hour attempt.

## Minimal production-shaped shape

Both Gen1 and `rxindex` ran sequentially in the same Kaggle session with identical configuration:

- MASAC BASE
- Scenario V3, natural seeded mission state
- environments: 2,048
- batch size: 4,096
- replay capacity: 1,000,000
- configured `learning_starts`: 409,600
- configured `updates_per_step`: 16
- AMP: FP16
- `torch.compile`: enabled
- fused optimizer: enabled
- vectorized critics: enabled
- collector overlap: enabled
- report size: 1 MB
- max steps: 1,800
- seed: 44

The fallback deliberately does **not** run 200 vector warmup steps. It fills replay using the natural Scenario V3 workload until replay is at least one batch and real packet activity is observed. In this run both schedulers satisfied that gate after four vector steps (8,192 transitions), with exactly matching network counters. One real learner update is then executed outside the measured region to warm Torch compile/AMP/fused optimizer. The measured region contains two collector/learner overlap cycles with 16 real learner updates per cycle.

This is a performance probe, not a learning-quality experiment. The configured production `learning_starts=409,600` is preserved in the resolved configuration, but the explicit benchmark learner updates are intentionally invoked before that threshold so the gate can measure the production learner path without another multi-hour replay-fill phase.

## Activity and scientific parity gate

The benchmark refused an empty-network workload. At the end of natural prefill, both schedulers had exactly:

- generated packets: 519,666
- received packets: 468,965
- PHY successes: 940,129
- PHY failures: 11,548
- collisions: 11,232
- payload bytes received: 480,220,160
- expired packets: 0

After the measured cycles, both schedulers again matched exactly:

- generated packets: 790,132
- received packets: 729,806
- PHY successes: 1,463,175
- PHY failures: 20,825
- collisions: 20,509
- payload bytes received: 747,321,344
- expired packets: 0
- done worlds: 0

Aggregate result: `scientific_parity = true`.

The gate therefore found no evidence that `rxindex` changed packet-level scientific behavior in this production-shaped workload.

## End-to-end measured result

Cold compile/prefill time is **not** used for the scheduler decision because the first arm pays shared Warp/Torch compilation and the second arm reuses caches. The decision uses only the post-warm measured overlap region.

| Metric | Gen1 | rxindex | rxindex effect |
| --- | ---: | ---: | ---: |
| Steady wall time, 2 cycles | 63.580452 s | 62.650999 s | -0.929453 s (-1.4619%) |
| Joint transitions/s | 64.422316 | 65.378047 | +1.4835% |
| Mean cycle wall time | 31.790148 s | 31.325416 s | -1.462% |
| Mean learner time/cycle | 2.812129 s | 2.876448 s | +2.29% slower |
| Mean collector wait/cycle | 28.927608 s | 28.420474 s | -1.75% |
| Mean replay add/cycle | 0.005144 s | 0.005362 s | effectively neutral |
| Mean actor refresh/cycle | 0.004320 s | 0.005361 s | negligible absolute cost |

The measured collector path is still the dominant part of the cycle: about 28–29 seconds of collector wait versus about 2.8–2.9 seconds of learner work. `rxindex` reduces that collector-side wait slightly, but the resulting end-to-end improvement is only about 1.5%.

## Memory

Environment-state allocation:

- Gen1: 1,190,191,104 B
- `rxindex`: 1,283,833,856 B
- delta: 93,642,752 B = 89.3047 MiB
- increase: +7.8679%

Replay allocation is identical at 13,105,000,000 B.

GPU1 stayed close to capacity in both cases because replay 1M is resident on the learner GPU:

- Gen1 PyTorch peak allocated: 91.55% of T4; peak reserved: 94.69%
- `rxindex` PyTorch peak allocated: 91.54% of T4; peak reserved: 94.96%

No OOM occurred. The extra `rxindex` state is simulator/Warp state on the collector side, consistent with Task 5's previously measured +89.30 MiB receiver-index cost.

## Relationship to Task 5 microbenchmark

Task 5 previously measured repeated active-streaming packet-scheduler throughput at only +0.5942% for `rxindex`, while adding the same +89.30 MiB (+7.87%) simulator state. The production-shaped overlap probe now measures +1.4835% joint transitions/s / -1.4619% steady wall time.

The production-shaped result is directionally positive, but still small. It does not reveal a hidden large training-speed benefit that would justify carrying additional scheduler state and promotion risk.

## Decision

**KEEP GEN1. Do not production-promote `rxindex`.**

Rationale:

1. scientific parity passed;
2. there was no OOM;
3. `rxindex` reduced measured end-to-end wall time by only about 1.46%;
4. repeated Task-5 packet-only gain was only +0.5942%;
5. the optimization still adds 89.30 MiB / +7.87% environment state;
6. the observed gain is too small to justify a production-promotion gate or merge.

`rxindex` should remain available only as an experimental/reference implementation. Gen1 remains the production scheduler and rollback/oracle implementation.

This closes the receiver-pending-index optimization task. The next performance work, if explicitly requested later, should return to the remaining Task-4 hotspots (`emit()` free-slot scans, packet scans, and stale/no-op event pressure) rather than promoting `rxindex`.

## Evidence

- Aggregate: `outputs/production-shaped-rxindex-probe/minimal/downloaded/production-shaped-probe.json`
- Gen1: `outputs/production-shaped-rxindex-probe/minimal/downloaded/gen1.json`
- rxindex: `outputs/production-shaped-rxindex-probe/minimal/downloaded/rxindex.json`
- Probe log: `outputs/production-shaped-rxindex-probe/minimal/downloaded/probe.log`
- Aggregate SHA-256: `ed5b06b68c2a3170bf70d90f2be9f57b7bc39a1c1a7f1dd2ee95b9c1d0502c18`
- Gen1 SHA-256: `12a6570e1e686223173fe2e1ad037b8e4046783ad88aa43c8d10ba183e90c9dc`
- rxindex SHA-256: `c32df06bf2362c0bda0f793936cc8518ffdcd5ef29799b9fe24247c2e5bc17a3`
- Probe-log SHA-256: `c9152adbac228e5ab15817f0317fe7551f4a5ed01d480c1819dc13f34ffe4882`
