# Current task: exact-network performance audit

Updated 2026-10-24 00:43 Asia/Ho_Chi_Minh (2026-10-23T17:43Z).

## Resume here

**Task 6: receiver buffer cache — IN PROGRESS, awaiting the T4 A/B gate.**
Worktree: `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`.
Branch: `feat/cpu-replay-20261022`. Candidate commit: `cb7c5d6c0c0ba50640c75bea83238f422ea4d902`; baseline: `4cd9358fe0d7cc851bc95ed1c6d5bd231666d310`.
The cache is already committed/pushed. Inspect the source before attempting to recreate it.

1. Check `bjkjbhm/uav-receiver-cache-t4-comparison-20261024`, **version 2**, account_35. Version 1 stopped before simulation because the diagnostic used invalid `runtime.device=cuda:0`; version 2 uses `cuda` and was preflight validated. An output/log download while a version runs can still return the previous version's output: verify version and artifact before interpreting a failure.
2. On completion, download `receiver-cache-ab.json`, `receiver-cache-cuda-oracle.log`, and the kernel log. The job alternates baseline/candidate in B,C,C,B,B,C order, with 4096 worlds, seed44, one warmup step excluded, three repeats. Cases: active 1KB peer selection (32 timed steps, original active state restored on device before every step, restore outside timing); active 1MB GCS streaming (one timed step after warmup). Compare medians, spread, world errors, and the 24 array hashes. It then runs the existing native-vs-Warp packet oracle on CUDA.
3. Keep the cache only with measured benefit and exact-state/oracle agreement. If no reproducible gain, remove the candidate and its source-inspection-only regression, then rerun relevant tests/full suite. Preserve all scientific settings.
4. Record the decision and evidence; commit/push this record and report Task 6 COMPLETE only when its gates finish. Work on the next candidate only after this decision and the user's next instruction.

## Fresh verification and root-cause evidence

- Full suite at candidate cb7c5d6: **674 passed, 74 skipped, 8 warnings, 1221 subtests passed in 243.11s**, exit 0. Log: `/tmp/uav-task6-full-20261024.log`.
- Task 4 diagnostic is now COMPLETE: `bjkjbhm/uav-component-isolation-20261023-160905`, baseline 4cd9358, 4096 worlds, two T4 visible. Saved in `evidence/2026-10-24-t4-component-isolation.json`.
- Matched T4 fixtures: active 1KB/dynamic 9726.67 transitions/s; active 1MB/dynamic 38.74995; active 1MB/reject 38.79999. Thus 1MB is ~251x slower in this diagnostic, while changing motion changes active-1MB throughput ~0.13%. Network-disabled dynamic is 15843.01 transitions/s. These are simulator diagnostics, not full-training rates.
- Additional completed instrumented probe: `bjkjbhm/uav-task6-dispatch-probe-20261023-160547`, diagnostic commit cbc7253, wider sensor FOV, 4096 worlds, 2 warmup + 1 timed step. Cumulative dispatches across all 3 steps: 4,735,399,723; IDLE + LISTEN account for 97.814%. The timed third step is 118.12 s. This supports microsecond MAC polling as the dominant implementation cost; these counts are not a production-run measurement. See `evidence/2026-10-24-t4-dispatch-probe.json`.
- GPU0 runs all 4096 environments, GPU1 runs learner/replay. Current production timing supports the collector bottleneck. AMP/compile/fused Adam optimize the learner and cannot remove the packet-event cost.
- Tasks 1–3 were completed in the prior audit; Task 5 queue-count candidate was reverted after no measured speedup. Framefree remains unpromoted because its T4 benchmark lost to Gen1. This is not a claim that every module's audit is complete.

## Production cohort: read-only monitoring

Live observations and ETA assumptions: `evidence/2026-10-24-performance-live-audit.json` and newer `evidence/2026-10-24-performance-latest-cohort.json`.

| Case | Canonical root W&B | Current child if any | Root Kaggle |
| --- | --- | --- | --- |
| MASAC BASE 1M | 63faeffb | none at 00:43 | jwekfjjwoigjo/uavnet-train-20261023-073957-9c4cd8 |
| MASAC PQL 1M | fc14508c | bc18a145, c01 running | ewkljfelweow/uavnet-train-20261023-051420-773792 |
| MATD3 BASE 1M | 2dcbf36c | none at 00:43 | lihuhi/uavnet-train-20261023-074258-316e25 |
| MATD3 PQL 1M | 278634f6 | none at 00:43 | kjhiguyf/uavnet-train-20261023-074845-a0387e |
| MASAC BASE 500k | b9d73bd3 | 30e43dda, c01 running | ekjfewflwejl/uavnet-train-20261023-051422-cbe381 |

The two children have meaningful verified progress beyond parent checkpoints; preserve them. fc14508c ended time_budget at 1066 episodes / 7,045,120 transitions; its child reached 3059 / 7,196,672 at 00:43. b9d73bd3 ended time_budget at 4096 / 10,309,632; its child reached 4096 / 10,797,056. Old roots/children inherit infrastructure commit 66f4d40; staged-resume cleanup recovery rules below still apply if a concrete failure appears.

## Operating constraints

- Keep healthy production jobs running. Apply performance changes only to controlled diagnostics/future launches.
- Scientific invariants: seed/RNG, reward, motion, PHY, CSMA/backoff/collision/retry/ACK order and timestamps, report custody/TTL/admission, energy, observations/state, exact continuation. Keep 1MB reports, 35 targets, 25 obstacles, 1800-step horizon and 20k episode target.
- Training completion requires episodes_completed >= 20000, episode_budget_complete=true, stop_reason=episode_budget. A time_budget segment is not final evaluation; zero completed episodes during a synchronized first wave is not a stuck counter.
- Continuation stays on the same account/lineage. Verify immutable HF full-resume/checkpoint/prefix before recovery; inspect actual owned children before submitting. A private-kernel permission denial is ambiguous until same-account owned-list lookup. Preserve an old-code child with meaningful progress; use b32d18e-or-newer controller only for concrete failed/absent/unsafe recovery.
- Credentials: trusted `load_credentials` helper, `/home/aduy/Documents/NCKH/uav_network/key.txt`; never print secrets. CUDA benchmarks use Kaggle because this host has no CUDA.
- Earlier handoff/replay design and completed work remain in `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and Git history of this record. For operational recovery, read the applicable continuation code/tests and newest live evidence before acting.
