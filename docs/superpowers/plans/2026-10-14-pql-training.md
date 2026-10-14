# PQL-inspired training implementation plan
> For agentic workers: use superpowers:executing-plans to implement and check each task in this session.

**Goal:** Implement approved Option A without changing MASAC/MATD3 equations.
**Architecture:** Split learner APIs, versioned snapshots, bounded collector/critic/policy workers, and feedback rate budgets. Preserve the fixed runner and add opt-in PQL profiles.
**Tech stack:** Existing Python, Torch, Warp, YAML, unittest; stdlib threads/queues.
**Spec:** docs/superpowers/specs/2026-10-14-pql-training-design.md

## Global constraints
4096 envs; horizon1; batch target8192; replay500000; warmup32; critic/collection8; MASAC policy every critic; MATD3 policy_delay2. No reward/motion/network changes. Owned text files <=500 lines.

## Review focus
- Snapshot publication and CUDA stream lifetime must not race optimizers.
- A failed learner must wake collectors and checkpoint drains.
- Terminal observations must be frozen before reset/buffer reuse.
- Warmup/resume must not mint retrospective update credits or reset Adam silently.
- Async scheduling is not bitwise reproducible; synchronous mode is the regression contract.

### Task 1: Split update APIs
Modify algorithms/masac.py and matd3.py. Produce update_critic(batch, collect_metrics), update_actor(batch, collect_metrics) and update_targets().
- [x] Freeze afeaa3d update functions in tests/legacy_updates.py and write split/oracle tests.
- [x] Observe missing-API failures.
- [x] Extract stages; keep update() as the equivalent combined wrapper.
- [x] Compare parameters, targets, losses, optimizer states and RNG; run algorithm tests.

### Task 2: Worker system and persistence
Create experiments/pql_replay.py, pql_workers.py, pql_system.py; add PQLConfig.
Interfaces: PQLSystem(config), submit(transition), flush(), metrics(), act_batch(observation, **options), save(path, progress), close().
- [x] Write tests for ratio/warmup, ring wrap, snapshot isolation, worker errors and resume.
- [x] Observe missing-feature failures; implement bounded work queues with exact ratio credits, immutable snapshots and owned replay.
- [x] Preserve target clocks and serialize optimizers/controller/RNG/replay at drained boundaries.
- [x] Run component tests and legacy regressions.

### Task 3: Runner/config/benchmark
Create experiments/pql_training.py, configs/motion-pql.yaml, scripts/benchmark_pql.py; route opt-in training and support indexed Warp devices.
- [x] Write CPU Warp integration tests and missing-CUDA device tests.
- [x] Run failures; implement terminal-safe collection, logging, progress and checkpoint boundary coalescing.
- [x] Smoke both algorithms and deterministic debug mode; run complete suite.
- [x] Benchmark exact T4 scenario: both algorithms passed E4096/B8192/replay500k on 2×T4; see docs/pql-validation-20261014.md.

### Task 4: Independent review and delivery
- [x] Review the complete diff for races, math changes, resume and limits.
- [x] Reproduce/fix material findings with tests; git diff --check; complete suite.
- [x] Commit reviewed code and publish branch pql-async-20261014; local278 tests and T4 smoke/benchmark results recorded.

## Review ledger
- Final: fixed cloud resume artifact mismatch — test_pql_artifacts tests RED→GREEN.
- Final: fixed inconsistent progress/unearned credit acceptance — test_resume_rejects_unearned_credit_inconsistent_history_and_nonfinite_weights RED→GREEN.
- Final: fixed zero-update benchmark pass and qualified CUDA/T4 verification — test_benchmark_cannot_pass_without_both_learner_updates RED→GREEN.
- No Critical findings, no deferred minors or rulings. Full CPU/Warp suite: 276 tests OK, 45 CUDA skips. Exact algorithm oracle and all material review reproductions pass. GPU measurements remain pending.

- T4 smoke job f25892f: CUDA tests10/10 and PQL tests7/7 passed on 2×T4; first large benchmark stopped before allocation at reset_peak_memory_stats. Added explicit cuda.init and a fresh-process CUDA regression. Local suite278 tests OK,46 CUDA skips. Re-measurement pending.

- T4 rerun e680276: both cases PASSED; MASAC41.36s/3961 transitions/s, MATD317.86s/9172 transitions/s; no production convergence claim.
