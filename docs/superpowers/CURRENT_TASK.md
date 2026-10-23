# Current task: full training-path audit and measured fixes

Updated 2026-10-24 01:45 Asia/Ho_Chi_Minh.

## Resume here — Task 7 IN PROGRESS

New user instruction after Task 6: audit all owned scenario/algorithm/runtime/cloud code and fresh online W&B metrics/logs, explain why 20,000 episodes × <=1,800 steps cannot finish within 12h, and implement justified fixes. This authorizes investigation and minimal verified fixes; proceed sequentially, preserving healthy production training and scientific settings. Task 6 remains completed/reverted at commit6062744.

- Worktree `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`, branch `feat/cpu-replay-20261022`, starting HEAD `6062744b6626fcc73f55fab8b7f321da74a5f1d9`.
- Read-only audits split into scenario, algorithms/replay/runtime, native network, cloud/lifecycle; root owns GPU packet path and integration. No agents may edit or touch production jobs.
- Fresh W&B 01:39 Oct24: BASE MASAC650/6,983,680 and MATD3 0/3,825,664 finished `time_budget`; both Kaggle parents COMPLETE. PQL1M child3939/7,331,840 running; MATPQL0/3,465,216 has time_budget recorded. Full20k completion conditions unchanged.
- Downloaded actual root continuation control files under `/tmp/uav-task7-terminal/account_32` and `account_33`. Both children are **RUNNING** on their same accounts: `jwekfjjwoigjo/uavnet-train-20261023-073957-c01`, `lihuhi/uavnet-train-20261023-074258-316e25-c01`; both pin source337608c2, verified prefixes650/6,983,680 and0/3,825,664. No new child W&B run yet at snapshot: preflight/restore is plausible, do not duplicate or restart them.
- Actual runtime budget: max_train_seconds36000 (10h), submitted ceiling39600 (11h), wrap-up reserve1020s for artifact_timeout900; a nominal12h allowance is not12h of training.
- Native/GPU IDLE/LISTEN semantics reviewed. LISTEN already coalesces up to nearest other event; broad skipping changes insertion ties, diagnostics or link snapshots. Candidate hypothesis: retain all logical events and reuse already-valid pair gain inside frozen-geometry packet_step instead of repeated displacement/sqrt checks. Scoped candidate now implemented after RED test (missing busy_in_step). Focused propagation/native packet/physics/retry/admission suite:79 passed,1 skipped,14 subtests in23.62s. Generic busy unchanged. T4 benefit gate pending; unpromoted.
- Proven ancillary issue for explicitly budgeted jobs: CPU-visualization builder accepts time_budget/unfinished episodes outside auto_continue; legacy partial previews without a declared target are intentionally supported by existing tests, so keep that path; compile/fused-only launches skip acceleration preflight. These do not explain primary collector slowdown. Fix one at a time with a failing test, not bundled speculative changes.
- Next: finish minimal design/test/benchmark plan, write performance gate before candidate code, smoke CPU, benchmark pinned baseline/candidate on T4 account35 after quota check. Keep candidate only if measured gain exceeds noise and all initialized state hashes/native oracle match. Then address concrete lifecycle gates with test-first changes, full suite, review, commit/push and update audit coverage/evidence. Do not promise completion in12h without measured full-training throughput.

### Task7 ledger

- Task7A: RED missing scoped Warp CCA helper; GREEN79passed/1skipped, CPUfixture2worlds dt.01 errors0. Candidate5fcb272 unpromoted; T4same-account35 quota26.7h available, kernel bjkjbhm/uav-frozen-gain-t4-ab-20261024 v1 launched. B,C,C,B,B,C, full140arrays,39CUDAtests; wait for actual result.
- Task7B RED:6 unfinished/inconsistent-budget cases accepted beforefix; GREEN:37passed,35subtests in8.18s, exit0 (HF_HUB_OFFLINE=1 WANDB_MODE=offline). Legacy no-budget preview still passes.
- Task7B Ruling: guard explicitly budgeted final handoffs, retain existing no-budget legacy partial previews — existing CPUHandoffTests deliberately require partial visualization; cost if wrong is legacy no-budget output still being mistaken for full training, so retain training_status labels and document scope. No production job changed.

## Previous Task 6 record

**Task 6: receiver buffer cache — COMPLETE, NEGATIVE RESULT, REVERTED.**
Worktree: `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`.
Branch: `feat/cpu-replay-20261022`. Candidate: `cb7c5d6c0c0ba50640c75bea83238f422ea4d902`; baseline: `4cd9358fe0d7cc851bc95ed1c6d5bd231666d310`.
Candidate is in history, but its two-file change has been reverted. Both files match 4cd9358 byte-for-byte. Do not recreate the cache. Primary peer-1MB gate: baseline median 39.17247858, candidate 39.16647574 transitions/s (-0.015324%, within variability); all 140 initialized arrays match across all six trials. No measured benefit, so reject this candidate. Reviewer confirmed the conclusion and byte equality.

All Task 6 gates are complete:

- Full suite after revert: **673 passed, 74 skipped, 8 warnings, 1222 subtests passed in 254.25s**, exit 0. `/tmp/uav-task6-after-revert-full-20261024.log` and `.exit`.
- T4 peer-1MB version 1: `bjkjbhm/uav-receiver-cache-peer1m-20261024`; all 140 arrays/trial include MAC RNG, pending events/packets and energy; errors=0.
- T4 control version 2: `bjkjbhm/uav-receiver-cache-t4-comparison-20261024`; active1KB peer medians 787.74247218 / 787.55057233 (-0.02436%); active1MB GCS 38.49941272 / 38.48308302 (-0.04242%); 24 array hashes/trial match, errors=0. The GCS control alone is not a peer-benefit test. V1 failed before simulation on invalid device config; only V2 was accepted.
- Existing native-vs-Warp packet oracle on candidate/T4 `cuda:0`: **38 passed in 12.333s**, exit 0. CPU suite and CUDA oracle scopes differ; do not describe local CUDA skips as having run.
- Evidence: `evidence/2026-10-24-task6-final-verification.json`, `evidence/2026-10-24-t4-receiver-cache-peer1m.json`, `evidence/2026-10-24-t4-receiver-cache-control.json`, `evidence/2026-10-24-t4-receiver-cache-cuda-oracle.log`.

**Next action:** commit/push the final evidence/revert on this branch; the resulting HEAD is the final task commit (do not amend the record just to embed its own SHA). Then stop Task 6. Wait for the user's next task instruction before implementing another candidate. The leading next hypothesis is exact MAC IDLE/LISTEN polling; it requires profiling and native parity, not relaxed timing. Learner AMP/compile/fused is a separate controlled candidate. Preserve all healthy production jobs.

## Fresh verification and root-cause evidence

- Full suite at candidate cb7c5d6: **674 passed, 74 skipped, 8 warnings, 1221 subtests passed in 243.11s**, exit 0. Log: `/tmp/uav-task6-full-20261024.log`.
- Task 4 diagnostic is now COMPLETE: `bjkjbhm/uav-component-isolation-20261023-160905`, baseline 4cd9358, 4096 worlds, two T4 visible. Saved in `evidence/2026-10-24-t4-component-isolation.json`.
- Matched T4 fixtures: active 1KB/dynamic 9726.67 transitions/s; active 1MB/dynamic 38.74995; active 1MB/reject 38.79999. Thus 1MB is ~251x slower in this diagnostic, while changing motion changes active-1MB throughput ~0.13%. Network-disabled dynamic is 15843.01 transitions/s. These are simulator diagnostics, not full-training rates.
- Additional completed instrumented probe: `bjkjbhm/uav-task6-dispatch-probe-20261023-160547`, diagnostic commit cbc7253, wider sensor FOV, 4096 worlds, 2 warmup + 1 timed step. Cumulative dispatches across all 3 steps: 4,735,399,723; IDLE + LISTEN account for 97.814%. The timed third step is 118.12 s. This identifies microsecond MAC polling as a profiling target; dispatch-count share is not runtime share, and these counts are not a production-run measurement. See `evidence/2026-10-24-t4-dispatch-probe.json`.
- GPU0 runs all 4096 environments, GPU1 runs learner/replay. Cumulative production timing supports a major collector bottleneck. **Phase-specific exception:** between 00:30 and 00:43, MASAC BASE 1M learner time grew 741.14 s while collector wait grew only 0.082 s over 750.18 s; it was learner-limited in that window. MATD3 BASE collector wait grew 723.76 s/765.18 s; MATD3 PQL collector grew 737.87 s/741.29 s. AMP/compile/fused Adam may help learner-limited phases, but do not remove packet-event cost.
- Corrected cache-cost interpretation: installed Warp 1.18 BoolOp code generation short-circuits `and`/`or`. In production 1MB/3MB, each source holds at most three reports, so the old `bytes_used` call is guarded by report-present/nonexpired/unknown checks; a naive 35x35 target-loop count overstates actual production scans. Measure the real gain.
- Fresh acceleration-config audit: all four canonical 1M W&B runtime configs explicitly show amp_mode=off, compile_models=false, fused_optimizers=false. Code supports them; do not claim they are enabled. Future learner acceleration needs a separate controlled numerical/performance gate; preserve active runs and BASE/PQL fairness.
- Tasks 1–3 were completed in the prior audit; Task 5 queue-count candidate was reverted after no measured speedup. Framefree remains unpromoted because its T4 benchmark lost to Gen1. This is not a claim that every module's audit is complete.

## Production cohort: read-only monitoring

Latest snapshot: **01:22 ngày 24/10/2026 UTC+7**, `evidence/2026-10-24-performance-latest-cohort.json`. Detailed diagnosis/ETA: `evidence/2026-10-24-runtime-root-cause-report.md`. Earlier timing-phase snapshot preserved in `evidence/2026-10-24-performance-cohort-0043.json`.

| Case | Canonical root W&B | Current child | Latest completed episodes / global step |
| --- | --- | --- | --- |
| MASAC BASE 1M | 63faeffb | not observed yet | 650 / 6,983,680; time_budget, W&B finished, Kaggle wrapping up |
| MASAC PQL 1M | fc14508c | bc18a145 c01 running | 3815 / 7,294,976 |
| MATD3 BASE 1M | 2dcbf36c | not observed yet | 0 / 3,825,664; time_budget, W&B/Kaggle wrapping up |
| MATD3 PQL 1M | 278634f6 | none yet | 0 / 3,379,200; running |
| MASAC BASE 500k | b9d73bd3 | 30e43dda c01 running | 4096 / 11,083,776 |

Root Kaggle refs: MASAC BASE `jwekfjjwoigjo/uavnet-train-20261023-073957-9c4cd8` (account_32); MASAC PQL `ewkljfelweow/uavnet-train-20261023-051420-773792` (account_21); MATD3 BASE `lihuhi/uavnet-train-20261023-074258-316e25` (account_33); MATD3 PQL `kjhiguyf/uavnet-train-20261023-074845-a0387e` (account_34). The 500k root/child use account_28, `ekjfewflwejl/uavnet-train-20261023-051422-cbe381` / `uavnet-train-20261023-051422-c01`.

The two existing children have meaningful progress beyond verified parent prefixes; preserve them. PQL parent ended 1066 / 7,045,120; 500k parent ended 4096 / 10,309,632. Old roots/children inherit infrastructure commit 66f4d40. A finished W&B parent may still have a running Kaggle notebook saving/uploading a full checkpoint and scheduling its child. Recheck actual owned state and immutable manifest before recovery; do not duplicate a healthy wrap-up or child.

## Recovery watch restored at 01:10 Asia/Ho_Chi_Minh

Automation `6adb1a0b96dc8191b26ad5ed86545aac` (UAV 1M Cohort Recovery) was disabled by the prior assistant at 21:22 Oct 23 after Sandbox Resource not found. Current Sandbox, W&B/Kaggle and GitHub reads succeeded. Re-enabled the existing hourly watch with recovery controller/source pinned to tested 4cd9358 in an isolated checkout, preserving current healthy children and excluding experimental performance commits. The update response confirms is_enabled=true; next_run_time was null and no new scheduled execution has been verified. Do not describe it as successfully executing yet. Evidence: `evidence/2026-10-24-recovery-watch-restored.json`.

## Operating constraints

- Keep healthy production jobs running. Apply performance changes only to controlled diagnostics/future launches.
- Scientific invariants: seed/RNG, reward, motion, PHY, CSMA/backoff/collision/retry/ACK order and timestamps, report custody/TTL/admission, energy, observations/state, exact continuation. Keep 1MB reports, 35 targets, 25 obstacles, 1800-step horizon and 20k episode target.
- Training completion requires episodes_completed >= 20000, episode_budget_complete=true, stop_reason=episode_budget. A time_budget segment is not final evaluation; zero completed episodes during a synchronized first wave is not a stuck counter.
- Continuation stays on the same account/lineage. Verify immutable HF full-resume/checkpoint/prefix before recovery; inspect actual owned children before submitting. A private-kernel permission denial is ambiguous until same-account owned-list lookup. Preserve an old-code child with meaningful progress; use b32d18e-or-newer controller only for concrete failed/absent/unsafe recovery.
- Credentials: trusted `load_credentials` helper, `/home/aduy/Documents/NCKH/uav_network/key.txt`; never print secrets. CUDA benchmarks use Kaggle because this host has no CUDA.
- Earlier handoff/replay design and completed work remain in `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and Git history of this record. For operational recovery, read the applicable continuation code/tests and newest live evidence before acting.
