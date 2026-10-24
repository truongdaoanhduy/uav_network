# Current task: full training-path audit and measured fixes

Updated 2026-10-24 10:55 Asia/Ho_Chi_Minh.

## Cloud independence implementation checkpoint

Plan: plans/2026-10-24-sandbox-independent-flow.md. Added private CPU controller generation/state/heartbeat, terminal-only continuation/quota decisions, adoption/no-overwrite, independent session handoff. CLI auto_continue seeds it and retains GPU receipt if seed fails. CPU visualization timeout39600 and resume completed evaluation episode/seed records; publication-only recovery reads saved Kaggle assets. Keep all30configured evaluation episodes; require W&B and immutableHF delivery before pipeline complete. Regression RED confirmed on pre-fix 44fdd43; focused controller/continuation56passed38subtests; final fullsuite707passed74skipped1250subtests; fix source pushed at7222fd47. Real cloud smoke is currently blocked pre-push by Kaggle private-dataset download timeouts reproduced on spare accounts35 and27; no phantom child/kernel was created and all five production continuations remain running. Re-run real E2E when that service path responds, then deploy controllers. Do not yet claim Sandbox can close.

Fresh10:47 W&B: recoveredPQLaa859075 now actually trains global7,053,312 above restored7,045,120; recovered500k01763049 at14,049,280 above13,914,112. Three healthy1M children preserved and progressing: MASACBASE9,269,248, MATBASE9,527,296, MATPQL9,850,880. No new full-budget completion claim.

## Latest user priority: Sandbox-independent full flow — ACTIVE

User repeated that once W&B shows Running, Sandbox/their computer will be turned off; training continuation and final CPU visualization must still happen. Implement a CPU-only Kaggle controller per lineage; do not rely on the local ChatGPT/Sandbox watcher. Design: specs/2026-10-24-sandbox-independent-flow-design.md. Reuse existing GPU self-chain, act only after terminal parent, wait same-account quota, adopt actual children, chain bounded CPU controller sessions, and verify final CPU visualization. Preserve all five current GPU jobs. New design builder check, test-first implementation, full suite, real labelled end-to-end cloud smoke, and deployment to five lineages required before claiming unattended readiness.

Previous Task7 source at0c80b7d: CPUfullsuite680passed74skip1235subtests; allcorefixescommitted. Acceleration smoke v2 bjkjbhm/uav-task7-acceleration-smoke-20261024 pinned0c80b7d COMPLETE; compile-only and fused-only preflights passed, realCUDAprobe1passed14warnings22.66s; v1compile-only/fused-only passed, staleCUDAcritic-dtypeassertion corrected to existing AMP contract and rerun. Netcache5fcb272 rejected+1.30% below2% and reverted. Kernel-blockgeometry hypothesis (4096worlds/default256threads gives16blocks on40SM T4) is unimplemented backlog; current user priority is cloud independence.

Current production: healthy1MBASE MASACa8ac9ffc/MATD3 7a57e2a5/PQLMATD3bdb9a503 preserved; same-account21PQLc02 aa859075 and account28BASE500kc02 01763049 submitted on source4cd9358, timeout39600, rootprefix1066/7045120 and4745/13914112. Their bootstrap is RUNNING; fresh training progress still needs verification. Recovery evidence: evidence/2026-10-24-task7-recovery-submissions.json.

All99ownedPython modules,8scripts,20YAMLconfigs audited by root/read-only domain agents. Final Task7 report/verification consolidation remains; do not redo audits or restart production. Watcher6adb1a0b96dc8191b26ad5ed86545aac was paused02:19 for missingSandbox, resumed10:03 with latestlineages; only configuration enabled verified, no new scheduledrun. New cloud controllers replace this dependency.

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

### New steering and live recovery at09:30

- Healthy1M children: MASACBASEa8ac9ffc4096/8,630,272; MATBASE7a57e2a54096/8,818,688; MATPQLbdb9a5034096/9,162,752. Preserve all three.
- MASACPQLc01bc18a145 failed4096/8,908,800 during final full-resume torch.save: basic_ios iostream error / unexpectedpos3030776896. Confirmed save I/O failure; disk pressure strongly supported by last5.11GBfree versus parent full-resume15.52GB and old66f4d40 retaining the staged input. No current CUDA-OOM evidence in that log. Top-level continuationValueError is secondary, job-result empty.
- Rootfc14508c:v3 descriptor downloaded: immutableHF4672be4ad3ea3d13a419e7593960d3fb01955420, resume.pt15,522,939,889B SHA8d26e1e9671157180c4b048c0b95a515bded6ead06526d6c6c8a468b64fecbed, prefix1066/7,045,120. Need inspect child artifact descriptors/Kaggle output for any newer valid full state before rollback; never label policy-only4096 as exact resume. Failedc01 must remain preserved; recovery uses unusedc02 sameaccount21 and tested4cd9358 source/controller, explicitly override source because helper preserves parentcommit.
- 500kchild30e43dda finishedtime_budget4745/13,914,112. Plannedc02failedonlyinsufficientquota0.41h; sameaccount28 waiting, artifactv3. Do not switch accounts. Verify immutablefullmanifest and ownedchildstate before future quota-resume.
- /tmp from overnight session was cleared; committed/home worktree persisted at346632c plus D WIP. Recover records from git/evidence, do not rerun completed remote training or completed T4 AB.

### Task7 ledger

- Task7A: RED missing scoped Warp CCA helper; GREEN79passed/1skipped, CPUfixture2worlds dt.01 errors0. Candidate5fcb272 unpromoted; T4same-account35 quota26.7h available, kernel bjkjbhm/uav-frozen-gain-t4-ab-20261024 v1 launched. B,C,C,B,B,C, full140arrays,39CUDAtests; wait for actual result.
- Task7B RED:6 unfinished/inconsistent-budget cases accepted beforefix; GREEN:37passed,35subtests in8.18s, exit0 (HF_HUB_OFFLINE=1 WANDB_MODE=offline). Legacy no-budget preview still passes.
- Task7B Ruling: guard explicitly budgeted final handoffs, retain existing no-budget legacy partial previews — existing CPUHandoffTests deliberately require partial visualization; cost if wrong is legacy no-budget output still being mistaken for full training, so retain training_status labels and document scope. No production job changed.

- Task7C RED: compile-only/fused-only skipped preflight; BASE preflight allocation ignored hybrid75 (actual allocation-expression CPU reproduction). GREEN:50passed,2skipped,12subtests in7.54s, exit0. Bootstrap now gates any requested acceleration; BASE uses production make_replay and reports cold/hot bytes. No flags enabled by this fix.
- Task7C Ruling: align preflight replay with production while fixing independent-flag gating — otherwise expanded preflight could falsely OOM on all-GPU replay; cost if wrong is a misleading diagnostic, so bounded real two-T4 smoke remains required, not a claim of canonical accelerated1M success.

- Task7A FINAL: T4v1 COMPLETE, baseline39.11293086 vs candidate39.62140881t/s, **+1.300025% reproducible**, all140arrays match6trials,39CUDAoracle tests passed37.222s. Below predeclared2% gate; REVERTED allthreecandidatefiles byte-equal6062744. Do not call it noise/no gain; it is too small for the acceptance gate. No further micro-cache candidate this task; event-loop architecture needs separate measured redesign.
- Task7D RED actor-grad failure reproduced from actual BASEprobe context on CPU BF16; fixed cache_enabled=False. PQLprobe actual report expression failed on missing HybridMeasuredReplay.data; storage formatter now reports cold/hot tiers and preserves plain keys. GREEN65passed1skipped in8.44s, exit0. Real CUDA probe/preflight smoke pending; fullsuite after netrevert pending.

- Final review Important topology finding FIXED: expanded acceleration guard must support non-dual-T4 CUDA. RED one-board P100 facade failed oldguard; GREEN34passed16subtests9.97s after generic usable-CUDA guard, BASE/PQL role resolution and device_count loops. Load real test consumers before patching constructors to avoid import-time mock capture. No re-review; fresh fullsuite and2T4smoke next.
- Fullsuite before topologyfix:679passed74skipped8warnings1234subtests498.67s, exit0. Postfix suite pending.
- Recovery submissions actual at09:50: PQL1Maccount21c02 aa859075 and BASE500kaccount28c02 01763049, sameaccounts, source/controller4cd9358, both timeout39600 after freshquota30h. Prefixes1066/7,045,120 and4745/13,914,112; immutableHFsize/SHA verified. See evidence/2026-10-24-task7-recovery-submissions.json. Submitted is not yet verified training progress; checkKaggle/W&B.

- Post-review fullsuite:680passed74skipped8warnings1235subtests486.84s, exit0. Actual T4smokev1: compile-onlyhybrid and fused-onlyhybrid both PASSED (6collections,4critic/4actorupdates;3compiledgraphs/54kernels andactualfusedAdam). CUDAprobe applied critic/actor/alpha updates successfully, then a stale test assertion incorrectly required FP16criticoutputs; existing production evidence gate/tests explicitly allow FP32critic reductions with FP16actors/completedcriticcalls. Corrected that CUDA-only assertion to the existing contract; no productionprecision change. Preservev1error/evidence; rerunv2 pinnedsource beforecompletion. CPUfocused40passed1skip4subtests6.27s.

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

## 2026-10-24 user-approved report-size decision

- HARP-LAWN Table II uses UavNetSim-v1 with a 1024-byte data-packet payload; it does not define a separate 1 MB report.
- User explicitly approved switching the primary future training workload to 1 KiB after this verification.
- New primary config: `configs/comparison-20k-hybrid75-1kb.yaml` with report=1024 B, payload=1024 B, buffer=3072 B. Preserve env4096, 20k episodes, 1800 steps, replay1M, batch8192, CPU replay75, warp_packet.
- Keep existing 1 MB runs/config immutable as historical/stress-test evidence. Do not continue a 1 MB lineage into a 1 KiB lineage; new 1 KiB training must start a fresh lineage from episode 0.
- Existing T4 isolation evidence: active1KB 9726.67 transitions/s vs active1MB 38.75 transitions/s (~251x slower for 1 MB under forced active traffic). This is a simulator diagnostic, not a full-training ratio.
- Evidence: `docs/superpowers/evidence/2026-10-24-harp-lawn-1kb-primary-profile.md`.


## 2026-10-24 1 KiB learner acceleration benchmark — COMPLETE

- Spare-account production-shape MASAC benchmark completed on Kaggle T4 x2: `bjkjbhm/uav-1kb-learner-accel-20261024` v7. Runner commit `89440b0`; production code under test is parent `fbbb556`.
- Exact benchmark shape: report/payload 1024 B, env4096, batch8192, replay1M, CPU75/GPU25 hybrid replay filled to 1,000,000 rows. Timed `sample + update`, 20 updates, two interleaved repetitions/case.
- Baseline median 566.02 ms/update. AMP-only 534.07 ms (1.060x), compile-only 534.69 ms (1.059x), fused-only 577.57 ms (0.980x), all-three 353.78 ms (**1.600x**, 37.50% learner-time reduction).
- Combined cases had AMP critic/actor/alpha 23/23 applied, 0 skipped; 3 compiled graphs / 88 Inductor kernels; all Adam groups fused; parameters finite. Full-combination GPU1 peak allocated ~5.55 GB vs baseline ~6.62 GB.
- Ruling: combined AMP+compile+fused passes the throughput/execution gate; fused-only does not. Do **not** mutate current non-AMP 1 KiB production lineages. Acceleration may be used only for a fresh matched cohort/future experiment from episode 0.
- Evidence: `evidence/2026-10-24-1kb-learner-acceleration-result.json` and `.md`.
- Next: fresh-check the four existing 1 KiB production lineages. Preserve them if healthy; only start an accelerated matched cohort if the user later chooses to replace/compare them rather than silently changing an in-flight experiment.


## 2026-10-24 matched acceleration across BASE/PQL — COMPLETE

- Kaggle T4x2 matched benchmark v8 completed from runner `b42a9a7`, no production mutation.
- MATD3 BASE: 158.82 -> 139.29 ms/update, **1.140x**.
- MASAC PQL: 2407.09 -> 2453.36 transitions/s, **1.019x** with meaningful run-to-run variation; not a robust promotion.
- MATD3 PQL: 3509.38 -> 3272.39 transitions/s, **0.932x**; combined AMP+compile+fused is slower.
- Ruling: do not globally enable the three acceleration flags. Keep current four 1 KiB production lineages unchanged. MASAC BASE's prior 1.600x result remains path-specific.
- Evidence: `evidence/2026-10-24-1kb-matched-acceleration-result.json` and `.md`.
- Next task: PQL-specific one-variable isolation (AMP-only, compile-only, fused-only; then only justified combinations) on spare T4x2, preserving production.


## 2026-10-24 PQL one-variable acceleration isolation — COMPLETE

- Kaggle T4x2 diagnostic `bjkjbhm/uav-1kb-learner-accel-20261024` completed from runner `1b2ddc9`; no production mutation.
- MASAC PQL: AMP-only **1.074x**, compile-only **0.884x**, fused-only **1.011x** vs baseline.
- MATD3 PQL: AMP-only **1.018x**, compile-only **0.919x**, fused-only **0.997x** vs baseline.
- Ruling: reject compile for PQL; do not promote fused alone; AMP is the only single variable that improved both paths.
- Evidence: `evidence/2026-10-24-1kb-pql-acceleration-isolation-result.json` and `.md`.
- Next controlled task: compare **AMP+fused (compile off)** against AMP-only and baseline on spare T4x2. Preserve live production lineages.

## Operating constraints

- Keep healthy production jobs running. Apply performance changes only to controlled diagnostics/future launches.
- Scientific invariants for the new primary 1 KiB line: seed/RNG, reward, motion, PHY, CSMA/backoff/collision/retry/ACK order and timestamps, report custody/TTL/admission, energy, observations/state, exact continuation, 35 targets, 25 obstacles, 1800-step horizon and 20k episode target. Report/payload are 1024 B and buffer is 3072 B. Existing 1 MB lineages remain immutable stress-test/history and must not be mixed with the new 1 KiB lineage.
- Training completion requires episodes_completed >= 20000, episode_budget_complete=true, stop_reason=episode_budget. A time_budget segment is not final evaluation; zero completed episodes during a synchronized first wave is not a stuck counter.
- Continuation stays on the same account/lineage. Verify immutable HF full-resume/checkpoint/prefix before recovery; inspect actual owned children before submitting. A private-kernel permission denial is ambiguous until same-account owned-list lookup. Preserve an old-code child with meaningful progress; use b32d18e-or-newer controller only for concrete failed/absent/unsafe recovery.
- Credentials: trusted `load_credentials` helper, `/home/aduy/Documents/NCKH/uav_network/key.txt`; never print secrets. CUDA benchmarks use Kaggle because this host has no CUDA.
- Earlier handoff/replay design and completed work remain in `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and Git history of this record. For operational recovery, read the applicable continuation code/tests and newest live evidence before acting.

## 2026-10-24 1 KiB cohort launched and verified

- Primary 1 KiB cohort is now live from episode 0 on source `0c9fbc5b93acfd3008f9802950570849bc42997c`:
  - MASAC BASE account_01: Kaggle `duytrngoanh/uavnet-train-20261024-073119-a53a63`, W&B `7f985331`.
  - MASAC PQL account_03: Kaggle `vuliu123456/uavnet-train-20261024-074226-5b2381`, W&B `e0bfbcaf`.
  - MATD3 BASE account_06: Kaggle `libiu123/uavnet-train-20261024-074821-e57c46`, W&B `767cb8da`.
  - MATD3 PQL account_09: Kaggle `nhtbrg/uavnet-train-20261024-080451-eda89d`, W&B `30a40bf7`.
- All four GPU kernels and their four CPU cloud controllers were freshly health-checked running with `unattended_ready=true`.
- One accidental newer MATD3 BASE duplicate caused by an ambiguous 502 submit response (`a379f8bd` + controller `libiu123/uav-flow-10ea62cf6a6c06-0000`) was deleted after confirming the older `767cb8da` had more progress. The retained lineage stayed running.
- 62 s live W&B rates: MASAC BASE 792.45, MASAC PQL 1122.64, MATD3 BASE 3367.93, MATD3 PQL 2443.40 global steps/s. Versus canonical 1 MB root averages this is about 4.13x, 5.79x, 32.28x, 25.56x respectively.
- BASE bottleneck has shifted to learner: MASAC learner 2223.96 s vs collector-wait 30.32 s; MATD3 learner 999.95 s vs collector-wait 83.17 s. Do not optimize packet path first now.
- Rough early 20k ETA using historical 1 KiB episode/step relation: ~11.87 h MASAC BASE, 8.77 h MASAC PQL, 3.30 h MATD3 BASE, 4.53 h MATD3 PQL. Treat as early estimates, not guarantees.
- Evidence: `docs/superpowers/evidence/2026-10-24-1kb-cohort-launch-and-live-throughput.md`.
- Next controlled task: benchmark learner-side AMP/compile/fused on a spare account without mutating these four production lineages.
