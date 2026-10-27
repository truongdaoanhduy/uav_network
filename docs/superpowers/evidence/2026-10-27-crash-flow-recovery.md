# G3 training crash audit, W&B visibility and bounded recovery

Observed 2026-10-27T05:59:29.381088+00:00 (12:59:29 Vietnam).
Implementation source: `0fe0e7c4bb7fe7d2a18a6fad64c8cbdba4a0505d`.
Visibility source: `8cfc5da`.
Historical G3 production source: `77a1d0f52ed4e78bc2823c3ada03d661568cd923`.
Branch: `feat/cpu-replay-20261022`.

## Outcome

The complete project query returned 382 runs: 271 finished, 81 failed, 30 crashed, zero running/pending. These are project-wide historical counts, not 111 newly failed jobs.

All four G3 MATD3 learners completed the actual 20,000-episode budget, their controllers completed, and their CPU evaluations finished 30 episodes each with W&B/HF visualization publication status ok. All four MASAC learners and their controllers are crashed; their available artifacts do not contain full training state. No new production cohort, restart, cancellation, state rewriting, or remote artifact deletion was performed. One isolated diagnostic on a legitimately authenticated idle two-T4 account completed successfully.

| Algorithm/mode | HYBRID CPU75/GPU25 replay | FULLGPU replay |
|---|---|---|
| MASAC BASE | [63202ad9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/63202ad9), crashed, 8,192 episodes | [25eecfd1](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/25eecfd1), crashed, 8,192 episodes |
| MASAC PQL | [6f0bcde3](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/6f0bcde3), crashed, 7,045 episodes | [3b39b0c8](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3b39b0c8), crashed, 8,196 episodes |
| MATD3 BASE | [ce8f3be6](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ce8f3be6), complete, 20,000 episodes | [93fdf127](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/93fdf127), complete, 20,000 episodes |
| MATD3 PQL | [3fcf25c7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3fcf25c7), complete, 20,000 episodes | [76b73021](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/76b73021), complete, 20,000 episodes |

CPU evaluation IDs: HYBRID BASE 15db2a32, FULLGPU BASE 9c62d7c4, HYBRID PQL 239f3eb9, FULLGPU PQL ba98364f.
Completion is established from episode_budget_complete=true, stop_reason=episode_budget and CPU publication metadata, rather than W&B finished alone.

## Cause assessment

The eight MASAC learner/controller heartbeats stop within eleven seconds, 2026-10-26T19:31:57Z through 19:32:08Z (02:31:57–02:32:08 Vietnam, October 27). The four learners had different progress, replay layouts and BASE/PQL execution paths. No caught failure report or traceback was available in their W&B files. This correlation strongly favors a common external platform/session/account/connectivity or tracking interruption. It does not establish the exact Kaggle termination mechanism, or prove that every process died at the last heartbeat.

Fresh read-only access to private kernels using their owners' credentials failed authentication for accounts 16–19. That blocks retrieving authoritative kernel logs/status now. It is not evidence of when or why those credentials became invalid, nor proof of an account ban. No account switching to evade restrictions was attempted. Public W&B status API evidence was unavailable; no outage conclusion is claimed.

| Possibility | Evidence and conclusion |
|---|---|
| New code update, FP16 or torch.compile | All eight learners were pinned to 77a1d0f, AMP off and compile false. Later branch updates and the measured acceleration policy were not injected into these jobs. They cannot explain this old cohort through changed runtime flags. |
| Configured session/learner deadline | Receipts specify 39,600 s kernel and 36,000 s learner limits. Heartbeat loss was approximately 6 h 23 min after launch, before either configured deadline. A different platform limit or preemption remains possible. |
| Scenario generator failure | The earlier G2 seed-9642 placement failure was independently reproduced and fixed in G3. No analogous error was emitted here; the four MATD3 G3 learners completed 20k. This is absence of supporting evidence for that specific recurrence, not proof of every random layout. |
| MASAC numerical or algorithm error | Last recorded metrics are finite and training had progressed well beyond warmup. No caught numerical traceback was found. A final unsampled error cannot be excluded. Shared heartbeat timing across controllers weakens four unrelated algorithm failures as an explanation. |
| Host/GPU memory | Sampled oomKills=0. Last host RSS was about 2.5 GB FULLGPU and 11.9 GB HYBRID. FULLGPU GPU1 memory was high, about 89.35% BASE and 96.55% PQL. Sampling cannot exclude a final OOM/peak or external kill; no new OOM is proven. |
| Disk/upload/checkpoint failure | No failure report establishes it. The latest MASAC artifacts were only roughly 74–92 MB policy files, not 15 GB full-state uploads. Large future snapshots nevertheless require bounded staging, which this change addresses. |
| Flow/controller resilience | Controllers stopped in the same window as learners; their control/auth dependencies did not provide independent recovery. More decisively, current MASAC artifacts lack replay/optimizer/RNG state, so exact continuation after abrupt interruption was unavailable. This recoverability defect is confirmed and fixed for future opted-in runs. |
| W&B presentation | The previous default state-descending sort put finished rows ahead of crashed rows. Existing saved-view limits/filters had also previously hidden runs. The new shared reconciliation explicitly restores all-state visibility and sorts crashed first. |

Exact external cause remains unproven without owner/platform logs. Earlier records that describe external account invalidation as a definite root cause should be read as a hypothesis superseded by this assessment.

## Applied W&B changes

Updated the existing [Personal workspace](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/workspace) and saved view “UAV Network 2D - MASAC and MATD3”. Cleared run filters/search/selections, enabled the runset, raised maxRuns to 1000, and set state ascending then creation time descending. State, Algorithm and Mode are pinned; episode_budget_complete, stop_reason and session_status are available as completion columns.

Readback at 05:20:28 UTC confirmed the first eight server rows were crashed, all 382 authentic run states/counts were preserved, and both chart configuration hashes were unchanged. The Personal workspace was rechecked at 05:59:29 UTC: no filters, State pinned, ascending state sort, maxRuns=1000. The separate diagnostic-only validation saved view was deliberately preserved; open Personal or the primary saved view to see all states.

## Implemented recovery

Reused the existing exact BASE/PQL serializers and HF manifest uploader. Generic configuration is opt-in; the canonical comparison profile now requests full resumable backups:

```yaml
tracking:
  resumable_checkpoint_interval_seconds: 3600
  resumable_checkpoint_max_uploads: 4
```

Effective interval is max(requested interval, learner soft budget / cap). With a 36,000 s session this is 9,000 s, about 2.5 hours, plus checkpoint/upload time before the next interval. The cap counts attempts, including failures, per session; it excludes the existing final save. It is not a guarantee of four successful uploads or a maximum loss window under all failures.

- Wall-clock checks run after complete collection/reset boundaries, including before any episode finishes. PQL drains workers; full state retains replay, optimizer, scaler, RNG, environment, and pending update clocks.
- Only one unique owned temporary recovery directory exists per attempt. It is removed after confirmation or failure, before the next backup/final checkpoint. Final/input/resume_latest files and all remote history are preserved.
- Full snapshots bypass light policy upload throttling. Light uploads can continue moving latest while the independent immutable recovery/checkpoint_artifact pointer remains usable.
- Initial W&B publication omits the resumable alias. Only after logged.wait succeeds does the code persist that alias and advance the confirmed summary pointer. Failed confirmation retains the preceding pointer/alias. Both periodic and final paths have alias-aware regression tests.
- Backup errors are bounded and recorded through recovery/status, error_type, attempts, bytes and elapsed time. Logging failure itself has a credential-free fallback and does not become a new learner failure.
- This periodic path is HF-only. Explicit legacy W&B backend overrides remain valid and disable periodic large HF backups. PQL HF full backups require replay checkpointing.

Old MASAC artifacts cannot be retroactively converted into exact resume state. Their policy weights are preserved but using them would be a warm restart, not recovery of lost replay/Adam/RNG. This change does not deploy independent supervision, automatically resubmit a crashed lineage, repair account credentials, or prevent external hard kills.

Immutable HF snapshots retain remote history and consume storage cumulatively. With roughly 15.5–15.9 GB per full snapshot, four attempts plus final checkpoints are a material remote storage/network cost. Local staging is bounded; total remote storage is not constant.

## Verification on exact implementation source 0fe0e7c

Fresh full CPU suite, offline HF and disabled W&B:

```text
python3 -m pytest -q --disable-warnings --junitxml=outputs/crash-flow-audit-20261027/cpu-full.xml
834 passed, 95 skipped, 8 warnings, 1369 subtests passed in 535.64s
exit 0
```

The full suite began at 2026-10-27T12:44:09+07:00. No source edits followed this verification. Initial pre-fix results (7 failed/824 passed) are retained under cpu-initial, not used as final success evidence. Focused changes followed RED/GREEN checks for visibility, timing/caps, exact resume, legacy backend compatibility, failure containment and deferred alias promotion. The independent review found one Important alias ordering defect; it was reproduced in periodic and final cases, fixed, and the affected combined suite passed 39 tests. No Critical or actionable minor finding remains.

Real private two-T4 diagnostic: duytrngoanh/uav-recovery-check-20261027, started 05:49:43Z, finished 05:53:56Z, terminal passed. It mounted no credentials, disabled W&B/HF access and launched no production training. Hardware was two Tesla T4s, each 15,636,037,632 bytes, PyTorch 2.11.0+cu128/CUDA12.8.

- Focused real-CUDA tests: 20 passed, 5 existing opt-in tests skipped.
- Eight small real-CUDA exact recovery cases passed: HYBRID/FULLGPU × MASAC/MATD3 × BASE/PQL, FP16 and compile OFF. Synchronous PQL was used for exact continuation. This does not claim compile-ON recovery or 20k training quality.
- Two extra CPU asynchronous PQL probes (MASAC/MATD3) restored at step 2 before the first episode, passed the checkpoint validator/environment restore and performed 24 subsequent updates with finite parameters. Bitwise equivalence of asynchronous schedules is not claimed. An initial harness budget-validation rejection was corrected only in the harness; its original log is retained.

Capacity-only synthetic probe used primary geometry E4096, full replay 1,000,000 joint transitions (13,105,000,000 bytes), MASAC FULLGPU. It produced resume.pt=15,485,450,455 bytes and policy=26,388,043 bytes (total 15,511,838,498). Disk free changed 20,940,464,128 → 5,428,613,120 → 20,940,464,128 bytes after cleanup. Peak process RSS was 17,699,971,072 bytes; serialization 48.031 s, streaming SHA256 59.964 s, whole stage 108.445 s. This proves capacity/cleanup on that diagnostic shape, not universal peak RAM sufficiency for all workloads.

The capacity state was synthetic, not a valid learned checkpoint or quality result. Its “uploaded” coordinator status used a local verifying fake tracker: published_to_hf=false. No 15 GB network transfer was performed by this diagnostic, so production HF throughput/upload duration and end-to-end transfer reliability remain unmeasured here. Existing MATD3 final manifests establish that normal full-size HF files had previously been published; that is separate evidence.

## Review resolution and continuation

The [independent review](2026-10-27-crash-recovery-review.md) and [sanitized status snapshot](2026-10-27-crash-flow-status.json) are adjacent. [CUDA receipt/results](2026-10-27-crash-recovery-cuda.json), [diagnostic driver](2026-10-27-crash-recovery-driver.py) and [GPU probe](2026-10-27-crash-recovery-probe.py) preserve the reproducible diagnostic.

Reviewer limitations were resolved as follows: fresh CPU suite passed; actual two-T4 serialization/restore measured; workspace/current states verified by parent; async continuation exercised without bitwise claims. Exact historical external cause, live W&B alias fault injection, and real full-size HF transfer remain explicitly unproven. Independent supervision and old missing replay reconstruction are outside this patch. Existing deliberate dashboard creation/layout rebuild semantics are unchanged.

No production restart is queued. Before any next launch, refresh legitimate owner credentials and recheck authentication, idle status, quota and duplicate lineages. Restore from the immutable confirmed recovery reference and validate source/protocol/precision compatibility; do not silently treat a policy-only artifact as an exact continuation or merge the historical off/off lineages with new acceleration settings.
