# Diagnostic coverage audit — 28/10/2026

Source frozen and tested: `56604cc71f294cd94e5251491ab7d1397a95200b`.
No production training launched/restarted/cancelled. No scientific config,
checkpoint format, reward, precision, seed bank or evaluation schedule changed.

## Outcome and boundaries

The logs are substantially more useful for crash/failed/weak-training diagnosis,
but cannot reveal every cause. Python cannot finish a report after SIGKILL or
power loss; a last memory sample does not rule out a later peak. Platform/account
causes still require Kaggle/OS/service evidence. Existing runs using pinned old
source do not acquire new logs retroactively.

[Coverage matrix](../../diagnostic-coverage.md) maps meaningful failure/quality
situations to available evidence, next checks and limits. [Reading guide](../../training-diagnostics.md)
explains the table/plots without universal loss thresholds or automatic policy grading.

## Verified fixes

- Independent best-effort console/file/journal/upload channels. Reproduced closed
  stdout and disk-full failures, safe cohort reset, nested NaN/numpy/unknown-object
  evidence, exception-chain/resource/identity preservation and guarded BASE/PQL
  fallback. Reporting must not replace the primary exception.
- Redaction covers nested sensitive fields, configured credentials and environment
  token/password/API-key/legacy KAGGLE_KEY families. Unknown tensors/objects stay
  bounded type markers; no replay/tensor/config dump per event.14actual environment/
  credential values scanned against tracked files,0hits (no values printed).
- Journal caches selected warning/replay/PQL/reward/action/safety/report/eval
  evidence, source steps and resource sample age. Resources sampled at most60s on
  event cadence and fresh on failure, not by a new background sampler.
- BASE/PQL/offline and CPU watchdog use actual progress markers; stack file owned
  until native timer cancellation. No termination or deadlock diagnosis. Native
  learner fatal handlers enabled via PYTHONFAULTHANDLER=1. Parent negative return
  codes record signal names; SIGTERM metadata only when actually observed.
- CPU inner progress survives optional progress-publication failure. Final records
  include actual count/checkpoint/artifact/seed/scenario/overlap when supplied by
  protocol; count describes displayed result, not sum of checkpoint evaluations.
- Independent reviewer found2Important findings; both reproduced RED then GREEN in
  one fix pass.1minor deferred: timer rearm throttle gives270–300s default inactivity
  interval instead of exactly300s, avoiding per-vector-step rearm overhead. This
  can write an early diagnostic stack; it never stops a job or proves a deadlock.

## Personal Workspace and current evidence

API verified at01:01:16UTC (08:01:16Vietnam)28/10. Personal view ID preserved:
`VmlldzoxODA5MjI1Nw==` / `nw-nwusertruongdaoanhduy-w`.

- 17visible columns,4pins(Name/State/Created/Runtime). Algorithm and redundant Mode
  hidden; evaluation/episodes_completed replaces Mode. All real states retained,
  newest Created first. Unknown/absent values stay blank.
- 27plots+1guide;12individual runs per plot, no aggregation. Cohort sample count and
  disk free added; actor loss retained in advanced diagnostic profile. System
  metrics use runtime, disk/cohort use global_step, CPU episodes and checkpoint
  evaluation keep their respective axes.
- Authenticated native browser verified Personal badge,17actual column headers,
  eval count/Created/Runtime and hidden Algorithm/Mode. Mission section displays5
  panels and actual delivery/success/return series. Personal tab left as deliverable.
  Screenshots inspected internally only; no personal screenshot persisted.
- Original384IDs/states/historyLineCounts unchanged during publication and after
  canary. Saved/unrelated views and all view IDs unchanged. Final service readback
  at01:06:53UTC:385runs(273finished82failed30crashed,0running); sole addition was
  logging-onlycanary `f0vws7re`, finished.

Concrete quality example re-read at01:10:42UTC (08:10:42Vietnam): training run
`ce8f3be6` finished20,000episodes, budget complete, final cohort n3616/delivery
59.3323%/success0. CPU run `239f3eb9` finished30episodes, aggregate target delivery
48.9524%, success0/30. This is completed execution with incomplete mission outcomes
on that evaluation bank, not evidence of successful overall training, nor a proof
of the cause or a definitive algorithm ranking. Training and evaluation domains
must not be mixed. No learning-rate/reward adjustment made on that observation.

## Tests and online probe

- Full offline CPU suite on frozen56604cc: **924passed,95skipped,8warnings,
 1396subtests passed in554.26s (9m14s)**. JUnit2415cases including subtests/skips,
 0failures/0errors. Job `f82ae82e9f60`. GPU-required checks skipped; no GPU result
 is claimed in this task. No runtime changes after source freeze.
- Focused Task1:48passed/2subtests; Task2:60passed; curation:11passed; post-review
 fix set:77passed. These overlap; do not sum them as unique tests.
- compileall/diff check and no-Torch github_supervisor import pass.8warnings are7
 existing SDK Pydantic deprecations and1existing Gym async close warning in a
 deliberate nonfinite-failure test.
- Actual online logging-only probe at56604cc (`f0vws7re`) finished. Live status and
 atomic replacement confirmed while running. After finish downloaded4files:
 training-events.jsonl, training-status.json, metrics-guide.md, thread-stacks.log.
 Verified real resource data/source and native Timeout stack. Probe used3s timer
 only for verification; production default remains300s. No rollout, policy-quality
 metric or GPU training fabricated. Do not infer live stack-upload latency from
 its final-file success.

## Rulings and remaining minor

1. Existing isolated host worktree reused; cloud skill helper scripts unavailable
   in remote Sandbox, so a scoped manual ledger/brief boundary replaced helpers,
   preserving tests/review gates. Cost if wrong: continuity/bookkeeping gaps;
   versioned commits/ledger/evidence make the boundary reviewable.
2. Implement documented bounded choices under existing user authorization without
   another routine approval pause; no science/eval schedule changes. Cost if wrong:
   preference-level UI/evidence choice; changes are versioned and reversible.
3. Reviewer did not judge live service/fullsuite; independently verify API/browser/
   canary/fullsuite rather than infer them from code review. Cost if wrong: stale
   operational claims; timestamped readbacks and source-matched JUnit limit this.
4. Integrate using existing authorized main+feature push and retain host worktree;
   inspect remote ancestry and atomic hash readback, never force. Cost if wrong:
   shared-branch effect; no production dispatch or divergent-history overwrite.

Deferred minor: watchdog rearm precision up to30s early, as described above.

## Versioned evidence

- [Tests](2026-10-28-diagnostic-coverage-tests.json)
- [Personal API](2026-10-28-diagnostic-coverage-personal.json)
- [Browser](2026-10-28-diagnostic-coverage-browser.json)
- [Online probe](2026-10-28-diagnostic-coverage-online-probe.json)
- [Final service](2026-10-28-diagnostic-coverage-service.json)
- [Quality example](2026-10-28-diagnostic-coverage-quality-example.json)
- [Review](2026-10-28-diagnostic-coverage-review.json)
- [Continuity ledger](2026-10-28-diagnostic-coverage-ledger.md)

## Integration

Runtime source56604cc and completion records95c4fe5 were pushed atomically to
main+feat/cpu-replay-20261022. Both remote hashes matched; GitHub connector fetched
95c4fe5 successfully. This follow-up records that verification only; no runtime,
tests, config or workflow changed after the924-pass source freeze.
[Integration record](2026-10-28-diagnostic-coverage-integration.json).
