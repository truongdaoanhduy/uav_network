# W&B observability: evidence and reading contract

Status: source, tests and API publication complete; direct browser rendering awaits sign-in authorization.

## Scope and source

User requested clear online W&B plots/logs, Created/Runtime and useful run-table
columns, without metric clutter. Baseline a046a3b; final runtime source
`bc1b6479a6cc9ebed80aeb756ddf8b0811fb7a79`; fixture-only follow-up
`ecd597f928f68a1f455a943ff490cda0888ae5e3` isolates synthetic authentication
state when asserting redacted tracebacks.

[Design](../specs/2026-10-27-wandb-observability-design.md),
[plan](../plans/2026-10-27-wandb-observability.md),
[human guide](../../training-diagnostics.md).

Before: Personal view had 50 plots and more than 30 visible/pinned columns;
Created was hidden. Primary saved view retained 18 older plots and different
sorting/axes. Existing rich metrics were available, but their populations and
file publication needed clearer presentation.

## Final behavior

- Personal and primary saved view target 26 line plots plus one Vietnamese guide:
  completed mission4, progress3, periodic fixed-policy evaluation2 (collapsed),
  final CPU episode2, delivery bottlenecks4, numerical5 (collapsed), resources4,
  recovery2 (collapsed). Each plot selects at most 12 separate runs, no aggregation.
- Runs table: 18 columns; only Name/State/Created/Runtime pinned. Stage, algorithm,
  mode, account, seed, global step, completed training episodes, cohort
  delivery/success, aggregate eval delivery/success, budget complete, stop reason,
  failure type. All real states; newest Created first; no group/ID filter.
- Training/cohort uses global_step; periodic evaluation uses source checkpoint
  step; CPU episode curves use completed-evaluation count; native system metrics
  use runtime seconds. Native CPU/RAM and both GPU/memory series verified from
  historic MATD3 run ce8f3be6 (901 samples in relevant series); allocated memory is %.
- RunJournal keeps UTC/source/run/algorithm/mode/seed/account identity, actual
  phase/progress and selected scalar learner/mission/evaluation/recovery evidence.
  It does not serialize tensors/replay/full config or arbitrary reprs. Secret
  fields/known credential values are removed/redacted; NaN/Inf explicit strings.
- Console, append and atomic latest status have independent I/O guards.
  Optional file publication errors remain contained. CPU reporting preserves
  the original exception and real completed counts across resume, pause,
  non-resumable failure and close failure. Attempt-specific recovery errors clear
  after a new/successful attempt; checkpoint/retry/state format stays unchanged.
- Files: training-events.jsonl, training-status.json, metrics-guide.md and
  existing diagnostics/cohorts/failure.json/batch/state/config/CPU metrics.
  Status registers live plus explicit now requests at transitions and at most
  every 60s at progress cadence, including failed attempts. Growing journal and
  diagnostics use end registration and explicit now requests at caught failure/
  close. Large raw episodes/replay are not live-synced by this feature.

## Cloud logging integration proof

SDK 0.30.0, two genuine logging-only runs; no env rollout, policy quality metrics,
GPU allocation, checkpoint upload or production launch:

1. [ybnn37l0](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ybnn37l0)
   at source cf5c921: first live revision confirmed; atomic revision2 was not
   downloaded within 110s. After finish, revision2 was present. The run remains
   failed because that integration assertion actually failed. Do not rewrite it.
2. [aqp31obe](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/aqp31obe)
   at source 6255ecc: fixed explicit refresh confirmed both revisions before
   finish. Revision 2 event 16:05:39Z; backend confirmation 16:05:49Z. Finished
   16:05:56Z (23:05:56 Vietnam). All three files downloaded, and events were exactly
   probe_started, probe_atomic_replace, probe_complete. No training values logged.
   Subsequent bc1b647 only bounds attempts when optional save raises; successful
   cloud delivery path is unchanged.

Official SDK documentation and actual installed Run._save/Go runfiles uploader
were inspected to understand live/now/end policies. The observed missed refresh
is evidence for explicit bounded requests, not a promise about all SDK versions
or network delivery within60s. Probe reports are in ignored
outputs/verification/wandb-logging-probe*/verification.json.

## Verification and review

- Additional observed REDs covered closed stdout in journal/recovery/reporting,
  CPU prefix and non-resumable completion counts, original failure preservation,
  stale recovery error, final paused-summary print and bounded status requests.
- Focused checks after main review fixes:75 passed,9 subtests; handoff/publication
  22 passed,9 subtests. Latest logging/recovery/tracking/CPU suite56 passed after
  bounding failed upload attempts. These are separate test invocations, not a
  sum of unique tests.
- First broad run exposed stale test contract/fixtures; migrated real journal
  state and diagnostic callbacks while retaining actual artifact/evaluation
  assertions. Interrupted pre-final suites are not completion evidence.
- Independent Superpowers reviewer approved final source bc1b647 with no remaining
  Critical, Important or Minor findings. Earlier findings were reproduced and fixed.
- compileall/diff checks pass. No-torch GitHub supervisor import passes. Final
  tracked/candidate secret scan checked 12 local values with 0 hits.
- Broad runtime verification 04cc82187d77:889 passed,2 failed,95 skipped. Both
  failures were assertions on clear text after earlier authentication tests left
  a one-character dummy key in the process environment; conservative redaction
  correctly replaced that character. No credential-redaction behavior was relaxed.
  Fixture-only ecd597f isolates realistic dummy values and a nonexistent key file;
  the entire CPU resume file passed 8 tests with inherited w/k/h dummy keys injected.
- Final full offline CPU suite on ecd597f: **891 passed, 95 skipped, 8 warnings,
  1389 subtests passed in 529.84s (0:08:49)**. Job 50d61b83acc4; JUnit independently
  read:0 failures/0 errors,2375 cases including subtests/skips. Runtime code is bc1b647.
  [Versioned test results](2026-10-27-wandb-observability-test-results.json).
- Source pushed atomically to main and feat/cpu-replay-20261022 at ecd597f;
  ls-remote readback confirmed both exact SHAs. This final evidence commit changes
  documentation only. No production training launched.

## Workspace publication proof

Published and read back at 2026-10-27T16:21:09Z (23:21:09 Vietnam):

- Personal `VmlldzoxODA5MjI1Nw==`, primary saved `VmlldzoxODA5Nzc1Ng==`
  (`nw-gmcixx5iveu-v`) retained their existing identities. No extra saved view.
- Each has 18 visible columns, 4 pinned, 26 line plots + 1 guide, per-plot limit 12,
  aggregate=false, expected axes, all states, newest Created sort.
- Normal startup reconciliation was exercised and preserved the authored panels.
- All 384 run IDs/states/historyLineCount values were identical before/after:
  272 finished,82 failed,30 crashed; 0 running/pending. Fingerprint
  `94a3149c60daf3d98b4133cf78074ca6c4ccf9a224e8f658093b3171d8690373`.
- Unrelated saved views remained byte-equivalent. Backup specs and before-run
  records retained under outputs/wandb/observability-20261027.
- Versioned [live verification JSON](2026-10-27-wandb-observability-live-verification.json)
  and [logging probes](2026-10-27-wandb-logging-probes.json).
- Historical training run ce8f3be6 has actual cohort delivery/search/success/return,
  completed-episode, critic-loss and both GPU series. CPU run 239f3eb9 has 30 episode
  delivery/success points and aggregate delivery 48.95238%,success 0/30. Thus finished
  is distinct from satisfactory policy performance.

The user's opened tab was inspected and identified as Weights & Biases at
forge.coreweave.com/wandb/.../workspace?nw=gmcixx5iveu. It was signed out and
showed Page not found/Login. The initial generic-origin review rejection was
resolved by selecting that exact W&B tab. Login then reached id.coreweave.com,
where automatic approval review rejected access because W&B inspection had not
explicitly authorized that identity-service sign-in. No credentials were entered
and no login bypass attempted. API publication/readback is verified; visual
frontend rendering is unverified pending explicit authentication authorization.

## Interpretation and limits

Policy quality starts with delivery/success on matched scenarios/seeds/budgets;
return is supporting evidence. Loss, Q, TD and gradients diagnose learning, not
an automatic good/bad grade. PDR is packet delivery, not target delivery. Check
cohort n/partial/source and do not compare live-world episode ages as closed
outcomes. BASE sample-use is segment-based; PQL may restore replay counters.

Canonical 20k profile still disables periodic mini-evaluation. Missing curves
remain empty. CPU aggregate eval columns are eval/*, not training final_* or the
last CPU episode. No evaluation schedule, reward, precision, algorithm, replay,
scenario or checkpoint semantics changed. New logging applies to future jobs
pinned to this source; existing historical metrics can be replotted now.

Native runtime/Created describe the current W&B run, not total lineage or Kaggle
submission. Full upload clock measures time since confirmed upload, not captured
state age. Hard kill/network loss can prevent the final traceback/event upload;
logs identify observations and context, not every possible root cause. No
production training was started/restarted/cancelled; no run history was deleted
or states rewritten. Two logging-only diagnostic runs were intentionally added.
