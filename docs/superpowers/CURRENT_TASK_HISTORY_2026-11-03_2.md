- 18 visible columns, 4 pinned: Name/State/Created/Runtime plus stage/algorithm/mode,
  account/seed, progress, cohort/eval delivery/success, budget complete, stop reason,
  failure type.26 line plots + 1 Vietnamese guide,12 individual runs/plot, no
  aggregation, all real states, newest Created; correct checkpoint/CPU/runtime axes.
- All 384 run IDs/states/historyLineCount unchanged:272 finished82 failed30 crashed,
  0 running/pending at publication. Unrelated saved views unchanged; normal startup
  reconciliation preserves plots. Before specs retained in ignored outputs/wandb.
- Journal/status/guide, selected scalar evidence, independent console/disk/optional
  upload guards, accurate CPU resume/pause/failure counts and primary traceback.
  Explicit status-only now requests at transitions and at most 60s progress,
  including failed attempts. Growing files end + caught-failure/close flush.
- Real probe ybnn37l0 exposed missed atomic live update for 110s; genuine failed
  state retained. Fixed logging-only probe aqp31obe confirmed two revisions while
  running and 3 files / 3 events after finish. No rollout/quality metrics/GPU allocation.
- **Final offline CPU suite:891 passed,95 skipped,8 warnings,1389 subtests in 529.84s.**
  Job 50d61b83acc4, log /tmp/shell_mcp/jobs/50d61b83acc4.stdout.log. JUnit independently
  read 0 failures/0 errors;2375 total cases including subtests/skips. Exact source ecd597f.
- Independent source review: no remaining Critical/Important/Minor. compileall,
  diff check, no-torch supervisor import pass; secret scan 12 real values, 0 hits.
  Scientific config/algorithm/environment diff empty.
- Earlier broad889 passed2 failed exposed synthetic one-character key state leaking
  across tests, causing conservative traceback redaction to replace normal letters.
  Fixture isolation fixed this; production redaction was not weakened. Interrupted
  pre-final runs are not completion evidence.

## Interpretation / limits / next action

Canonical 20k periodic mini-eval remains disabled; missing data stays empty. Judge
mission delivery/success on matched protocol, not loss or GPU utilization alone.
Created/Runtime describe current W&B run, not submit time or total lineage. Full
upload clock is not captured-state age. Hard kill/network loss may prevent final
traceback/upload; no universal root-cause guarantee. New logging applies to future
source-pinned jobs; historic data remains intact and cannot gain retrospective logs.

User has now explicitly authorized the sign-in required for browser visual QA. Do not start a new scientific/retraining task implicitly. Before any
later launch, recheck legitimate account/quota/idle/datasets/T4/no duplicates and
GitHub-synced owner; launch only on user instruction.

Guide: docs/training-diagnostics.md
Evidence: docs/superpowers/evidence/2026-10-27-wandb-observability.md, with live,
logging-probe and test-results JSON beside it.
Design/plan: docs/superpowers/specs/2026-10-27-wandb-observability-design.md and
docs/superpowers/plans/2026-10-27-wandb-observability.md.

## Previous completed reliability work

General MASAC hardening 820b50e, final evidence/help a046a3b. External GitHub
supervisor active cron 15 min; its orchestration is independent of sandbox.
10 local accounts authenticated at 18:17 on 27 Oct; only account_01/account_02 synced
and allowlisted in GitHub at that observation. No production roots launched.
Old MASAC source 77a1d0f heartbeat loss within 11s remains a shared external-incident
hypothesis, not proven Kaggle/OOM/ban. Retained policy-only artifacts cannot restore
lost replay/optimizer/RNG. Do not silently rotate lineage owners or call a seed
restart exact recovery.

See docs/github-actions-full-flow.md and
 docs/superpowers/evidence/2026-10-27-general-failure-hardening.md for prior evidence
and launch restrictions.
