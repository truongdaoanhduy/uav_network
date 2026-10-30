# XYZ seed44 cohort on GitHub Actions

Ngày31/10/2026. User has explicitly delegated execution choices and asked to finish
Kaggle training plus continuation/CPU visualize after Sandbox closes. This extends
the existing approved GitHub supervisor; it does not replace training algorithms.

## Goal and chosen approach

Use the current15-minute Actions reconciler to start four roots sequentially:
MASAC BASE, MATD3 BASE, MASAC PQL, MATD3 PQL, all seed44 and20000 episodes.
Every root uses the same verified shared-actor/critic Hybrid75 XYZ profile.
Existing continuation and best-policy CPU30-episode evaluation/visualization stay intact.

My implementation choice: reconstruct queue progress from Kaggle control receipts,
using an immutable unique W&B tracking.group per case. No database, background host
training controller, new dependency, or alternate account is required.
A host-only loop would stop advancing when the host disappears. Launching all roots
at once would defeat the chosen single-active-job budget. Reusing the current server
reconciler provides durable progress and its existing recovery/CPU flow.

## Exact constraints

- Only account_01, authenticated owner kesfljhk. Never rotate accounts or cancel jobs.
- Runtime source pinned481f81f288d8e84a010dba52e4c8e30b5eebd391 (math fix5fee6c7).
- 6UAV/map1000m/20T15O/1200steps/XYZ Bayesian prior0.5/learned RF/2D visualize.
- E4096/B8192/replay1M/Hybrid75/shared actor+critic/FP32/eager; warmup32.
- Training20k; validation5 and final30 use existing independent evaluation banks.
- Best actor vs full latest-resume remain separate. No XY checkpoint warm-start.
- Cohort starts disabled; enable only after all6full-ring readiness cases have terminal evidence and the selected Hybrid cases passed.
- Account quota below1hour pauses new roots; continuation uses the same account.
- No claim of convergence or multi-training-seed proof from this first seed44 cohort.

## Reconciliation contract

All owned active jobs, including unrelated/benchmark/CPU jobs, block new roots.
Read terminal managed receipts and match root source, config, algorithm, mode, seed,
budget and unique case group. Duplicate/unreadable managed roots fail closed.
An existing root is adopted after a lost submission response; it is never blindly
resubmitted. A failed root waits for existing exact recovery or a human repair.

Advance only after a matched GPU segment reports20000/episode_budget_complete and
its CPU child/retry reports30evaluations plus successful W&B/HF publications for
the same immutable checkpoint. A completed GPU stage alone does not advance the queue.
Dry-run returns the desired initial launch without mutating Kaggle.

Reuse cloud.cli.launch in a temporary clean checkout of the pinned source, with
credentials_from_environment; no raw credential is written to the plan, argv or log.
The case group survives GPU continuation and CPU controls through existing overrides.

## Files and interfaces

- configs/xyz-cohort-seed44.json: disabled four-case plan, source/owner/limits/readiness.
- cloud/cohort_queue.py: load/validate plan, reconstruct state, invoke pinned launcher.
- cloud/github_supervisor.py: optional cohort outcome after existing reconciliation.
- tests/test_cohort_queue.py: real queue decisions, fake external Kaggle/process boundary.
- Existing supervisor report artifact records each decision and all block reasons.

## Deferred deliberately

The independent/FullGPU profiles are readiness comparators, not additional20k roots
in this first cohort. Seeds100044/200044 remain a later study stage; do not present seed44
as a multi-seed paper result. Broader studies can be explicit additional case plans.
