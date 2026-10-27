# Current task: W&B metrics, logs and compact dashboard — IN PROGRESS

Updated 2026-10-27T22:56:17.999239+07:00. Baseline a046a3b; initial implementation 89c6a8d. User requests meaningful online plots/logs/Created/Runtime; no production launch.

Independent Superpowers review found closed-console failures, inaccurate CPU resume/failure counts and secondary failure masking. All were reproduced RED and fixed; attempt-specific recovery errors now clear after a successful attempt. Existing tracker and axes fixtures were migrated without changing scientific/checkpoint contracts. Focused suites: 75 passed + 9 subtests; CPU handoff/publication: 22 passed + 9 subtests. compileall, diff check, no-torch GitHub supervisor import and tracked secret scan pass (12 values, zero hits).

The first full suite reported 7 failed, 872 passed, 95 skipped in 525.92s; identified outdated fixtures/column assertions, migrated since. It ran during review edits, so it is not final-source evidence. Next: freeze amended source, rerun full CPU suite and request follow-up review, then publish/read back live Personal and primary saved views. Before-state views backed up at outputs/wandb/observability-20261027/workspace-before.json. Native CPU/RAM/GPU0+1/VRAM metric names/units confirmed from actual historical W&B system rows. Human reading guide rewritten. No live view mutation or production start/restart/cancel performed yet.

Spec: docs/superpowers/specs/2026-10-27-wandb-observability-design.md
Plan: docs/superpowers/plans/2026-10-27-wandb-observability.md

## Previous completed task context

# Current task: General MASAC failure hardening — COMPLETE / DEPLOYED

Updated 2026-10-27T18:44:05+07:00. Implementation `820b50ec000eee8e08e0875868f457d104c4057f`; final evidence/help follow as a docs-only commit. Source and evidence are pushed to main and feat/cpu-replay-20261022.

## User request

Check whether additional key.txt accounts are usable, re-investigate historical MASAC crashes and implement/propose a general failure response. Do not promise every external failure is preventable or restart production implicitly.

## Fresh diagnosis

- 18:15–18:17 W&B reads: four MASAC G3 learners and controllers remain crashed, source 77a1d0f. heartbeatAt spread 11s: 02:31:57–02:32:08 Vietnam on 27/10.
- Strongest evidence remains a common external platform/session/account/connectivity/telemetry event. Exact Kaggle-side cause is unproven; heartbeat loss alone is not proof all compute died, OOM or ban.
- Old owner credentials are absent from the current key file; old replay/optimizer/RNG do not exist in their retained policy-only artifacts. They cannot be reconstructed or called exact resumes.
- No new scientific/algorithm/precision fix is claimed to explain that old off/off cohort.

## Accounts

Fresh auth/quota check at 18:17:33: 10 local enabled accounts all authenticate and return GPU quota; 01 remaining 23.35h, 02 25.03h, 03–10 each 30h. This does not prove allocation/phone verification/idle/private datasets/accelerator for a future launch.

GitHub allowlist remains account_01,account_02; matching identities duytrngoanh and vuliu123456. Additional local accounts were not silently enabled in GitHub. Root full-flow launch must explicitly choose a synced allowlisted account; do not let local quota auto-selection pick 03–10. Do not rotate a lineage owner or evade service restrictions.

## Implemented general fixes

- Per-account API/list failures no longer stop subsequent accounts; sanitized error class and account_unavailable report.
- Unresolved recovery, unknown service, missing recovery or reconcile/account errors return nonzero instead of false green.
- Verified recovery now carries initial_global_step and real environment_state_saved evidence. BASE/PQL save callbacks derive that marker from actual saved-world snapshots; W&B artifact metadata publishes it. Zero-episode recovery without this proof remains blocked.
- Quota-paused recovery reports waiting_for_quota, not submitted.
- Cancelled jobs are not automatically restarted.
- Three consecutive crash-recovery GPU children maximum; counter survives child specs and resets after a clean segment. Shared builder enforces it. The conservative gate can also block CPU handoff after a third recovered segment fails; operator review is intentional.
- --report persists credential-free JSON; Actions always uploads uav-supervisor-report for reconciliation, retention 14 days.
- Scientific configs, source pinning, replay/batch/env/episode budgets, precision policy and W&B/HF history are unchanged.

## Verification and cloud proof

- RED: 15 observed failures before fixes.
- Focused lifecycle/checkpoint suite: 135 passed, 38 subtests. Four actual small BASE/PQL × MASAC/MATD3 saved-world checkpoints now publish correct evidence and exactly restore.
- Full offline CPU suite on final implementation 820b50e: **868 passed, 95 skipped, 8 warnings, 1383 subtests passed in 506.22s (0:08:26)**. Task a163687c6241, log /tmp/shell_mcp/jobs/a163687c6241.stdout.log.
- compileall, diff check, no-torch supervisor import and tracked secret scan: pass (0 hits against 12 local secret values).
- Independent review: no Critical/Important. Minor deferred: a manual dry-run --check-services failure before reconciliation has console diagnostics but no separate JSON artifact. Scheduled reconciliation reporting is unaffected.
- Feature dry-run 40497048893 success; downloaded/read artifact 12730654618: account_01=[], account_02=[].
- Main dry-run [40497938044](https://github.com/truongdaoanhduy/uav_network/actions/runs/40497938044) success at exact source 820b50e; artifact 12731566730 uploaded (189 bytes), service probes ok, both managed sets empty. Workflow active, cron every 15 minutes. Schedule timing is advisory; no new scheduled execution/GPU full flow is claimed.
- Fresh W&B at 2026-10-27T18:44:05+07:00: **0 running/pending**. No production launch/restart/cancel/delete/adoption. No credential secret value changed.

## Next action / limits

This requested diagnosis/fix/proposal task is complete. Only on an explicit next launch request: preflight legitimate accounts/quota/idle datasets/T4/no duplicates, then launch approved roots with --auto-continue and confirm Kaggle + W&B + GitHub-managed metadata.

Recommended next reliability measurements: real ~15GB HF upload/peak resources; an earlier first positive-progress checkpoint after cost measurement; small cloud fault-injection train→recover→CPU visualize smoke. Before a first confirmed checkpoint, a hard kill cannot exact-resume progress; restart from seed must be labelled/authorized. No live GPU/new large upload was performed here.

Guide: docs/github-actions-full-flow.md
Evidence/failure-response matrix: docs/superpowers/evidence/2026-10-27-general-failure-hardening.md
Design/plan: docs/superpowers/specs/2026-10-27-general-failure-hardening-design.md and docs/superpowers/plans/2026-10-27-general-failure-hardening.md
Previous migration evidence: docs/superpowers/evidence/2026-10-27-github-external-supervisor.md
