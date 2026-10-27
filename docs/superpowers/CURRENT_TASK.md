# Current task: GitHub Actions external supervisor migration — COMPLETE / DEPLOYED

Updated 2026-10-27. Final source commit `d5295874dce715d972a518bf6660c68312141fce` is pushed to both `main` and `feat/cpu-replay-20261022`. Existing untracked local `wandb/` is preserved and must not be committed.

## User request

Replace the old long-lived Kaggle CPU `continuation-watch` controller with an external GitHub Actions supervisor so the local Sandbox/laptop may be closed while cloud continuation and final CPU evaluation/visualization still progress. Reuse the proven credential pattern from `/home/aduy/Documents/NCKH/uav_search_target` and determine how that project sourced keys.

## Result

- GitHub Actions is now the orchestration authority for new `--auto-continue` lineages.
- Kaggle runs bounded compute only: GPU training/continuation and CPU evaluation/visualization.
- New supervised kernels use reserved `gh-` refs, record `orchestration=github_actions`, and do not create `continuation-watch` controllers.
- GPU completion emits `continuation.json` with `status=external_pending`; GitHub reconciles the missing next stage.
- Reconciliation is idempotent through deterministic continuation/CPU refs.
- CPU retry/resume uses `evaluation-progress.json` and `uav-media-<hash>-aNN` refs.
- Failed/crashed training can continue only from a verified immutable full-state resumable checkpoint; historical policy-only crashes remain blocked from exact resume.
- The supervisor scans only reserved refs (`gh-tra...`, `gh-val...`, `uav-media...`) and never reads unrelated Kaggle notebook outputs.
- Account use is fail-closed through `KAGGLE_ENABLED_ACCOUNTS`; current nonsecret allowlist is `account_01,account_02`. Stale secrets for disabled accounts can remain stored but are ignored.

## Old-project key pattern

The old project used GitHub repository secrets for Kaggle API tokens (`secrets.KAGGLE_API_TOKEN` and account-specific secret names in a matrix). W&B/HF credentials used by Kaggle compute came from private Kaggle credential datasets. No old local script that populated the GitHub secrets was found.

The new repo includes `scripts/sync_github_supervisor_secrets.py`, which is designed to read local `key.txt` and pass values to `gh secret set` / `gh variable set` through stdin without placing values in argv or logs. In this session, the execution environment blocked direct upload of secret values, so do **not** claim the W&B/HF/Kaggle values were refreshed. Existing repository secret names were already present; the successful GitHub dry-run proves the enabled Kaggle credentials were usable for kernel listing.

## Deployment proof

- Workflow: `.github/workflows/uav_external_supervisor.yml`, `UAV External Supervisor`, ID `398085505`.
- Schedule: every 15 minutes plus manual `workflow_dispatch`; non-overlapping concurrency.
- Workflow left `active` after verification.
- Failed pre-hardening runs are preserved as evidence:
  - `40484608284` at `f1b2035`: stale account secrets/unrelated kernels were scanned.
  - `40485463400` at `0bd7828`: allowlist fixed stale accounts, but unrelated notebook outputs still produced reconciliation errors.
- Final manual dry-run `40488127508` at `d529587`: **success**. Reconcile report contained `account_01: []` and `account_02: []`; no `would_submit_*` or `submitted_*` action and no Kaggle child submission.

## Verification

- Focused final cloud suite: **88 passed, 38 subtests passed**.
- Full CPU repository suite: **855 passed, 95 skipped, 8 warnings, 1,379 subtests passed in 513.94 s** with `HF_HUB_OFFLINE=1 WANDB_MODE=disabled`.
- `python -m compileall -q uav_network scripts`: exit 0.
- `git diff --check`: clean before final documentation update.
- `key.txt` is ignored (`key*.txt`) and not tracked.
- No literal `KGAT_...` token value was found in tracked source.
- No production training job was launched, restarted, cancelled, or deleted by deployment verification.

## Evidence

- Design: `docs/superpowers/specs/2026-10-27-github-external-supervisor-design.md`
- Plan: `docs/superpowers/plans/2026-10-27-github-external-supervisor.md`
- Final evidence: `docs/superpowers/evidence/2026-10-27-github-external-supervisor.md`

The earlier crash/W&B/recovery/acceleration records remain in `docs/superpowers/evidence/` and git history. No further source fix is queued for this task.
