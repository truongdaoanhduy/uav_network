# Current task: GitHub Actions external supervisor — COMPLETE / DEPLOYED

Updated 2026-10-27T18:02:13+07:00. Supervisor implementation commit `a0c04c34d3e09408d13205219a1221baa8083131`; launch guide/help/ignore follow-up `bd184a783f4b5dac791894b8f248f1bb522662e3`. Source and final documentation are pushed to `main` and `feat/cpu-replay-20261022`.

## User request and result

Replace Kaggle's long-lived CPU `continuation-watch` controller with GitHub Actions so GPU training, continuation and final CPU evaluation/visualization progress after local Sandbox/laptop shutdown.

- New `--auto-continue` root launches use `orchestration=github_actions`, `external_supervisor_required=true` and reserved `gh-tra-g...`/`gh-val-g...` refs.
- They do not create a Kaggle CPU controller. Kaggle performs bounded GPU/CPU compute; GitHub reconciles the next stage.
- Incomplete clean segments continue from full state. Full 20k completion transitions to CPU evaluation: 30 episodes, seed 2000000044, visualization and verified W&B/HF publication.
- Crash recovery requires a verified immutable resumable checkpoint matching lineage/commit/progress. Missing full state blocks recovery; historical policy-only MASAC crashes cannot be called exact resumes.
- Deterministic child refs and workflow concurrency prevent duplicate submissions. Partial CPU evaluation retries use saved progress.
- Scientific configs, rewards, replay, episode budgets, AMP/compile policy and history remain unchanged.

## Live deployment proof

Workflow: `.github/workflows/uav_external_supervisor.yml`, `UAV External Supervisor`, ID `398085505`; `active`, 15-minute schedule plus manual dispatch, no overlapping reconciliation.

Latest successful manual dry-run on `main`: [40492765863](https://github.com/truongdaoanhduy/uav_network/actions/runs/40492765863), source `bd184a783f4b5dac791894b8f248f1bb522662e3`, completed at 18:00:11 Vietnam time on 27/10/2026.

- GitHub-held Kaggle identities: `account_01=duytrngoanh`, `account_02=vuliu123456`; matched the current local `key.txt`.
- W&B authentication/project read: `ok`; HF identity/repository read: `ok`.
- Reconcile report: `account_01: []`, `account_02: []`; no managed lineage and no child submitted.
- Fresh W&B query at 2026-10-27T18:02:13+07:00: **0 running/pending**.
- No production training was launched, restarted, cancelled, deleted or adopted during this task.
- This proves deployment, credential/read access and empty-set reconciliation; no new production train → visualize cycle was launched to claim live end-to-end completion.

## Account selection and credentials

GitHub secret names: `KAGGLE_API_TOKEN_ACCOUNT_01`, `KAGGLE_API_TOKEN_ACCOUNT_02`, `WANDB_API_KEY`, `HF_TOKEN`. Values were not printed or refreshed in this follow-up.

Variables: `WANDB_ENTITY=truongdaoanhduy-uit`, `WANDB_PROJECT=uav_search_target`, `HF_REPO_ID=duy95/uav-search-checkpoints`, `KAGGLE_ENABLED_ACCOUNTS=account_01,account_02`.

Local `key.txt` currently has additional enabled accounts; these are NOT enabled in GitHub automatically. Future full-flow launches must explicitly select an allowlisted account; do not omit `--account` and let the local quota selector choose an unsupervised account. Do not enable extra accounts or switch an existing lineage owner without a fresh authorized preflight.

Old reference `/home/aduy/Documents/NCKH/uav_search_target` used GitHub repository secrets for Kaggle and private Kaggle datasets for W&B/HF. No old secret-population helper was found. `scripts/sync_github_supervisor_secrets.py` uses stdin-only secret setting; earlier direct secret refresh was blocked, so historical evidence does not claim it happened.

## Verification

- Focused cloud lifecycle suite: **83 passed, 38 subtests passed**.
- Full offline CPU suite covering final supervisor behavior: **858 passed, 95 skipped, 8 warnings, 1381 subtests passed in 539.41s (0:08:59)**. Background task `650f5711fb93`, log `/tmp/shell_mcp/jobs/650f5711fb93.stdout.log`. Started on implementation `a0c04c3`; later `bd184a7` changes only guide/CLI help/gitignore, not supervisor behavior.
- `python -m compileall -q uav_network scripts`: pass.
- `git diff --check`: pass; tracked secret scan: **0 hits** against 12 local values.
- Lightweight supervisor import rejects any torch import and passes.
- Independent review: no critical/important findings. The minor test gap was fixed with explicit credential arguments, `whoami` and actual lazy iterator consumption/denial checks.
- `wandb/` logs are preserved and now ignored; `key.txt` stays ignored/untracked.

## Next action

This task is finished. Only on an explicit launch request: preflight quota/idle kernels/credential datasets/accelerator/no duplicate lineage, then launch one approved algorithm per enabled account with `--auto-continue`. Confirm Kaggle running, W&B progress and GitHub-managed manifest before saying the user can close Sandbox for that root. A W&B `running` row alone is insufficient; legacy unmarked jobs are not retroactively managed.

Guide: `docs/github-actions-full-flow.md`.
Evidence: `docs/superpowers/evidence/2026-10-27-github-external-supervisor.md`.
Design: `docs/superpowers/specs/2026-10-27-github-external-supervisor-design.md`.
Plan: `docs/superpowers/plans/2026-10-27-github-external-supervisor.md`.
