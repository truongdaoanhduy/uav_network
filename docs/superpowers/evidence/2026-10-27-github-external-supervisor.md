# GitHub Actions external supervisor migration — final evidence

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Replace the long-lived Kaggle `continuation-watch` controller with a short-lived GitHub Actions supervisor while preserving the scientific training/evaluation configuration and existing W&B/HF history. No production training lineage was launched, restarted, cancelled, or rewritten during deployment verification.

## Old-project reference

The reference project `/home/aduy/Documents/NCKH/uav_search_target` contained:

- `.github/workflows/kaggle_gpu_cpu_pipeline.yml`
- `.github/workflows/kaggle_gpu_cpu_reconcile.yml`

The workflows sourced Kaggle API credentials from GitHub repository secrets (`secrets.KAGGLE_API_TOKEN` and account-specific secret names in the reconciler matrix). W&B/HF credentials for Kaggle compute were provided through private Kaggle credential datasets, not embedded into workflow YAML or public kernel specs. No historical local `gh secret set` helper was found in that repository.

## Implemented architecture

- GitHub Actions is the orchestration authority for new `--auto-continue` lineages.
- New supervised root kernels receive a reserved `gh-` slug marker and `orchestration=github_actions`.
- Kaggle performs bounded GPU training/continuation and CPU evaluation/visualization only.
- GPU completion writes a credential-free external transition receipt; it does not self-submit the next stage.
- Deterministic child refs make reconciliation idempotent.
- CPU evaluation resume uses `evaluation-progress.json`; CPU retry refs use `uav-media-<hash>-aNN`.
- Hard-kill recovery requires a verified immutable full-state resumable artifact with matching W&B run ID, git commit, episodes and global step. Policy-only historical checkpoints are not treated as exact recovery.
- The external supervisor only scans reserved refs (`gh-tra...`, `gh-val...`, `uav-media...`) and never reads unrelated Kaggle notebook outputs.
- Enabled account selection is explicit through repository variable `KAGGLE_ENABLED_ACCOUNTS`; stale repository secrets for disabled accounts may remain stored but are not used.

## Commits

- `7ebd6e0` — workflow, safe local secret-sync helper, design/plan and tests.
- `c18eece` — retire Kaggle controller launch path for new auto-continue jobs.
- `73dd34d` — external transition receipts instead of GPU self-submission.
- `37a0af8` — idempotent GitHub reconciler and CPU retry/resume.
- `f1b2035` — strict verified full-state hard-kill recovery.
- `0bd7828` — explicit enabled-account allowlist.
- `8e17edf` — reserved `gh-` namespace for externally supervised lineages.
- `9bce1c7` — unrelated-kernel filtering hardening.
- `d529587` — reconcile only reserved supervisor namespaces, including CPU retry refs.

`main` and `feat/cpu-replay-20261022` both pointed to `d5295874dce715d972a518bf6660c68312141fce` after the final source push.

## GitHub deployment evidence

Workflow: `UAV External Supervisor`, workflow ID `398085505`, default branch `main`.

Repository variable set during deployment:

- `KAGGLE_ENABLED_ACCOUNTS=account_01,account_02`

Direct upload of secret *values* from local `key.txt` was blocked by the execution environment's safety layer, so this record does **not** claim that W&B/HF/Kaggle secret values were refreshed during this task. The repository already exposed the required secret names before deployment. The successful dry-run demonstrates that the existing Kaggle secrets for the enabled accounts were sufficient to list owned kernels. W&B/HF remote API values were not exercised because no reserved GitHub-supervised lineage existed in the dry-run result.

Deployment runs:

1. Run `40484608284`, source `f1b2035`: failed. Pre-allowlist implementation scanned stale account secrets and unrelated historical kernels.
2. Run `40485463400`, source `0bd7828`: failed. Allowlist worked, but the enabled account still contained unrelated notebooks whose malformed/unavailable outputs caused reconciliation errors.
3. Run `40488127508`, source `d529587`: **success**. `workflow_dispatch`, `dry_run=true`; reconcile step completed successfully. Report was exactly empty managed sets for the two enabled accounts (`account_01: []`, `account_02: []`), with no `would_submit_*` or `submitted_*` action.

The workflow was left `active` after the successful dry-run. No Kaggle child was created by the dry-run.

## Verification

TDD/regression gates included observed RED failures before each fix and GREEN after implementation.

- Final focused cloud regression: **88 passed, 38 subtests passed**.
- Final full repository CPU suite with `HF_HUB_OFFLINE=1 WANDB_MODE=disabled`: **855 passed, 95 skipped, 8 warnings, 1,379 subtests passed in 513.94 s**.
- `python -m compileall -q uav_network scripts`: exit 0.
- `git diff --check`: exit 0 before documentation finalization.
- `key.txt` is not tracked and is ignored by `.gitignore` (`key*.txt`).
- Literal `KGAT_...` token-prefix scan over tracked source found no token values.
- Supervisor lightweight import was previously verified with imports of `torch` rejected: `LIGHTWEIGHT_IMPORT_OK`.

## Operational behavior after migration

- Closing the local Sandbox/laptop does not terminate the GitHub Actions supervisor because orchestration runs in GitHub's hosted runner.
- A future supervised GPU segment that ends cleanly is reconciled by the next scheduled/manual workflow invocation.
- A failed/crashed segment is resumed only from a verified exact full-state recovery point.
- A final completed episode budget transitions to the existing canonical CPU evaluation/visualization protocol.
- Existing historical Kaggle/W&B/HF jobs are not adopted retroactively unless they carry the new reserved marker and `orchestration=github_actions` receipt.
