# GitHub Actions External Supervisor Design

## Intent

Replace the long-lived Kaggle `continuation-watch` controller with a short-lived GitHub Actions reconciler so training can continue and move to CPU evaluation/visualization after the local Sandbox is closed, while the orchestration plane remains outside the Kaggle account executing the learner.

## Goals

- GitHub Actions is the production orchestration authority for `--auto-continue` lineages.
- Kaggle jobs only perform bounded compute stages: GPU train/continuation and CPU evaluation/visualization.
- A scheduled GitHub job can be rerun safely: duplicate invocations must not create duplicate continuation or CPU jobs.
- A normal time-budget stop continues from the exact full-state checkpoint and a completed 20,000-episode budget moves to the canonical 30-episode CPU evaluation/visualization path.
- Future hard-killed jobs can be recovered only from a verified resumable checkpoint; policy-only historical artifacts must never be presented as exact recovery.
- Existing W&B/HF history and run states are preserved.
- No scientific configuration, reward, batch, replay, environment count, episode budget, evaluation seed, AMP/compile policy, or scenario semantics change.

## Non-goals

- Do not repair or relabel the four historical MASAC G3 crashes; they lack exact resumable state.
- Do not evade Kaggle account restrictions or switch a lineage to a different account silently.
- Do not make GitHub Actions a long-running worker. Every reconciliation invocation must terminate quickly.
- Do not put raw credentials in workflow YAML, job manifests, logs, artifacts, or commits.

## Approaches considered

### A. Keep Kaggle CPU controller

Rejected. It survives local Sandbox shutdown, but learner and controller remain in the same external platform/account failure domain. The 2026-10-27 G3 event showed learner/controller heartbeats disappearing together.

### B. One long-running GitHub Actions controller

Rejected. It wastes runner time, is subject to hosted-runner execution limits, and would make lifecycle correctness depend on one uninterrupted workflow execution.

### C. Scheduled short-lived GitHub Actions reconciler — selected

A cron/manual GitHub workflow checks current Kaggle/W&B/HF state, performs at most the required idempotent transitions, then exits. Kaggle owns compute; GitHub owns orchestration; W&B is observability/lineage metadata; HF is durable checkpoint storage.

## Architecture

```text
GitHub Actions (scheduled/manual)
        |
        | reconcile per enabled account
        v
Kaggle GPU train/continuation ----> W&B metrics
        |                              |
        | receipts + terminal state   | recovery pointer
        v                              v
   GitHub reconciler <---------- HF resumable state
        |
        +-- incomplete clean segment -> deterministic GPU continuation
        +-- completed budget ---------> deterministic CPU evaluation/visualization
        +-- hard failure + verified resumable state -> exact recovery continuation
        +-- no exact resumable state -> blocked, never warm-restart silently
```

## Orchestration semantics

1. `--auto-continue` still prepares the same-account private credential datasets and checkpoint-storage metadata required by later stages.
2. The launch manifest records `orchestration = "github_actions"` and no Kaggle `continuation-watch` kernel is submitted.
3. A GPU stage never submits the next stage itself when orchestration is GitHub Actions. It writes `job.json`, `job-result.json`, and a credential-free transition receipt describing the desired next stage.
4. The GitHub reconciler lists recent owned Kaggle kernels, downloads only small control receipts, and ignores jobs not marked `orchestration = "github_actions"`.
5. Deterministic child references (`-cNN`, `-cpu`, bounded CPU retry refs) are checked before every push. Existing active/complete children are adopted rather than duplicated.
6. For a clean terminal segment, existing `next_stage()` validation remains authoritative.
7. For a failed/crashed terminal segment, recovery requires a W&B `recovery/checkpoint_artifact` that is immutable, marked resumable, belongs to the same source run/commit, and has monotonic verified episode/global-step counters. Otherwise the lineage is reported blocked.
8. CPU evaluation remains canonical: 30 episodes, seed 2000000044 from the production config, with W&B and HF publication verification.

## Secrets

The old `/home/aduy/Documents/NCKH/uav_search_target` project confirms the established pattern: GitHub Actions receives Kaggle API tokens from repository secrets while Kaggle compute reads W&B/HF credentials from private Kaggle datasets. The new supervisor additionally needs W&B/HF credentials itself for observation/recovery validation.

Repository secrets:

- `KAGGLE_API_TOKEN_ACCOUNT_01` ... `KAGGLE_API_TOKEN_ACCOUNT_24` for enabled accounts that have been synced.
- `WANDB_API_KEY`.
- `HF_TOKEN`.

Repository variables:

- `WANDB_ENTITY`.
- `WANDB_PROJECT`.
- `HF_REPO_ID`.

A local secret-sync command reads `key.txt` and calls `gh secret set` through stdin. It prints only secret names/status, never values. Disabled accounts are not newly uploaded. It never deletes pre-existing repository secrets automatically.

Kaggle jobs continue to receive W&B/HF through the existing private credential-dataset mechanism; secrets are never embedded in kernel source/spec JSON.

## GitHub workflow

- `workflow_dispatch` for an immediate/manual reconcile.
- `schedule` every 15 minutes. The schedule is advisory; correctness must not depend on an exact interval.
- One Ubuntu job, not a 24-job matrix. All configured account secret environment variables are passed as masked values and the Python reconciler processes only non-empty ones.
- Concurrency group disables overlapping reconciler runs without cancelling an in-progress run.
- Checkout uses full history so a pinned source commit can be verified.
- Install only minimal orchestration dependencies, not CUDA/torch training dependencies.

## Failure behavior

- GitHub schedule delay: next invocation reconciles the same desired state.
- Duplicate GitHub invocation: deterministic child refs and owned-kernel checks prevent duplicate submission.
- Kaggle API transient failure: current invocation fails/reports; later invocation retries without changing scientific state.
- W&B transient failure during normal clean handoff: Kaggle receipt remains sufficient for normal continuation/CPU submission; W&B is mandatory only for hard-kill recovery validation.
- HF/W&B recovery validation failure: do not recover; report blocked.
- Missing/invalid Kaggle token: skip/report that account; never switch lineage owner.
- CPU visualization partial session: later reconcile uses saved evaluation progress and a deterministic retry lineage.

## Compatibility / removal of old method

- New launches do not create `continuation-watch` kernels.
- The `continuation-watch` runner is removed from generated kernel metadata/runner selection for new source.
- Historical kernels pinned to older commits are untouched.
- Legacy controller code may remain only as reusable pure receipt/reconciliation helpers during migration; no production launch path or GitHub workflow schedules it.

## Verification

- TDD: launch no longer submits a Kaggle controller and records GitHub orchestration.
- TDD: remote GPU finalization emits an external transition receipt without submitting child jobs.
- TDD: GitHub reconciler is idempotent for active, completed, failed, continuation-existing, CPU-existing, and partial CPU cases.
- TDD: secret sync never prints secret values and uses stdin for `gh secret set`.
- Workflow YAML tests assert schedule/manual trigger, concurrency, secret references, and no `continuation-watch` launch.
- Focused cloud lifecycle suite, full CPU suite, compile/static checks, and secret scan before push.
