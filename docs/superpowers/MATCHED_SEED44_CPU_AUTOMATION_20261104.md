# Seed-44 training completion -> CPU evaluation/visualization (GitHub Actions)

## Evidence from 2026-11-04 (read-only inspection)

- All ten original W&B training runs in group `matched-architecture-seed44-20261104` have state `crashed`; **none completed 20,000 episodes**.
- Their last W&B timestamps cluster between 04:30:03 and 04:31:11 Vietnam time, after the laptop powered off at 03:45. The laptop did not directly stop these cloud runs.
- The runs logged real GPU learner progress and checkpoint recovery uploads. They did not supply a terminating Python traceback in `training-status.json` or the retrieved case-01 `thread-stacks.log`. The latter shows progress markers increasing to 2,355,200 close to the final event, not evidence of a confirmed training deadlock. An exact cause cannot be inferred from state `crashed` alone.
- The Kaggle CLI currently reports authentication required for checked account tokens (01, 06, 10); do not assume this is an account-ban diagnosis, and do not claim Kaggle kernel error logs were examined.
- The existing `UAV External Supervisor` workflow is disabled, and it can resume/relaunch GPU tasks. **Leave it disabled** under the user's one-session-only constraint.

## New independent workflow

The dedicated `UAV Matched CPU Visualization` workflow polls the ten pinned W&B run IDs from `configs/matched-seed44-cpu-automation.json`. It checks the case ID, W&B group, source commit and algorithm settings, then refuses CPU submission unless ALL of the following hold:

1. W&B run state is `finished`.
2. The final summary confirms `episode_budget_complete=true`, `episodes_completed=20000`, and `stop_reason=episode_budget`.
3. The same run logged an immutable model artifact (`:vN`) with matching source run ID, source commit, exactly 20,000 episodes, completed-budget metadata, and a checkpoint manifest.
4. No CPU kernel with the deterministic lineage-derived reference already exists.

Only then does GitHub Actions submit ONE Kaggle **CPU-only** visualization kernel using the account's Actions secret and previously uploaded private credential datasets. The CPU pipeline evaluates the immutable checkpoint on 30 episodes at evaluation seed 2000000044, creates an animation and final image, and publishes evaluation metrics/media to W&B and Hugging Face. The script itself never submits or resumes GPU jobs.

The scheduled workflow executes on a GitHub-hosted Ubuntu runner; it does not use the user's machine or sandbox. The explicit `workflow_dispatch` `dry_run=true` mode performs read-only W&B checks and can validate access before enabling side effects. GitHub Actions does not guarantee exact cron timing; the schedule is opportunistic.

For the **ten existing crashed runs**, the correct result is ten `blocked_failed` entries and **zero visualization submissions**. A partial checkpoint must never be silently labeled a finished 20k run.

Required GitHub Actions secrets: `WANDB_API_KEY`, `HF_TOKEN`, and `KAGGLE_API_TOKEN_ACCOUNT_01` through `KAGGLE_API_TOKEN_ACCOUNT_10`. Required variable: `HF_REPO_ID`. Existing private Kaggle W&B/HF datasets must remain accessible. If Kaggle authentication is unavailable, a future completed run cannot yet be visualized until credentials are repaired. This workflow does not fix expired/invalid tokens and does not alter running GPU kernels.

Future cohorts with new W&B IDs/commit must be deliberately added to the pinned manifest; this workflow does not assume arbitrary group members are authorized. GitHub Actions results and sanitized JSON reports are retained as workflow artifacts.
