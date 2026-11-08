# UAV training continuity

# Already-run 35T25O MASAC/MATD3 BASE matrix — DO NOT REPEAT

On 2026-11-08, 10 latest W&B runs were read live and pinned in
[the 10-run comparison](docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md),
[full evidence](docs/methods/evidence/wandb_latest_ten_2026-11-08.json),
and [case reuse registry](docs/methods/evidence/wandb_10_case_reuse_registry_2026-11-08.json).
All source commits in W&B are `a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`.
Five runs finished 20,000 episodes and five are partial. All 10
best checkpoint Git revisions and file entries were verified in the
[HF availability record](docs/methods/evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json).

**Never recommend restarting a duplicate of these ten cases just to
compare them again, even in a new chat.** Compare at matched episode
budgets and matched checkpoint/global transitions. Only propose a new
cohort for a scientifically material difference in code/objective,
scenario, observation/action, reward/CBF, network, hyperparameters,
seed or evaluation protocol. A comment-only code commit is not such
a difference. For partial runs, do not claim full 20k convergence;
resume/extend only on explicit new research need with verified checkpoint.
The full per-episode raw snapshots are also preserved locally in
`outputs/wandb-ten-20261108`, with a dated backup in `~/Downloads`.

When resuming the W&B/Kaggle training audit, CPU replay work, or a previous
session, read [CURRENT_TASK.md](docs/superpowers/CURRENT_TASK.md) before editing
code or launching a job. It contains the current task, verified evidence,
next action, and continuation rules. Read its linked task brief only when
working on CPU replay.

Update the current-task record with the actual commit, verification result,
and next action before handing off or finishing a task. Commit and push
the record on the working branch so a fresh chat can retrieve it.
Keep timestamps on observations from running services; recheck live state
before making operational decisions.
