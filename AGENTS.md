# 2026-11-08 MASAC correctness and evaluation reproducibility

For scientific/reviewer claims about MASAC lower than MATD3, first
read `docs/methods/MASAC_REVIEWER_BASELINE_CORRECTNESS_AUDIT_2026-11-08.md`.
Original hybrid SAC loss is source-checked; NO critical proven
formula bug. Original Fu Hybrid-MASAC uses different network
architecture and must not be claimed byte-identical.
The same MASAC actor check shows early Gaussian movement
sampling accounts for most of sampled-vs-mean performance delta.
W&B reveals much higher CBF corrections in MASAC at matched
early checkpoints (0.93/0.98 per UAV step versus MATD3 0.61/0.80).
**Original W&B best validation != local replay** even with same
checkpoint SHA/seeds/scene layout hashes: original MASAC H256
52.00%, local 57.71%. This is UNRESOLVED; do not claim
bitwise full reproducibility, do not mix evaluation provenance,
and do not auto launch training for this audit.
The documented next task is a no-training, first-divergence
evaluator runtime trace; no redundant 10-run experiments.

# Explicit paper-core baseline vs previously trained shared variants

User requires **source-faithful MASAC/MATD3 baseline** rather than just
a well-performing parameter-sharing architecture. Read
[the detailed code/paper audit](docs/methods/PAPER_CORE_VS_EMPIRICAL_BASELINE_35T25O_2026-11-08.md)
first. Current 35T25O strict profile:
`configs/comparison-paper-core-35t25o-xyz-sensing.yaml`.
It uses independent Actor and Critic modules for each UAV, no Actor
LayerNorm, no PQL, U8, H512, mask-aware SAC dual temperature adapter.
Preflight: `PYTHONPATH=. python scripts/verify_paper_core_baseline.py`.
The profile has **NOT** been trained at 20k as a strict matching pair;
the ten historical W&B runs are not interchangeable with it.
For manual Kaggle UI, use
`scripts/build_paper_core_35t25o_ui_notebooks.py` **only after commit**
to create two pinned private notebooks. Do not auto-launch or suggest
the old ten cases need repetition.

Never relabel the best trained Shared+ActorLayerNorm runs as an
exact per-agent author-paper reproduction. The original source
equations already passed documented unit tests; do not arbitrarily
rewrite mathematical loss terms. Source auditing found no proven
critical formula violation requiring silent mutation. Distinguish
MATD3 action-space Gumbel and MASAC mask-aware entropy as declared
UAV-specific adapters, not original single-agent SAC/TD3 formulas.

# UAV training continuity

# Already-run 35T25O MASAC/MATD3 BASE matrix — DO NOT REPEAT

**Gumbel/critic follow-up already executed (2026-11-08):**
[trained Actor diagnostic](docs/methods/MATD3_TRAINED_ACTOR_GUMBEL_SUPPORT_AUDIT_2026-11-08.md)
using 5 fixed validation seeds and both best MATD3 Shared U8 checkpoints.
The original Hugging Face best files are **evaluation-only Actor snapshots**;
they do NOT include learned Critics. Do not claim a numerical trained Q-gap
was measured; do not replace Gumbel `hard=False` with `hard=True` based
on these outputs. Do not train a duplicate run simply to recover
Critic values missing from saved snapshots. Use the existing registry first.

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
