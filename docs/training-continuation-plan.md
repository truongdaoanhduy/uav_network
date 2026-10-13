# Training continuation implementation plan

> For agentic workers: use superpowers:executing-plans for native execution,
> RED/GREEN checks and one fresh whole-branch review.

Goal: finish the 1,000-episode experiments from recovered checkpoints and retain
online checkpoints before Kaggle's hard time ceiling.

Architecture: extend the existing training loop, v2 checkpoint, online tracker
and Kaggle runner. Restore model/target tensors and update counters; rebuild Adam
and replay, documenting this as policy warm-start. A soft deadline finishes a
segment normally and uploads its model; completed-episode budget is reported
separately from segment completion.

Tech stack: existing Python/PyTorch, Gymnasium/UavNetSim and W&B/Kaggle SDKs.
Spec: the bounded design in the current conversation; user authorized continuing
the pending training, local fixes, GPU jobs and GitHub synchronization.

## Constraints and review focus

- Six UAVs, fixed 2D/FOV/native networking/energy and original learning equations.
- Every owned file <500 lines; generated outputs/dependencies/keys stay ignored.
- New W&B run IDs preserve old crashed runs; progress and parent provenance are real.
- Legacy checkpoint without counters must fail rather than invent its progress.
- Changed scenario/model settings must fail before sampling or tracker startup.
- Empty completed history at a deadline must not invent episode aggregates.
- Restored update/exploration counters persist, but refill is based on new samples.
- Periodic upload finishes before its checkpoint is considered available online.

## Task 1: Safe warm-start and soft deadline

Files: experiments/continuation.py; algorithms/checkpoints.py; configuration.py;
experiments/training.py and cli.py; tests/test_continuation.py.

- [ ] RED: restore a real small checkpoint and compare all weights/counters;
      reject missing progress and changed world, using actual model files.
- [ ] RED: real short training continues cumulative episodes/seeds and keeps
      prior records; Adam/replay are explicitly new, target/update state retained.
- [ ] RED: a tiny soft deadline saves a finite checkpoint with truthful incomplete
      budget and zero completed episodes, instead of raising on final metrics.
- [ ] Implement warm_start_checkpoint and max_train_seconds config/CLI, validate
      progress and write it atomically with policies; drain final collection updates.
- [ ] GREEN: targeted tests and full native suite; commit.

## Task 2: Durable online checkpoints and cloud inputs

Files: experiments/tracking.py; cloud/cli.py, remote.py and kernel.py;
configs/kaggle.yaml; tests/test_tracking.py, test_cloud.py, test_cuda.py.

- [ ] RED: periodic checkpoint upload yields a committed artifact while the run
      stays active; empty segment metrics remain absent.
- [ ] RED: per-algorithm artifact inputs become learner checkpoint paths and
      requested account quota checks avoid unrelated accounts.
- [ ] Implement log_artifact + bounded upload wait, checkpoint input downloads,
      parent metadata and --warm-start-artifact mapping; use a 10-hour soft budget.
- [ ] GREEN: CPU suite and real CUDA continuation + online artifact smoke; commit.

## Task 3: Recovery, evaluation and restart

- [ ] Retrieve only checkpoint 500/940 and original episode JSONL; keep originals.
- [ ] Annotate copied policies with verified matching episode/counter/history
      provenance; publish them as inputs through a separate checkpoint-recovery run.
- [ ] Evaluate both recovered policies on matching CPU seeds and save previews.
- [ ] Fresh review; fix Important/Critical findings with RED/GREEN evidence.
- [ ] Publish exact source and launch continuation to 1,000 total episodes,
      pin its two real online runs; verify weights/update/refill/artifact behavior.
- [ ] Sync main and report actual status and warm-start limits.

## Execution ledger

2026-10-13: source 022a63f. Kaggle cancelled after its 11-hour ceiling. Recovered
MASAC 500 / global_step 399360 / updates 197680, MATD3 940 / global_step 728026 /
updates 362013. Latest logs contain 504/946 completed episodes; later work than
the latest saved policies is excluded from the continuation prefix.
Ruling: use an isolated branch in the user-requested folder, preserving its
installed runtime and downloaded checkpoints; no duplicate project/worktree.
Ruling: policy warm-start is required by the available v2 data; claiming full
Adam/replay or trajectory resume would be false. New runs are labeled accordingly.
