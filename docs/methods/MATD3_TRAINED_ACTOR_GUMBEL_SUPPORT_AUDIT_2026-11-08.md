# MATD3 learned Actor Gumbel-soft vs hard-onehot diagnostic — 2026-11-08

## Decision

**Keep Shared Actor and Shared Critic. Keep Actor LayerNorm on; no isolated evidence
proves LayerNorm itself helps Shared models. Keep Critic LayerNorm off (not tested).**
For the existing 35-target/25-obstacle environment, MATD3 Shared H512 U8
is the primary strongest validated checkpoint, and Shared H256 U8 is the
faster alternative. MASAC Shared H256 U8 is the completed baseline.

**Do not change MATD3 Gumbel-Softmax hard=False based on this evidence.**
The available best checkpoints contain *only learned Actor weights*, not
learned Critic/twin Q. Direct trained-Q(soft)-vs-Q(hard) cannot be measured
from these files, and randomly initialized Critic outputs are scientifically
irrelevant. The W&B logged model artifacts point to the same immutable
Hugging Face Actor-only best checkpoint.

## What was actually executed

- Pinned W&B cohort: [full ten-run audit](WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md)
  (see path docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md).
- MATD3 best checkpoint cases 01 (H512) and 05 (H256), all parameters Shared,
  Actor LayerNorm enabled, U8, seed44. Model snapshots downloaded from
  their exact Hugging Face Git revisions and verified against SHA-256 recorded
  in the W&B experiment evidence. No training/optimization was run.
- Environment: original checkpoint-configured `warp_packet` through
  `WarpMissionEnv` on CPU. Deterministic learned Actor actions control
  original environment dynamics; no reward, sensing, CBF, network or
  hyperparameter changes.
- Seed bank: 1000000044, 1000000045, 1000000046, 1000000047, 1000000048.
- At every visited observation, sample **one** Gumbel-Softmax destination
  (`tau=1.0, hard=False`) from learned Actor probabilities, then compare
  it with **one-hot of that same sample's argmax**, not with deterministic
  argmax of probability logits. The resulting action-support difference
  isolates categorical representation, not exploration-mode mismatch.
- Eligibility: destination mask exposes at least 2 legal options including
  silent. A forced-silent action necessarily equals a one-hot vector.

## Measured numbers (5 fixed-seed on-policy Warp rollouts)

| Metric | MATD3 01 H512 U8 | MATD3 05 H256 U8 |
|---|---:|---:|
| UAV observations | 32,484 | 32,544 |
| Observations with K>1 valid destination | 338 | 348 |
| K>1 fraction (%) | 1.041% | 1.069% |
| K=1 fraction (%) | 98.959% | 98.931% |
| Soft/hard categorical L1, eligible mean | 0.037241 | 0.010765 |
| Soft/hard L1, eligible p90 | 0.083948 | 0.000064 |
| Soft/hard L1, eligible maximum | 0.860582 | 0.783153 |
| Soft/hard L1, all-agent-states mean | 0.000387 | 0.000115 |
| Actor maximum categorical probability, eligible mean | 0.982828 | 0.992588 |
| Soft sample mass on selected hard category, eligible mean | 0.981380 | 0.994618 |

The K>1 proportion in **these Actor-controlled evaluation trajectories**
is only ~1%. This does not prove a network mask bug: communication candidates
are allowed only when a UAV has a valid, admissible report and a feasible
link. Among K>1 samples the mean soft-hard L1 is small, but some action
vectors have large discrepancies (see maxima), so claiming zero mismatch
for trained actors would be false.

## Distinguish available evidence from impossible inference

- **Established:** the shared configurations outperformed matched independent
  configurations in the saved single-seed ten-run study. This does **not**
  isolate sharing of the Actor from sharing of the Critic.
- **Not established:** LayerNorm benefit for Shared networks. The matched
  LN vs NoLN experiment was run only with Independent H512 U16 models;
  do not generalize it as the isolated effect for the preferred Shared U8 models.
- **Established:** learned Actor output on these validation trajectories
  is frequently near a categorical vertex when more than one action is legal.
- **Not established:** trained twin-Q prediction discrepancy, Bellman-target
  bias or performance change from switching soft Gumbel to straight-through
  hard Gumbel. The best checkpoint format explicitly omits Critic and replay.
- **Limitation:** only five fixed validation seeds and one Gumbel sample per
  visited observation, with trained deterministic on-policy trajectories.
  Sampling replay off-policy during learning could expose different masks
  and less-saturated stochastic Actors.

## Next action

1. **No repeat of the ten prior Kaggle jobs.** Reuse W&B report/run IDs
   and existing best Actor checkpoints.
2. Before requesting a Q-discrepancy measurement, locate a historical
   **full learner checkpoint with trained twin Critics** (not just the
   evaluation-only Actor `checkpoint.pt`) from a complete training-state
   archive, if such an archive exists. Do not reconstruct trained Critics
   from fresh initialization. If none was retained, Q-error is unknowable
   from these ten original runs. Document the limitation rather than train
   again merely to fill this audit gap.
3. If new scientifically justified experiments are launched **for a different
   question**, optionally save a read-only critic snapshot alongside best
   Actor for later numerical diagnosis; do not modify network objectives.
4. Next highest-value **analysis-only** task: inspect eligible communication
   opportunities and delivery bottlenecks from already-recorded W&B
   diagnostics; check buffer occupancy, packet attempts, receiver admission,
   RF links, battery and CBF rather than modifying Gumbel speculatively.

## Reproduction

- Offline per-run original W&B evidence:
  `docs/methods/evidence/wandb_latest_ten_2026-11-08.json`.
- Reproducible no-training script:
  `PYTHONPATH=. python scripts/diagnose_matd3_gumbel_support.py --cases 1 5 --seeds 1000000044 1000000045 1000000046 1000000047 1000000048 --max-steps 1500`
- Script requires the exact verified actor checkpoints already saved in
  `outputs/wandb-ten-20261108/diagnostics/checkpoints/` on the user laptop;
  these private checkpoint files are deliberately not committed.
- Result JSON:
  `docs/methods/evidence/matd3_trained_actor_gumbel_support_2026-11-08.json`.
- Script/test do not contact or modify W&B, Hugging Face, GitHub,
  checkpoints, or Kaggle.
