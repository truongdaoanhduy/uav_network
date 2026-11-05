# Matched shared critic execution — 2026-11-06

Source base: 52e500a65108e04df6f1ed0fd01dc64a51e70e4a.

User request: implement real critic chunks of2048 in MASAC and MATD3, enable
shared actors/critics in the current profile, then list every other old/current
configuration difference for all four BASE/PQL cases. Do not launch training.

- Preserve logical batch8192, sampling order, objectives, loss reductions,
  optimizer counts, gradient clipping placement, target update schedules.
- Chunk target, actor-through-critic, and critic regression paths for both
  shared and independent models; vectorized/nonvectorized paths must work.
- Split by original replay rows before agent/destination expansion.
- Checkpoint only pure critic computations; sample policies/noise beforehand.
- Canonical XYZ reference becomes shared/shared and critic_chunk_size2048.
- Preserve all other canonical hyperparameters and dataclass legacy defaults.
- Keep the two1024 proposals as dated historical proposals, not active profiles.
- Compare old W&B resolved config05beef6c against resulting canonical profile;
  exclude chunk from the remaining-differences table; mark absent/unused fields.
- Separate source changes from config changes and evidence from crash hypotheses.
- No production launch, no claim of GPU memory/convergence validation.
