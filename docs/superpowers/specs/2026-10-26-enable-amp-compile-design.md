# Enable AMP + torch.compile for the 1 KiB production profile

## Intent
Enable the already-implemented production AMP FP16 and `torch.compile` paths for future runs using `configs/comparison-20k-hybrid75-1kb.yaml`, without changing MASAC/MATD3 mathematics or any currently running G3 lineage.

## Goals
- Set `runtime.amp_mode: fp16` and `runtime.compile_models: true` in the 1 KiB profile.
- Keep model parameters, optimizer state and replay storage FP32 as implemented today.
- Keep the current Inductor/default compile implementation; do not copy BF16 or `reduce-overhead` from other repositories.
- Make W&B grouping/tags distinguish accelerated future runs.
- Preserve exact-resume guards so off/off checkpoints cannot continue as on/on.

## Non-goals / anti-goals
- No change to env count, batch, replay capacity/layout, reward, network, scenario, seed, episode budget, learner architecture or fused Adam.
- No launch/restart/cancel of the eight existing G3 jobs.
- No claim of speedup, lower VRAM, or equal learning quality until a matched CUDA benchmark measures it.

## Constraints
- Existing G3 jobs are pinned to source `77a1d0f`; this branch change affects only future jobs that use the new commit.
- `amp_mode` and `compile_models` are already part of fixed/PQL resume compatibility checks.
- Existing source uses FP16 autocast + GradScaler and default Inductor compilation with dynamic shapes; this change activates that path rather than rewriting it.

## Approaches considered
1. Rewrite AMP/compile in MASAC/MATD3. Rejected: the code already exists and is covered by production execution tests; duplicating it adds risk.
2. Keep the profile off and rely on launch-time `--set` overrides. Rejected for this request: easy to forget and weakens reproducibility/provenance.
3. Enable the two existing flags in the canonical 1 KiB profile and mark the W&B group/tags. **Chosen**: smallest durable change, with historical jobs protected by source pinning and resume guards.

## Verification
Test-first assert the profile enables FP16 AMP + compile and carries acceleration provenance. Then run the existing AMP/compile production execution, precision, replay and resume tests, followed by the full offline CPU suite. Real T4 benchmarking is separate from this code/config activation.
