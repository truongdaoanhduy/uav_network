# Shared Actor + Shared Critic for MASAC and MATD3

## Intent
Create a reproducible controlled profile that enables full parameter sharing for both MASAC and MATD3 while preserving the existing independent canonical profile for comparison.

## Goals
- MASAC: `share_actor_parameters: true`, `share_critic_parameters: true`.
- MATD3: `share_actor_parameters: true`, `share_critic_parameters: true`.
- Keep scenario/training constants identical to `configs/comparison-20k-hybrid75-1kb.yaml`: 6 UAV, 20 targets, 15 obstacles, 1200 steps, seed 44, 20k episodes, E4096, B8192, replay 1M, HYBRID75, FP32, BASE, link-budget +3 dB.
- Give the profile a distinct output directory/group/tag so W&B and artifacts cannot be confused with the independent baseline.
- Preserve the old canonical profile unchanged.

## Non-goals
- Do not change MASAC or MATD3 learning equations; sharing support already exists in the learner implementations.
- Do not enable PQL.
- Do not change replay layout, AMP mode, seed, scenario, reward, sensing, communication, or update ratio.
- Do not claim a production ranking from smoke tests.

## Approaches considered
1. Edit the existing canonical profile in place. Rejected because it destroys the clean independent baseline and makes old/new results harder to compare.
2. Add CLI overrides at every launch. Rejected because it is easy to omit one flag and weakens provenance/readability.
3. **Chosen:** create a dedicated shared-actor-critic profile copied from the canonical profile with only sharing identity/output metadata changed. This is explicit, reproducible, and preserves the baseline.

## Verification
- A test must prove the new profile enables both sharing flags for both algorithms.
- The test must prove all scientific/training fields other than the two sharing flags and identity/output metadata match the canonical profile.
- Existing parameter-sharing tests for MASAC/MATD3 must pass.
- Run a tiny CPU smoke for MASAC and MATD3 from the new profile after tests pass.
