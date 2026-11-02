# MASAC Power Causal Audit Design

## Intent
Determine whether the late-training MASAC TX-power collapse is caused by loss of behavioral/replay support that produces a miscalibrated critic, without changing SAC losses, rewards, entropy targets, optimizer settings, or scenario semantics.

## Goals
- Fix the XYZ/coarse-belief checkpoint observation-version validator so periodic evaluation can load version-5 checkpoints produced by the same source.
- Add an opt-in runtime-only TX-power support intervention that changes collector behavior only; default is exactly disabled.
- Add a bounded critic-calibration probe comparing critic Q ordering for low/high TX power against realized environment return from the same simulator snapshot.
- Run a short matched PQL A/B from the same 8192-episode full resume checkpoint: baseline support fraction 0 versus intervention support fraction 0.10.

## Non-goals
- Do not change SAC Bellman equations, actor loss, alpha loss, target entropy, reward coefficients, scenario, replay sampling, optimizer hyperparameters, or 20k episode budget.
- Do not claim a production algorithm improvement from this diagnostic.
- Do not merge to main during the experiment.

## Approach
1. Correct observation-version validation by reconstructing the saved Config and comparing the stored version with `observation_version(config)` rather than hard-coding XYZ version 4.
2. Apply power support only in non-deterministic MASAC collection. For a deterministic 10% subset of active, non-silent rows, cycle the learned power action through a fixed grid `[-1, -0.5, 0, 0.5, 1]`. Selection is derived from global step and flattened row index, so it consumes no actor RNG and leaves all non-selected action components unchanged.
3. Calibration probe takes exact simulator snapshots, constructs matched low/high-power joint actions while keeping movement and routing fixed, reports critic `Q_high-Q_low`, realized discounted/undiscounted return delta over a bounded horizon, delivery and PHY-success deltas, and rank agreement.
4. A/B resumes the same full PQL checkpoint at 8192 episodes and runs a bounded short segment. Success for the support-collapse hypothesis requires intervention to materially increase high-power support and move critic calibration toward the environment ordering relative to baseline.

## Constraints
- Source lineage: exact cohort commit `05beef6cd0d97d39d51b220f9f7789e915b5bc10`.
- Same seed 44, 4096 envs, batch 8192, 1M replay, CPU75/GPU25, PQL, 1500-step scenario.
- No 20k run.
- Credentials/tokens must never be logged or committed.
