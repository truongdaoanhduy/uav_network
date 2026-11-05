# MASAC Actor LayerNorm Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. User explicitly requests the same approved MATD3 hidden-layer architecture for MASAC; no further design approval is needed.

**Goal:** Enable standard nn.LayerNorm in both MASAC actor hidden layers, BASE/PQL, and explain LayerNorm/ReLU ranges accurately.
**Architecture:** Reuse the actor encoder normalization path; masac.actor_layer_norm=true in the current reference profile. Both algorithms default false for historical configurations. Preserve SAC Gaussian mean/log_std/categorical heads, entropy/losses, critics and the existing MATD3 behavior.
**Tech Stack:** Existing PyTorch/pytest and bounded CPU Warp training.
**Spec:** User's direct request to apply the preceding MATD3 Linear-LayerNorm-ReLU design to MASAC; AGENTS.md continuity rules.

## Global Constraints
- Two hidden widths256/256, standard eps1e-5/affine=true; no new dependency.
- Keep MATD3 output_normalization=false, existing log_std limits, rewards and hyperparameters.
- Missing/false flag preserves old state-dict layout and historical checkpoint identity/hash compatibility.
- No production training launch; short local smoke does not prove convergence.

## Review Focus
- Shared/independent actors, BASE/PQL collector/policy copies receive the same flag.
- Amplified observations yield finite policy/log probabilities and nonzero gradients through the encoder and all MASAC heads.
- New policy/evaluation snapshots round-trip; old missing-field snapshots retain exact old actions.
- Historical best-state hashes work both before and after the MATD3-only flag existed; changed active architecture is rejected.
- Critics stay unnormalized; MATD3 retains its preceding behavior.

### Task 1: Extend and verify actor normalization
**Files:** algorithms/models.py, algorithms/masac.py, configuration.py, experiments/best_checkpoints.py, current reference XYZ YAML, tests/test_actor_layer_norm.py, source/provenance/current-task records.
**Interfaces:** Optional actor_layer_norm config; existing actor sampling and checkpoint formats with serialized config.
- [x] Extend actor normalization tests for MASAC and historical config schemas; observe missing-feature RED.
- [x] Reuse standard LayerNorm encoder construction, enable MASAC reference profile and preserve historical loading/hashes.
- [x] Run focused gradient/checkpoint/MASAC/MATD3 checks, then bounded BASE/PQL CPU smoke.
- [x] Review whole change, run full pytest, inspect failures and record evidence.
- [ ] Commit/push verified code and continuity record on the current working branch.

## Rulings and evidence
- Existing linked worktree is clean at dc698ad6e0f73a89694a12819b91b094baa675d0; reuse it.
- User explicitly requests the same MATD3 design for MASAC. Implement the stated architecture without another selection/approval question.
- LayerNorm standardizes features; it does not clip to[-1,1]. ReLU outputs nonnegative features; later signed Linear weights permit signed actions. Hidden normalization does not bound final heads.

- RED:22 failed/20 passed before implementation; all42 actor-normalization checks passed after implementation.
- Actual pre-change MASAC policy reloads with actor_layer_norm=false and exactly identical deterministic continuous/destination actions.
- Reused existing mlp builder for optional normalization; actor flag moved to shared OffPolicyConfig without changing the MATD3 serialized key or disabled module names.

- Focused GREEN:101 passed,556 subtests in23.23s. CPU smoke BASE/PQL each4 episodes,48 transitions,21 learner updates; finite nonzero actor gradients. Full scenario6 UAV/20 targets/15 obstacles/256x256 retained with reduced2env/horizon12/batch8.

- Fresh read-only whole-change review:0 Critical/Important/Minor findings. Declined GPU/convergence/long-run saturation judgment; accepted as stated verification limits. Actor construction/copy paths reviewed; real BASE/PQL orchestration confirmed in CPU smoke.

- First full suite:1286 passed,1 failed,103 skipped,1624 subtests. Failure test_pre_xyz_xy_selected_winner_restores_without_relabeling reproduced alone: its synthesized pre-XYZ fixture retained the new MASAC flag in historical identity. Hash proof shows the config hash was already accepted and identity matches when postdated flags are omitted. Corrected only the fixture;44 actor/legacy checks pass. Production validation was not weakened. Full green rerun required.

- Final full suite:1287 passed,103 skipped,8 warnings,1624 subtests in332.67s; JUnit0 failures/0 errors. No source/test changes during that run. Diff and500-line limits checked. Next: commit/push verified source and record actual commit.
