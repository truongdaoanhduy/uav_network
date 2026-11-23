# MATD3 Actor LayerNorm Implementation Plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. The user approved the two-hidden-layer LayerNorm proposal in the conversation; continue implementation without another approval gate.

**Goal:** Add standard nn.LayerNorm before ReLU in both MATD3 actor hidden layers, for BASE/PQL, while retaining output_normalization=false.

**Architecture:** Add matd3.actor_layer_norm (default false for legacy checkpoints/configs). Enable it in the current reference XYZ profile; construct Linear→LayerNorm→ReLU hidden blocks only for DeterministicActor. Target/inference/PQL copies inherit the same config. Keep MASAC, critics, objective, rewards, safety and hyperparameters unchanged. This is a declared MATD3 actor architecture variant, not exact upstream architecture.

**Tech Stack:** Existing PyTorch nn.LayerNorm, pytest, existing CPU Warp smoke infrastructure.

**Spec:** Approved conversation proposal plus AGENTS.md and current-task record.

## Global Constraints
- Two configured hidden widths [256,256], standard eps=1e-5 and elementwise_affine=true.
- No normalization on XYZ/RF outputs or categorical head.
- Preserve old state-dict layout when actor_layer_norm=false; old missing-field policies/evaluation snapshots load as false.
- Config must serialize the true flag so new policy/snapshot loading rebuilds the architecture.
- No production Kaggle jobs or unrelated algorithm tuning.

## Review Focus
- Both shared/independent actors and target actors contain the configured blocks.
- Hidden feature amplification does not silently bypass LayerNorm; actor gradients remain finite/nonzero in the bounded test.
- New policy and evaluation-snapshot round trips preserve actions.
- Missing-field legacy checkpoint inference preserves actions and no norm layers.
- BASE/PQL use the same flag; canonical profile declares the architecture choice.

### Task 1: Integrate the approved architecture
**Files:** configuration.py, algorithms/models.py, algorithms/matd3.py, current reference YAML, tests/test_matd3_layer_norm.py, provenance/current-task docs.

- [x] Add focused profile/gradient/checkpoint tests and observe RED for the missing architecture.
- [x] Add optional actor_layer_norm config and standard hidden LayerNorm blocks; enable the current profile.
- [x] Observe GREEN; run focused existing MATD3, shared and checkpoint checks.
- [x] Run bounded actual XYZ BASE/PQL CPU smoke, keeping seed44 and recording the reduced smoke budget.
- [x] Run the full repository pytest suite; inspect every failure.
- [x] Perform a fresh whole-change review, address material findings and update provenance/current-task.
- [x] Commit and push the verified change on the current working branch.

## Decisions and evidence
- Existing linked worktree verified; no new worktree needed.
- Authorization covers the architecture described in the preceding response; no additional design/plan approval is required.

- RED observed: all 10 new checks failed for missing flag/layers/serialization/defaults.
- GREEN: 34 focused checks passed; old no-norm state-dict layout remains unchanged.

- Fresh review identified one Important issue: legacy best-state hashes changed when false was added to config. Eight historical MATD3/MASAC × BASE/PQL × schema cases observed RED. Preserve missing/false identity equivalence and legacy config hashes; keep true architecture mismatch rejection.
- Review declined to infer full convergence from bounded smoke: accepted as an explicit verification limitation, not a scope change.

- Compatibility fix GREEN: 64 focused checks passed, including eight old-best-state cases.
- Actual old W&B checkpoints db61bd0c/d51b0350 reload with no norms and identical seed1000000044 XYZ actions (atol1e-7).
- Full-suite pass 1 exposed required source guard maintenance: configuration.py exceeded the 500-line limit by one; two source SHA pins correctly noticed the approved architecture edit. Condensed one existing comment and repinned only reviewed MATD3/models source.
- That run also hit the existing async-evaluation drain timing case test_real_training_selects_validated_snapshot_and_carries_it_in_full_resume[masac-False]: worker cancelled at bounded_shutdown with empty worker log under concurrent probe load. It passes in isolation (7.43s). No unrelated production timing changes; rerun full suite without competing probes.

- BASE/PQL CPU smoke complete: 4 episodes/48 transitions/21 critic updates each; min reset XYZ tanh derivative0.0351/0.0334, actor norm0.0697/0.0210. These are smoke metrics, not study performance.
- Fresh review has no remaining material finding after the compatibility RED→GREEN fix. Final complete-suite rerun passed without competing probes.

- Serial full run: 1262 passed,103 skipped,1 failure in test_api_kernel_code_contains_all_runtime_source_without_remote_checkout. Reproduced its existing compressed-byte equality failure deterministically by building gzip at times100 and101. Production bundle content was unchanged. Fixed the test to decode the actual embedded archive and compare every source member byte-for-byte; retained its checksum/credential/exclusion guards. No packaging production change.

- Final full suite: **1263 passed, 103 skipped, 8 warnings, 1623 subtests passed in316.39s**. No source/test changes during this run. `git diff --check` passes. Source commit `4a5aa227920eea08410d212ff92a8a935d42c0f3` pushed to the current branch; full seed44 learning validation remains a separate experiment.
