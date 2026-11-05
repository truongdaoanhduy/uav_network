# Matched shared chunk2048 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Give both learners real bounded critic execution and one explicit shared profile.
**Architecture:** Reuse non-reentrant checkpointing around pure twin-critic calls.
Slice original replay rows, including axis1 for per-critic actions, before expansion.
Compute complete logical losses and step each optimizer once.
**Tech Stack:** PyTorch, pytest, YAML, existing CPU BASE/PQL runners.
**Spec:** docs/superpowers/specs/2026-11-06-matched-critic-execution.md

## Global constraints

Preserve8192 logical batch, RNG sequence, losses, clipping and target schedules.
Set canonical chunk2048/shared/shared only; retain all other hyperparameters.
Legacy configs/checkpoints keep their existing defaults. No GPU production launch.
All owned text sources <=500lines.

## Review focus

- Uneven final chunks must receive full-batch mean weighting.
- Per-critic joint actions have replay batch on axis1, not axis0.
- Checkpoint backward must not repeat policy/noise sampling or optimizer steps.
- Frozen critics must still propagate gradients to actor inputs.
- PQL config/copies and legacy checkpoint loading must preserve the chosen schema.

### Task1: Matched bounded execution and complete remaining config audit

**Files:** algorithms/critic_chunks.py,masac.py,matd3.py; canonical XYZ YAML;
tests/test_matched_critic_chunks.py, reference contract/source guard;
provenance/current-task; new config-difference report/evidence.

**Interfaces consumed:** chunked_call(function,state,joint,continuous,chunk_size);
critic_values(critics,state,actions,vectorized=False,per_critic=False).
**Interface produced:** chunked_critic_values(function,state,joint,chunk_size,
per_critic=False) -> (q1,q2), concatenated on original batch axis0.

- [x] Write real learner tests for bounded target/regression/actor calls across
  two algorithms, sharing and vectorization modes; nondivisible B11/chunk4.
  Assert config construction yields one actor/critic for each algorithm.
- [x] Run new tests before edits. Expected: regression/TD3 bound failures and
  canonical shared construction failure, proving the missing behavior.
- [x] Add pure twin-output checkpoint helper; integrate all remaining critic
  paths. Leave policy generation outside helper; retain full-batch reductions.
- [x] Set canonical2048/shared flags; update explicit reference contract and
  provenance to declare this authorized shared execution variant.
- [x] Verify values/gradients/RNG/optimizer state vs chunk0 over3updates,
  including clipping active and TD3 delay; retained activations decrease.
  Expected: focused pytest passes for shared/independent,vectorized/nonvectorized.
- [x] Validate actual canonical2048 configs for MASAC/MATD3 BASE/PQL; bounded
  CPU integration exercises real collectors/learners with small effective chunks.
- [x] Generate exhaustive old/resolved/current JSON and human table from four
  W&B configs; exclude chunk in remaining-differences table, annotate new inactive
  fields and algorithm groups not consumed by the selected learner.
- [x] Review intentional learner diff and repin only changed source digests.
- [x] Run bare pytest without competing heavy probes. Expected: all checks pass,
  skips reported; run git diff --check and text-line cap.
- [x] Commit; obtain one fresh-context whole-change review, address any material
  finding with RED->GREEN and suite; update CURRENT_TASK with actual source SHA.
- [x] Push working branch only; publish reviewable report. No training launch.

## Execution ledger

Current worktree is already isolated on experiment/hfbest-embedded-source-steps1500-20261105.
The user's explicit edit instruction authorizes inline implementation without
a further approval gate. Historical proposal files retain their1024 names/values;
the canonical profile is the sole selected current recipe.
