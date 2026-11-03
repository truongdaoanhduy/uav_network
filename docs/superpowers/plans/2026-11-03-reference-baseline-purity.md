# Reference MASAC/MATD3 Baseline Purity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task with TDD.

**Goal:** Make the active XYZ BASE algorithms source-traceable baselines rather than UAV-specific tuned variants.

**Architecture:** Preserve the UAV environment/action interface as a clearly documented hybrid-action adapter. Align SAC/MASAC update semantics to Haarnoja + BenchMARL/TorchRL and hybrid entropy factorization to Delalleau Hybrid SAC; align MATD3 to Ackermann et al. + Fujimoto TD3. Runtime-only optimizations may remain only when mathematically equivalent and disabled where not verified.

**Tech Stack:** PyTorch, NumPy, pytest, YAML.

## Global Constraints
- No 20k training.
- No reward/scenario/sensing/CBF changes.
- No performance-motivated algorithm tweaks in BASE.
- `critic_chunk_size=0`, `matd3.output_normalization=false` remain.
- PQL remains a separate execution variant and reuses the same corrected algorithm core; its execution plumbing is unchanged and the reference profile defaults to BASE.
- Hybrid action/masking must be labeled an interface adaptation and backed by published/library behavior.

## Reference contract
- SAC continuous: Haarnoja 2018/2019; OpenAI Spinning Up/SB3 implementation conventions.
- MASAC CTDE/masking: BenchMARL/TorchRL MASAC.
- Hybrid factorization: Delalleau Hybrid SAC (arXiv:1912.11077), separate continuous/discrete entropy weights.
- TD3: Fujimoto 2018 original implementation.
- MATD3: Ackermann et al. 2019 paper and released implementation.

### Task 1: Lock source-backed baseline config
- Add tests asserting no entropy annealing, alpha init 1.0, discrete target entropy weight 0.2-equivalent, no chunking, no MATD3 output normalization, and canonical TD3 delay/noise values.
- Make the explicit reference YAML satisfy them. The three historical XYZ study profiles keep their recorded variant settings; chunk0/normfalse are already applied there at0baea78.

### Task 2: Align MASAC actor/temperature mechanics
- Test SAC log-std range matches reference [-20, 2].
- Remove the UAV-specific alpha hard clamp in BASE mechanics.
- Preserve separate alpha_c/alpha_d because hybrid HSAC explicitly defines them.

### Task 3: Align MATD3 actor update to released MATD3
- Test peer actions in the actor critic-input come from replay while only the optimized agent action comes from its actor.
- Use source-backed differentiable categorical sampling for the discrete branch during actor optimization; hard integer selection remains only at the environment boundary.
- Keep twin min target, target smoothing, delayed update and Polyak update unchanged.

### Task 4: Provenance and regression
- Add a baseline provenance document mapping every algorithm-core mechanism to a source.
- Run focused algorithm tests, source-contract tests, `git diff --check`, then full suite; report any pre-existing failures separately.

## Completion ledger — 2026-11-03
- Resumed the existing in-progress baseline worktree at0baea78; did not re-implement its completed config change.
- Source review: exact author MATD3 critic losses need batch-only means before per-variable clipping. Added2- and6-agent Adam oracles: observed RED, then GREEN after undoing the extra agent average in source per-parameter mode. Legacy global mode remains unchanged.
- Clipping telemetry is part of verifying the chosen baseline rule; corrected disabled/per-variable reporting with two observed RED-to-GREEN checks. Per-variable mode reports maximum tensor norm rather than global norm.
- Provenance now explicitly labels soft-Gumbel/hard-replay/argmax-RF adapters, MASAC-only enumeration and effective conditional continuous entropy targets; these are not claimed as an exact upstream reproduction.
- Archived508 historical CURRENT_TASK lines in two adjacent files; concatenation is byte-identical and relative links remain valid. Observed documentation size RED-to-GREEN.
- Focused source regression:15passed,534subtests passed. Full CPU suite pending; no training20k, live W&B mutation or cloud launch.
- Config-only verification caught two omitted safety reward coefficients in the unfinished profile (default0 instead of1.0/0.5). Restored the common scenario rewards with an observed RED-to-GREEN namespace equality check; no algorithm comparison may silently change these rewards.
- Final review fixed all identified source-clipping/telemetry/provenance issues; no deferred review findings remain. Full CPU suite1217passed103skipped1605subtests,0failures; final supplemental focused17passed534subtests. No GPU/training-performance claim.
- Task1: complete — explicit source baseline config and identical scenario/rewards verified.
- Task2: complete — SAC source bounds and unrestricted alpha with published hybrid factorization.
- Task3: complete — author MATD3 replay peers/Gumbel and independently verified clipping/Adam updates.
- Task4: complete — provenance, reviewed branch, fresh full/focused regression and history preservation.
