# MASAC Valid-Destination Exact-Q Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce MASAC exact-Q compute by evaluating only destinations admitted by the existing observation mask, while preserving SAC losses/gradients on every probability-bearing action and benchmarking the change on the production-shaped 2×T4 profile.

**Architecture:** Reuse `valid_destinations()` as the single source of truth. Compact `(sample, destination)` rows before critic evaluation, scatter valid Q values back into the existing `[batch, destinations]` interface with zero in probability-zero slots, and keep chunking as a fallback only. Compile the compact critic-evaluation hot path, not a data-dependent Python control-flow wrapper.

**Tech Stack:** Python 3.13, PyTorch 2.11/CUDA 12.8, pytest, Kaggle 2×Tesla T4 benchmark pipeline.

**Spec:** User request in the 2026-11-03 conversation: performance-only MASAC; no change to SAC equations, entropy, reward, scenario, batch size, or valid-destination semantics.

## Global Constraints

- Keep MASAC SAC target, actor loss, alpha loss, reward, entropy targets, and optimizer semantics unchanged.
- Invalid destinations are exactly those rejected by `valid_destinations(observation, destinations)` and already have policy probability zero.
- Do not reduce logical batch size 8192, environment count 4096, replay capacity 1,000,000, or 20k episode budget.
- No new dependency or speculative abstraction.
- Keep `critic_chunk_size` implementation available as a memory fallback; canonical hybrid config defaults to zero after measured headroom verification.
- Before/after GPU benchmarks must use separate Kaggle accounts and the same production-shaped config.

## Review Focus

- A sample with only destination 0 valid must evaluate exactly one critic alternative and retain the same weighted soft value/loss.
- Mixed masks across samples and agents must scatter Q values to the correct destination slot without cross-sample or cross-agent mixing.
- Destination 0 must preserve the existing rule that the third continuous control is zero.
- Actor gradients through valid continuous actions must match dense exact-Q within numerical tolerance; invalid slots must contribute zero gradient because their policy probability is zero.
- Compiled execution must run without graph-break/fallback regressions on the compact critic hot path.

---

### Task 1: Valid-only exact-Q semantics

**Files:**
- Modify: `uav_network/algorithms/masac.py`
- Test: `tests/test_masac_valid_exact_q.py`

**Interfaces:**
- Consumes: `valid_destinations(observation, destinations) -> bool mask`.
- Produces: `_expected_q(..., valid_mask)` and `_expected_q_all(..., valid_masks)` with the same Q tensor shapes used by current losses.

- [ ] Write failing tests comparing dense reference Q on valid slots and weighted soft/actor objectives across sparse mixed masks, plus critic-row-count assertions proving invalid destinations are never evaluated.
- [ ] Run the focused tests and verify RED for missing valid-mask behavior.
- [ ] Implement the smallest compact gather/evaluate/scatter path; fill invalid Q slots with zero and preserve destination-0 continuous canonicalization.
- [ ] Pass valid masks from `next_observation` in critic targets and `observation` in actor updates.
- [ ] Run focused tests and `tests/test_critic_chunks.py` to verify GREEN and unchanged fallback semantics.

### Task 2: Compile the compact exact-Q critic hot path

**Files:**
- Modify: `uav_network/algorithms/optimization.py`
- Modify: `uav_network/algorithms/masac.py`
- Test: `tests/test_masac_valid_exact_q.py`
- Test: relevant compile tests discovered in `tests/`

**Interfaces:**
- Produces a compiled callable for the dense neural critic evaluation over already-compacted valid rows; compaction itself remains eager/data-dependent.

- [ ] Write a failing test that `compile_models(MASAC)` installs/uses the exact-Q compiled callable while preserving state-dict keys and numerical outputs.
- [ ] Verify RED.
- [ ] Compile only the compact critic-evaluation function with `dynamic=True, backend="inductor"`; retain existing actor encoder compilation.
- [ ] Verify focused compile tests GREEN and no change to checkpoint/state-dict contract.

### Task 3: Verification and matched A/B benchmark

**Files:**
- Modify only evidence/plan docs if needed; no scientific config tuning.

**Interfaces:**
- Baseline: current dense exact-Q benchmark evidence with `critic_chunk_size=0`.
- Candidate: valid-only exact-Q + compact hot-path compile.

- [ ] Run the complete local pytest suite and report every failure if any.
- [ ] Run CPU numerical-equivalence checks for Q, loss, gradient, RNG, optimizer state/update count.
- [ ] Commit/push the candidate source so Kaggle uses a pinned clean commit.
- [ ] Launch candidate production-shaped MASAC benchmark on a separate free Kaggle account; capture peak allocated/reserved VRAM, learner updates/s, compile graph evidence, and OOM status.
- [ ] Compare candidate against the fresh dense `chunk=0` baseline; accept only if numerical gates pass and compute/throughput improves without memory or compile regressions.
- [ ] Do not change entropy/scenario based on this microbenchmark; those decisions require the subsequent 20k cohort.
