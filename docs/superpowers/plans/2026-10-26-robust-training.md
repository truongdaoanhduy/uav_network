# Robust scenario and MASAC execution implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** prevent random generation exhaustion from terminating supported training and fit unchanged MASAC logical budgets on2T4, with clear run identities.
**Architecture:** certify deterministic fallback scenes once, use bounded PCG-equivalent reset retries on native/Warp, and chunk pure MASAC critic evaluation with activation recomputation.
**Tech Stack:** Python/NumPy/PyTorch/Warp, Kaggle, W&B.
**Spec:** docs/superpowers/specs/2026-10-26-robust-training-design.md

## Global Constraints
- Preserve active jobs, no HF/W&B deletion, no duplicated lineage.
- Keep20kepisodes/env4096/batch8192/replay1M/1KiB/1800steps/seed44 and scientific geometry/reward/network settings.
- Source/test-owned text<=500lines per file.
- Tests first; local tests HF_HUB_OFFLINE=1 WANDB_MODE=disabled.
- Generation protocol changes must reject older training semantics; historical artifacts retain their original provenance.
- User authorized implementation and rerunning corrected failed branches; do not add permission pauses.

## Review Focus
- Random layout jams after startup, not just9642: certified recovery plus full seed schedule and an extended independent cohort.
- Explicit invalid scenarios: reject instead of silently replacing.
- Masked reset: unchanged worlds, error flags, PCG state and exact-resume tensors.
- Actor gradients and RNG: chunks must preserve full logical8192 loss/step/clipping semantics.
- Cloud identity: no stale cpu75 tags on fullGPU, no train/controller/benchmark confusion.

### Task 1: Certified scenario recovery
**Files:** environment/world.py, new environment/scenario_recovery.py, environment/gpu/world.py/state.py, configuration.py, primary YAML, environment info and scenario tests.
**Interfaces:** generate_scenario(config,rng,options=None) remains; certified_scenes(config) returns immutable validated scene bank; device copies this bank once. Reset returns complete valid scene or startup configuration error.
- [ ] Write regressions for9642, a bounded seed cohort, forced exhaustion, native/Warp parity, masked reset and incompatible generation2 checkpoints.
- [ ] Run RED before production changes.
- [ ] Implement bounded full-scene retry and deterministic certified fallback; bump generation_version3 and trace fallback usage.
- [ ] Run focused GREEN and seed schedule scans; commit task.

### Task 2: MASAC memory execution
**Files:** algorithms/masac.py, new algorithms/critic_chunks.py, configuration.py, primary YAML, tests/test_critic_chunks.py and algorithm source contract.
**Interfaces:** runtime.critic_chunk_size nonnegative,0 keeps unchunked reference; primary1024. chunked_call(function,state,joint,continuous,chunk_size) preserves outputs/autograd, checkpointing differentiable chunks.
- [ ] Write RED output/gradient/RNG/optimizer-step regressions against the reference including uneven chunks.
- [ ] Implement only critic execution chunking, leaving sampling/logical loss/optimizer budget intact.
- [ ] Focused GREEN then real2T4 BASE/PQL production-shape critic+actor updates, peak VRAM and exact resume; commit.

### Task 3: Recognizable identities and failure reporting
**Files:** cloud/cli.py/controller.py/continuation.py and naming helper; cloud identity tests.
**Interfaces:** run names encode stage/algo/mode/layout/E/B/R/generation; controller diagnostic retains actionable parent failure where available.
- [ ] Write RED semantic identity tests for hybrid/FULLGPU, BASE/PQL and continuation.
- [ ] Implement shared concise naming, authoritative tags and actionable blocked reason; preserve no-blind-retry guard.
- [ ] Focused GREEN; commit.

### Task 4: Verify and launch corrected failed cases
**Files:** diagnostic runner/evidence/receipts, CURRENT_TASK.md.
- [ ] Run full CPU suite, inspect every failure and full seed schedule results.
- [ ] Publish pinned source, run real2T4 CUDA regressions/full suite/memory probe on idle account.
- [ ] Obtain whole-change independent review; resolve material findings with RED/GREEN.
- [ ] Fresh-reconcile remote jobs/accounts and launch only corrected requested cases from episode0 with unambiguous names and auto-continue/CPU30eval.
- [ ] Verify actual W&B progress, hardware/controller heartbeat and record evidence/next action; no universal bug-free claim.
