# W&B handoff and hybrid replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox syntax.

**Goal:** Correct training visibility and enable the requested explicit 1M 75/25 replay configuration.
**Architecture:** Reuse HybridRing behind one production replay factory; preserve default layout. Keep continuation outcomes separate from training completion.
**Tech Stack:** Python, PyTorch, existing Kaggle CLI, wandb.
**Spec:** docs/superpowers/specs/2026-10-23-training-handoff-replay-design.md

## Global Constraints
- Preserve active production jobs, scientific baseline, checkpoint identity/RNG and env state.
- replay=1,000,000; CPU fraction=.75; env=4096; batch=8192 for the separate requested config.
- Never print credentials or persist arbitrary secret-bearing exception strings.

## Review Focus
- Cloud overrides must not silently force CUDA or change requested replay size.
- Partial/full/wrapped hybrid rings retain identical next samples after restore.
- Old pure CPU/CUDA checkpoints remain usable under their original configuration.
- W&B session finished must not imply final evaluation or continuation succeeded.
- No duplicate submission following timeout; no production launch without hardware evidence.

### Task 1: Safe continuation outcome reporting
Files: cloud/continuation.py, cloud/remote.py, tests/test_cloud_continuation.py and one focused reporting test.
- [x] Write RED case for quota failure reason retained without raw exception message.
- [x] Implement safe failure code/phase and W&B pipeline status after training shutdown.
- [x] Focused GREEN regressions; included in verified fix commit.

### Task 2: Production hybrid replay and cloud placement
Files: algorithms/replay.py or dedicated hybrid module, experiments/cpu_replay_probe.py, production replay constructors/checkpoints, configuration.py, cloud/remote.py, separate config and focused tests.
- [x] Write RED tests for real hybrid construction, sample/restore parity and cloud argument propagation.
- [x] Reuse HybridRing with explicit configuration and factory, include layout in checkpoint validation.
- [x] Verify CPU smoke BASE/PQL with small replay, existing checkpoint regressions and default behavior.
- [x] Full suite and focused regressions verified; commit follows.

### Task 3: Operational verification
- [ ] Fresh-check current runs, quotas and idle accounts; keep existing jobs unchanged.
- [ ] Run one bounded two-T4 gate for requested env/batch/full hybrid replay and exact resume.
- [ ] Launch requested training only if gate passes, verify W&B steps advance.
- [ ] Record live evidence, commit/push and GitHub read-back; retain unresolved blockers explicitly.

Additional operational check: one interim CPU evaluation from MASAC BASE checkpoint 483a83a5 was submitted on idle account_03 using existing verified source 94487fd. Kernel vuliu123456/uavnet-validate-20261022-104625-33-cpu, W&B planned 9a9895ea. This is a one-episode pipeline check, not the final 20k evaluation.
