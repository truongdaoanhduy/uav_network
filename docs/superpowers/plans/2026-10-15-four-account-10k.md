# Four-account matched 10k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.
> Independent baseline resume and cloud continuation tasks are dispatched using
> superpowers:dispatching-parallel-agents; root owns CLI, PQL budget and launch.
> Steps use checkbox syntax.

**Goal:** Execute four comparable 10000-episode cloud series and autonomous CPU visualization.

**Architecture:** Reuse pinned Kaggle scripts and W&B immutable artifacts.
Both trainers restore full state for GPU continuation; server finalizer chooses
another GPU job or final CPU job based on verified progress.

**Tech Stack:** Existing Python, unittest, PyTorch, Warp, W&B and official Kaggle SDK/CLI.

**Spec:** docs/superpowers/specs/2026-10-15-four-account-10k.md

## Global Constraints

- Preserve scenario, reward, model and algorithm math.
- One common config; pql.enabled selects base versus PQL.
- W&B online; one selected Kaggle account per series.
- Exact recorded budget10000; CPU rendering uses no GPU.
- No local orchestration dependency; no credentials in source/outputs.
- Owned text files <=500 lines; no new dependencies.

## Review Focus

- Large vector batches ending above remaining budget: cap records exactly.
- Fresh Kaggle accounts without owned resources: authenticated token identity.
- Baseline resume: preserve Adam/targets/replay/counters, retain v2 warm start.
- Time-budget stop with stale/no-progress state: fail before submission.
- Quota/upload/child submission failure: preserve training and explicit diagnostics.

### Task1: Resume and exact budget

Files: experiments/training.py, continuation.py, fixed_checkpoints.py,
tracking.py; root edits pql_training.py. Tests baseline resume and matched10k.

- [x] Write/observe failing CPU resume and nonmultiple budget tests.
- [x] Implement v3 fixed-schedule snapshot using existing checkpoint helpers.
- [x] Upload light policy + full resume at final, restore state truthfully.
- [x] Cap final episode records in both trainers.
- [x] Run targeted resume/PQL/tracking tests.

### Task2: Server continuation and account identity

Files: cloud/continuation.py, remote.py; root edits client.py, cli.py,
kernel.py and visualize.py. Tests cloud continuation and matched10k.

Interfaces: build_continuation_spec(parent,results), submit_continuation(...),
kernel_ref_for_segment(root_ref,index,cpu=False).

- [x] Write/observe failures for incomplete->GPU, complete->CPU, no-progress and quota.
- [x] Add --auto-continue single-algorithm contract and pinned next-stage specs.
- [x] Restore resume.pt for both architectures, keep CPU checkpoint.pt download.
- [x] Resolve fresh-account owner through official authenticated SDK.
- [x] Keep CPU URL stable and record continuation metadata in W&B.
- [x] Run cloud suite and source limits.

### Task3: Verify and launch

Files: configs/comparison-10k.yaml and docs/experiment results.

- [x] Verify common resolved config differs only by PQL toggle/algorithm selection.
- [x] Run full CPU suite and publish source.
- [x] Provision only private single-credential datasets on chosen accounts.
- [x] Run four-case GPU/CPU smoke and forced GPU->GPU->CPU smoke.
- [x] Launch10000 series on four accounts; confirm W&B online and progress.
- [x] Publish quota/account/run URLs and current statuses.

Review: first-session quota cap added after Important finding; targeted tests RED→GREEN.
Host credential download issue diagnosed as IPv6 TCP timeout; scoped IPv4 helper verified both private files on accounts08–11, with no product/network setting change.

Production preflight finding: BASE benchmark mixed NumPy actions with Warp tensors.
Real Warp CPU regression reproduced TypeError before fix, then passed for both learners;
CUDA counterpart is included in Kaggle test_cuda.py. Trainer and config unchanged.
Full suite after fix: 343 tests, 296 passed and 47 CUDA-only skips (146.980s).
Matched two-T4 comparison: 6/6 passed; immutable data and CPU video proof saved under docs/benchmarks.

Checkpoint I/O finding: repeated full replay writes within one vector batch.
Regression showed [1,2,3,4,4] writes before fix and [2,4] after for both algorithms;
periodic resume-before-tracker contract retained. Model/episode hashes unchanged
against8de297a in a fresh CPU Warp probe. Final CPU suite344,297 pass47CUDA skips.
Ruling: retain pinned source of the four already-running production jobs;
checkpoint coalescing applies to future launches. Report its wall-time confound.
No production job was cancelled or duplicated to apply this performance patch.

Task3 complete: four W&B/Kaggle production jobs Running with positive critic updates at03:10UTC.
Four CPU smoke videos verified; two later MASAC submissions were manually recovered from saved artifacts, then final CPU handoff automatic.
Exact service cause of those two RuntimeError pushes is not retained; do not label the entire128-episode chain fully unattended.
Quota16/16 verified; private account inventory stays out of git. Final run report docs/runs-10k-20261015.md.
