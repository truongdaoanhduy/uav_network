# Overflow diagnostics and telemetry Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The user explicitly authorized execution of the preceding proposal.

**Goal:** Keep handled AMP overflow from crashing episode logging and preserve enough evidence to diagnose its numeric source.
**Architecture:** Shared scalar telemetry conversion; first-overflow observer at the pre-clipping boundary; bounded replayable snapshots and an offline FP32/FP16 comparison command. First full recovery backup occurs within 30 minutes while later attempts retain existing spacing and limits.
**Tech Stack:** Python, PyTorch, pytest, existing W&B/Kaggle/HF tooling.
**Spec:** ../evidence/2026-10-28-failed-training-audit.md (including follow-up); current user authorization.

## Global Constraints

- Preserve seed, reward, scenario, hybrid layout, loss formula, update/skip semantics and model architecture.
- Protect existing healthy lineages; diagnostic canaries are distinct from production and exact resume.
- Keep credentials out of artifacts. Optional diagnostic failures cannot stop healthy training.
- Evidence distinguishes raw gradient overflow, norm reduction overflow, and unproven upstream causes.

## Review Focus

- Nonfinite telemetry in old history and final summary must remain valid strict JSON.
- Invalid loss/model values must retain fail-fast behavior.
- Capture must preserve RNG and optimizer state and be bounded in threaded PQL workers.
- Snapshot failure must leave training and scaler backoff intact.
- First backup and later capped attempts must preserve prior verified recovery pointers.

### Task 1: Safe learner telemetry

Files: experiments/evidence.py, training.py, pql_training.py, tracking.py; tests/test_overflow_telemetry.py.
Interfaces: learner_telemetry(metrics: dict) -> dict, with null numeric value, nonfinite evidence and stable flags.
- [x] RED: real small MASAC/MATD3 BASE/PQL episode, summary, resume and W&B tests with inf/nan metrics.
- [x] GREEN: sanitize telemetry at episode and W&B boundaries, retaining finite scientific records.
- [x] Verify focused regressions and commit.

### Task 2: Capture and replay first overflow

Files: algorithms/precision.py; new experiments/overflow_capture.py; training.py, pql_training.py and tracking.py; scripts/replay_overflow.py; tests/test_overflow_capture.py.
Interfaces: observer receives raw gradients and event at skip; recorder captures batch/model/scaler/RNG; replay compares identical starting state and input in FP32/FP16 without changing production.
- [ ] RED: raw-inf vs finite-grad norm-overflow evidence, unchanged Adam/RNG, one snapshot per group, failed storage isolation and FP32/FP16 replay tests.
- [ ] GREEN: implement pre-clipping observation and integrate BASE/PQL; bounded first-event capture and published diagnostic files.
- [ ] Verify focused regressions; run available diagnostic canary with truthful device/provenance labels; commit.

### Task 3: Early recovery and final verification

Files: experiments/recovery_checkpoints.py; tests/test_recovery_checkpoints.py; diagnostic documentation/current task.
- [ ] RED: canonical first backup at 1800s rather than 9000s; later spacing and max attempts unchanged.
- [ ] GREEN: cap first delay at 1800s.
- [ ] Run full suite, compile/diff/secret checks; fresh whole-change review and address material findings with RED/GREEN.
- [ ] Push working branch, verify remote, report observed numerical evidence and limits.
