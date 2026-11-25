# Zhang–Xue v3.4 Reference Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (test-first) on this isolated branch.

**Goal:** Remove demonstrable packet queue/retry and network-aware reward defects from the user's v3.2 reference code, without changing production experiment semantics.

**Architecture:** Copy original research package into an isolated worktree. Improve bounded reference event transport and reward attribution; keep original baseline config and distinguish this harness from the real UavNetSim integration.

**Tech Stack:** Python 3.13, NumPy, PyYAML, PyTorch 2.7.1 CPU, unittest, Git worktree.

**Spec:** `docs/superpowers/specs/2026-11-25-zhang-xue-v34-reference-design.md`

## Global constraints

- No reset/clean, do not modify dirty main/study repositories.
- Gate 5 SimPy↔Warp parity deferred by user; do not claim completion.
- No GPU training or network push.
- No positive false-positive information credit in R1; no ground-truth leakage into actor.
- Keep fixed original v3.2 config and provenance separate from changed v3.4 behavior.

## Review focus

- Queue-full send must not transmit.
- No-route and radio blackout recover without bypassing app TTL.
- Immutable binary packet bounds reject invalid nested mutable inputs or uint16 overflow.
- Safety/boundary costs must not become positive bonuses.
- Target/empty cell distinction must apply to **all** positive information-reward terms.

## Tasks

### Task 1: baseline and isolated workspace
- [x] Inspect Git working trees, avoid dirty source.
- [x] Copy ZIP after traversal validation, run 5 v3.2 baseline tests.
- [x] Verify Gate 4 CPU PyTorch toy actor gradients.

### Task 2: packet transport regressions
- [x] Write 4 failing cases for queue-full, no-route recovery, sender blackout, queue cleanup; observe RED.
- [x] Modify queue and event handler; verify GREEN.
- [x] Write two additional failing cases for mutable/on-wire-invalid packet and invalid destination; observe RED.
- [x] Validate packet range/immutable payload and peer destination; verify GREEN.
- [x] Run 20 seeds × 100 packet invariant smoke.

### Task 3: network-aware reward regressions
- [x] Write 3 failing cases for signs, FP information, true TP latch; observe RED.
- [x] Implement finite/positive costs and per-cell truth projection at reward-only boundary; verify GREEN.

### Task 4: independent verification and continuity
- [x] Run full reference mini-harness test suite (14/14).
- [x] Run selected existing production slotted/M2 tests (30/30).
- [x] Run Gate 4 PyTorch CPU.
- [x] Package ZIP, readback, commit local worktree, record final report.

## Deferred

Full production actor/replay integration, true UavNetSim SimPy adapter and Warp parity, local CUDA, Kaggle GPU multi-seed training, remote push. They cannot be inferred from the passing reference tests.
