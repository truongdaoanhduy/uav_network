# Packet Queue Count Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the redundant packet-pool scan in `fill_session()` without changing packet behavior.

**Architecture:** Keep `node_queue_count` as the single existing queue-count source of truth. Replace only the local recount loop in `fill_session()`; all queue mutation paths remain unchanged.

**Tech Stack:** Python, NVIDIA Warp, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-10-23-packet-queue-count.md`

## Global Constraints
- Exact native/GPU packet parity must remain unchanged.
- No new state or dependencies.
- One optimization only.

## Review Focus
- Queue count decrements on dequeue/drop/retry completion remain balanced.
- Multiple sources do not share the wrong queue count.
- Streaming window reinjection still respects `max_queue_packets`.
- Relay and retry traces remain exact.
- Long 1 MB streaming remains bounded.

---

### Task 1: Reuse maintained queue count in `fill_session`

**Files:**
- Modify: `uav_network/communication/gpu/streaming.py`
- Test: `tests/test_gpu_packets.py`

**Interfaces:**
- Consumes: `s.node_queue_count[e, source]` maintained by existing MAC queue operations.
- Produces: identical packet behavior with no local O(packet_capacity) recount in `fill_session()`.

- [x] **Step 1: Write failing structural regression test** asserting `fill_session` does not scan `range(p.packet_capacity)` merely to compute `queued`, and uses `s.node_queue_count[e,source]` for the queue-cap decision.
- [x] **Step 2: Run the test and verify RED.**
- [x] **Step 3: Replace the recount loop with the existing counter.**
- [x] **Step 4: Run packet exactness and performance regression tests.**
- [ ] **Step 5: Commit the isolated optimization.**
