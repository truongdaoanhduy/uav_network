# Scenario V2 Fixes and GPU Training Implementation Plan

> **For agentic workers:** Use test-driven-development for the independent retry, safety and telemetry tasks; integration and launches remain owned by the root coordinator.

**Goal:** Correct shared packet/safety defects, make diagnostics distinguish opportunity from policy choice, and collect real GPU evidence before matched baseline training.
**Architecture:** Reuse the existing Native/Warp packet model and safety filter. No learner changes and no new simulator architecture.
**Tech Stack:** Python, NumPy, Warp, pytest, Kaggle, W&B.
**Spec:** docs/scenario-v2-fixes-20261018.md; prior V2 design under docs/superpowers/specs/.

## Global Constraints
Standard MASAC/MATD3 unchanged; V1 YAML unchanged; seed44; shared V2 mission35targets/25obstacles/h1800; no per-hop reward; actual energy; no secret output; no interference with current jobs.
## Review Focus
Terminal retry vs queued/live packets; partial fragments and ACK-loss custody; feasible wall peer conflicts vs true cages; zero opportunity and pooled denominators; immutable source/online W&B/private credentials on GPU.

### Task 1: Terminal retry cleanup
Files: communication/backend.py, communication/gpu/mac.py; tests/test_retry_recovery.py.
- [ ] RED real low-power failure followed by high power same destination next step.
- [ ] Minimal terminal pending retirement; verify repeated batches, silence, TTL, fragmentation and ACK invariants.
- [ ] GREEN focused Native/Warp tests; coordinator reviews diff.
### Task 2: Wall peer conflict
Files: environment/avoidance.py, environment/gpu/motion.py; tests/test_safety_progress.py or test_boundary_escape.py.
- [ ] RED two UAVs atleftwall [0,715],[0,735] with opposite diagonal commands stall despite feasible tangents.
- [ ] Validate bounded wall guide with existing certificates; use existing tangent fallback on invalid guide.
- [ ] GREEN continuous progress plus swept safety, acceleration and Native/Warp parity; true cage still stops safely.
### Task 3: Passive action opportunities
Files: communication/diagnostics.py, communication/gpu/diagnostics.py, communication/backend.py afterTask1release, communication/gpu/telemetry.py (actualrecord_actionlocation), experiments/telemetry.py, tests/test_action_opportunities.py.
- [ ] RED exact joint eligible+silent and closer-peer-without-GCS counters plus zero/pooled denominators.
- [ ] Append counters without changing existing indices; hook Native/Warp sampling.
- [ ] GREEN native/Warp diagnostics and passive rollout equality.
### Task 4: Channel/source evidence
Files: docs/scenario-v2-fixes-20261018.md; docs/channel-model-evidence-20261018.md.
- [ ] Verify pinned upstream and primary papers; state deterministic PHY, link-budget limits and custody/beacon assumptions.
- [ ] Freeze tagged fixed-V2 config and learner/V1 hashes; choose minimal diagnostics rather than changing parameters for score.
### Task 5: Integration and GPU evidence
- [ ] Full active test suite and independent whole-diff review, resolve material findings with RED/GREEN tests.
- [ ] Commit/push feature branch; verify remote SHA.
- [ ] Kaggle CUDA validate both learners and BASE/PQL pipeline with online tracking; inspect results.
- [ ] Matched fixed-V2 baseline launches across available accounts, manifests and live W&B links; read early diagnostics and fix any proven shared error before claiming success.
