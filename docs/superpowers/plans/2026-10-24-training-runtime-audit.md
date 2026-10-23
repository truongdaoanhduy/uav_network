# Training Runtime Audit and Measured Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain actual 20k×1800 runtime from full owned-code audit and live evidence; retain only verified, useful fixes.
**Architecture:** Keep healthy training unchanged. First narrow performance candidate reuses valid link gains only in packet_step's frozen geometry; generic carrier-sense remains defensive. Follow with independent lifecycle gate fixes supported by failing tests.
**Tech Stack:** Python, PyTorch, Warp1.18, pinned UavNetSim/SimPy, Kaggle T4, W&B.
**Spec:** User's request to audit all scenario/algorithm/runtime code and online logs, find why 12h is insufficient, and implement justified fixes; operating/scientific invariants in ../CURRENT_TASK.md.

## Global Constraints

- Preserve report1MB, targets35, obstacles25, env4096, batch8192, max_steps1800, target20k.
- Preserve RNG/seed, exact packet timings/insertion ordering, backoff/ACK/retry, custody/TTL/admission, PHY, energy/reward/state, and exact resume.
- Never stop/restart/rewrite healthy production jobs. Diagnostics use same-account available quota; no account rotation to evade limits.
- No experimental source in recovery watcher. No new dependency. No secrets in output/evidence.
- Negative performance result means revert candidate; do not call an unmeasured change a fix.

## Review Focus

- Generic busy called immediately after zero allocation/position assignment must retain displacement validation.
- Frozen busy must match all state/cache/diagnostic arrays across arrival starts/ends, ACKs and movement between steps.
- T4 gate must measure active1MB peer traffic, all initialized arrays/RNG and three alternated runs each side.
- Final CPU visualization must reject time_budget or incomplete targets, without breaking complete20k handoff.
- Independent compile/fused settings must trigger preflight without silently enabling AMP or altering running jobs.

### Task 7A: Audit and frozen-geometry carrier sense

**Files:** communication/gpu/phy.py, communication/gpu/mac.py; tests/test_propagation.py; evidence/2026-10-24-task7-*.json and audit report.
**Interfaces:** Keep phy.busy(s,p,e,node,now) generic. Add busy_in_step(s,p,e,node,now), with explicit invalidated-cache/frozen-geometry precondition. mac imports this scoped path. No state schema change.

- [x] Read owned scenario/algorithms/native/cloud paths; establish phase-dependent collector/learner cost and actual10h/11h budgets. Record fresh child status and coverage.
- [ ] Add a Warp integration test before implementation: generic and scoped CCA yield same decisions and every initialized array after arrival queries, then after positions change plus per-step cache invalidation. Existing standalone allocated-state CCA test must still pass.
- [ ] Run it on baseline; expected missing scoped path failure. Preserve failing log.
- [ ] Factor one shared carrier-sense body; use valid pair_gain directly only in scoped path, otherwise existing gain(). Route the three MAC callers through scoped path.
- [ ] Run propagation/native packet/retry/energy/parity checks and CPU fixture smoke.
- [ ] Commit pinned candidate marked unpromoted. Run T4 B,C,C,B,B,C active1MBpeer4096 seed44, full state hashes, native oracle. Require benefit above variability; otherwise revert.

### Task 7B: Proven final visualization gate

**Files:** cloud/visualize.py; tests/test_cpu_visualization.py or existing final-budget test file.
**Interfaces:** build_visualization_spec(parent,results) retains shape; reject any learner without >=requested episodes, completion flag true, and stop_reason=episode_budget.

- [ ] Write/run failing tests for finished/time_budget, inconsistent completion and valid completed budget; reuse current continuation completion predicate if present.
- [ ] Implement smallest shared gate fix, preserve immutable artifact validation and complete learner output.
- [ ] Run CPU visualization/continuation/handoff tests; commit independently.

### Task 7C: Independent acceleration preflight gate

**Files:** cloud/remote.py; tests/test_production_execution.py or execution-check tests.
**Interfaces:** Existing generated execution_check invocation runs whenever supported AMP/compile/fused acceleration is requested; no runtime defaults changed.

- [ ] Write/run failing compile-only and fused-only bootstrap tests, plus default-off control.
- [ ] Change only bootstrap predicate with existing config API; run execution/cloud tests.
- [ ] Full suite, review whole change, refresh W&B, save audit findings/limits and final decision, commit/push all progress records.

## Execution

User explicitly requests implementing found fixes. Execute inline; read-only domain audits already delegated. Root integrates sequentially and requests final independent review. Shared wrap-up deadline/terminal metric batching remain measured follow-ups, not speculative patches in this plan.
