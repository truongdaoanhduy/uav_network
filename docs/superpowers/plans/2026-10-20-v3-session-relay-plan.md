# V3 continuation and relay audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development task-by-task. Steps use checkbox syntax.

**Goal:** implement the user-approved resume,feasibility and relay/power evaluation improvements.
**Architecture:** extend existing training checkpoint state with exact Warp world and collector-episode state; keep legacy restores explicit. Extend existing evaluation/oracle paths with paired controls and persisted evidence.
**Tech Stack:** Python,Torch,Warp,unittest,W&B,Kaggle SDK; existing dependencies only.
**Spec:** docs/superpowers/specs/2026-10-20-v3-session-relay-design.md

## Global Constraints
- Preserve scientific equations and algorithm blobs; preserve4096env/20000episodes/1800steps/seed44 for future Kaggle runs for production profile.
- Never label an interrupted session as episode-budget complete.
- No quota bypass,account creation,mass launches,or secret output.

## Review Focus
- Pending Warp async steps must not be snapshotted/restored; shape/dtype/config mismatch rejected before mutation.
- Seed advancement and per-env episode returns/lengths must survive zero completed episode segments.
- Legacy checkpoint continuation semantics remain explicit; no-progress cannot chain indefinitely.
- PQL workers must drain before checkpoint; collector/learner RNGs and queued-policy semantics remain valid.
- Paired controls preserve seeds/movement and physical mission accounting; zero delivered reports yields null J/delivery.

### Task 1: Exact world continuation and truthful tracking
Files: environment/gpu/vector.py; experiments/{training,pql_training,pql_system,fixed_checkpoints,continuation}.py; cloud/{continuation,remote}.py; tracking.py; focused tests as needed. A small shared environment-checkpoint helper is allowed.
Interfaces: vector.snapshot_state()->dict,vector.restore_state(snapshot)->obs dict. Existing training payload has optional environment/collector state. Summary must expose environment_state_saved and initial/current global_step; child spec stores expected global_step. No algorithm changes.
- [ ] Add failing real split-world and zero-episode BASE/PQL resume tests,malformed-state and no-progress guard tests.
- [ ] Run and record RED.
- [ ] Implement minimal compatible snapshot/save/restore and scheduling/logging changes.
- [ ] Run focused suite and record GREEN; verify algorithms unchanged; commit only task files.
- [ ] Independent task review and fixes.

### Task 2: Paired relay/power evaluation and full mission feasibility
Files: existing feasibility/evaluation helpers; new experiments/relay_audit.py and scripts/audit_relay.py if needed; tests/test_relay_audit.py; docs.
Interfaces: communication ablation takes Action + observation/config and returns Action without altering movement; audit runner takes checkpoint,seeds,modes,fixed_power_w,output directory and emits atomic per-episode evidence plus summary. Feasibility runner reuses run_feasibility with exact supplied config and records privileged label.
- [ ] Add failing real tests for direct-only suppression of peer hops,fixed-power mapping,seed/protocol pairing,missing delivered denominator and restart-safe output.
- [ ] Run RED,implement minimal wrapper/runner,run GREEN and commit.
- [ ] Run controlled full35-target feasibility and matched CPU evaluation as feasible; incomplete evidence is labeled honestly.
- [ ] Independent task review and fixes.

### Task 3: Integration,compute validation and handoff
- [ ] Full CPU suite and frozen algorithm hashes; whole-branch independent review.
- [ ] Read current W&B jobs and check one selected account quota without changing credentials.
- [ ] Pin/push isolated verified branch needed for remote validation; run one bounded CUDA resume/evaluation check if available.
- [ ] Start/resume authorized production training only when validation and quota allow; preserve separate CPU visualize handoff and report real budget status.
- [ ] Save user-facing findings and links with observed limits.
