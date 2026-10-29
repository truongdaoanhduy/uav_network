# Controlled 20/15 UAV Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the user's approved easier 20/15 2D mission and seven controlled study variants before later 3D/Bayesian sensing.
**Architecture:** Optional config flags extend the existing native/Warp pipeline; both learners canonicalize the controller-owned power slot. A small study CLI materializes validated case configs and reuses existing sequential train/evaluate flows.
**Tech Stack:** Python, NumPy, PyTorch FP32, Warp 1.18.0, YAML, existing pytest/unittest.
**Spec:** docs/superpowers/specs/2026-10-29-20t15o-controlled-study-design.md

## Global Constraints
- 6 UAV; map 1000 m; 20 targets; 15 obstacles; 1800 steps; finite 500 m radio range and battery.
- New profile power_control=link_budget, margin 3 dB. Legacy field default learned.
- Legacy action ABI stays 3 continuous slots, controller-owned slot canonical zero and excluded from MASAC entropy.
- Coverage is per-UAV local FOV+LOS, optional 8x8; default zero cells preserves legacy layout.
- 7 one-factor variants: baseline, learned_power, buffer16, buffer32, ttl600, coverage8, battery2x.
- Periodic CPU validation 5 episodes every 3686400 transitions, fixed validation/test banks; final 30.
- Training seeds 44,100044,200044. Plans do not submit cloud jobs.
- No 3D migration in this phase; document the verified reference sensing contract.
- Preserve existing CURRENT_TASK.md edits; no new dependencies; no source file over 500 lines.

## Review Focus
1. Controller power changes after UAV motion and the chosen receiver; no stale/clipped power misrepresented as guaranteed link success.
2. Shared/independent actors and PQL inference copies must all suppress the ignored dimension.
3. Legacy missing config/array fields may normalize only to inactive defaults; no fresh-state loss on exact resume.
4. Local coverage must not see another UAV's scan, hidden targets or remote GCS delivery; partial reset must preserve live worlds.
5. Study aggregation must expose missing/budget-incomplete/unequal-budget/precision-mixed data and count training seeds separately from eval episodes.

### Task 1: Link-budget power and learner semantics
Files: configuration.py; communication/link_budget.py, backend.py, gpu/power.py, gpu/mac.py; environment/gpu/state.py; algorithms/action_utils.py, masac.py, matd3.py; experiments/pql_workers.py; tests/test_controlled_power.py.
Interfaces: network.power_control in {learned,link_budget}, network.link_budget_margin_db finite [0,30]; link_budget_power_w(required,network); selected_power helper in native/Warp. canonical_continuous(...,learn_power=True), joint_action(...,learn_power=True). Inference copy carries learn_power.
- [ ] Write behavioral tests: selected receiver power and clamp/margin, actual native/Warp packet traces, actor power/replay invariance and zero power-head gradients for both algorithms/sharing, PQL inference.
- [ ] Run focused pytest; observe failures from missing behavior.
- [ ] Implement helpers, controller power after motion, ignored-slot canonicalization and 2-axis MASAC entropy.
- [ ] Verify controlled_power plus legacy power/energy/masking/parameter-sharing tests.
- [ ] Commit this deliverable.

### Task 2: Local coarse coverage and exact checkpoints
Files: configuration.py; environment/coverage.py, env.py, observations.py, gpu/geometry.py, gpu/world.py, gpu/state.py, gpu/observations.py, gpu/vector.py; experiments/environment_checkpoints.py, fixed_checkpoints.py, pql_checkpoints.py as required for legacy normalization; tests/test_coverage_memory.py.
Interfaces: sensor.coverage_grid_cells 0 or integer 1..32; env.coverage [N,G*G]; Warp s.coverage [E,N,G*G]; update after sensing and on reset only for selected worlds. Append memory before destination mask and into critic state.
- [ ] RED tests native/Warp FOV/occlusion/locality, persistence, selective reset, snapshot continuation and loading missing inactive legacy fields.
- [ ] Implement minimum state and observation integration; no network sharing of coverage.
- [ ] GREEN focused coverage + decentralized observation + exact resume/checkpoint tests.
- [ ] Commit.

### Task 3: Canonical profile, controlled cases and periodic evaluation
Files: configs/comparison-20k-hybrid75-1kb.yaml; configs/comparison-20k-hybrid75-1kb-35t25o.yaml; experiments/study.py; experiments/cli.py; cloud/identities.py only if needed for unambiguous study labels; tests/test_controlled_study.py.
Interfaces: plan_study(config_path,output_dir,*,seeds,variants,algorithms,modes,overrides=()); run_study(manifest_path,case_ids=None); summarize_study(manifest_path).
CLI: python -m uav_network study {plan,run,summarize}; plan is read/prepare only, run explicitly selects local execution; no new external scheduler.
- [ ] RED tests 84 unique cases, paired seeds/one-factor differences, old profile preservation, disjoint training/validation/test banks, nonzero eval cadence.
- [ ] RED real CLI/config materialization and sequential tiny execution; summary excludes incomplete budget and computes mean/sample std over independent seed means, validates final eval provenance.
- [ ] Implement profile update and minimal CLI/manifest/summary. Respect requested overrides; enforce FP32 for this study.
- [ ] GREEN study, config, production identity and async evaluation tests.
- [ ] Commit.

### Task 4: Verification, feasibility, smoke and handoff
Files: docs/controlled-20t15o-study.md; docs/superpowers/evidence/2026-10-29-20t15o-study.json; docs/superpowers/CURRENT_TASK.md (only our appended record); plan checkboxes/ledger.
- [ ] Full-horizon 20/15 Warp CPU privileged oracle on at least five seeds; retain actual outcomes without claiming failures prove impossibility.
- [ ] Tiny real train/evaluate smoke across MASAC/MATD3 BASE/PQL and three independent seeds, with tracking off; no convergence claim.
- [ ] Run full pytest, project unittest, compileall, git diff --check and secret-value scan with counts only.
- [ ] Fresh reviewer reviews a470a40..HEAD plus working diff and spec. Fix critical/important findings with RED->GREEN, then full suite.
- [ ] Document actual verification, commands for selected study cases, 84-case cost scope, future reference sensing/XYZ plan. Record user-visible limitations.
- [ ] Commit/push verified changes and completion record. Preserve user's initial unrelated CURRENT_TASK.md diff.
