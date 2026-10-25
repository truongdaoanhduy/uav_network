# Full Code Correctness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Audit the entire owned UAV codebase for scientific/modeling violations, configuration inconsistencies, algorithm/training bugs, and execution-pipeline defects, fixing only confirmed defects with TDD.

**Architecture:** Inspect the repository by domain boundaries, cross-check Dec-POMDP/CTDE assumptions against primary literature, use structural/static scans plus focused behavioral tests, then run the authoritative full suite. Preserve the current scientific profile and do not retune reward/hyperparameters or cloud jobs during correctness work.

**Tech Stack:** Python 3, PyTorch, NumPy, NVIDIA Warp, pytest, YAML, W&B/Kaggle tooling.

**Spec:** `docs/superpowers/CURRENT_TASK.md` and the user's 2026-10-25 full-code audit request.

## Global Constraints

- Primary profile remains `configs/comparison-20k-hybrid75-1kb.yaml` with report/payload 1024 B, buffer 3072 B.
- Actor execution must remain decentralized; centralized critic state is training-only CTDE privilege.
- Do not change reward, scenario, replay/batch, acceleration flags, or routing semantics unless a correctness defect is proven.
- Any source fix requires RED -> minimal GREEN -> focused tests -> full suite.
- Do not mutate live cloud jobs as part of this audit.

## Review Focus

- Actor information leakage through observations, masks, helper/oracle paths, or checkpoint semantics.
- Reward/termination/replay handling that changes Bellman targets or leaks post-terminal state.
- Native/Warp semantic divergence in observation, communication, energy, retry, or report custody.
- Config/profile fields that are ignored, overwritten, inconsistent, or silently defaulted.
- Training/evaluation/continuation paths that mislabel, resume incompatible state, or report false completion.

---

### Task 1: Repository/configuration inventory
- [x] Parse every owned Python module and load/validate every YAML profile.
- [x] Inspect configuration defaults/validation and primary 1 KiB profile for ignored/contradictory fields.
- [x] Scan TODO/FIXME/NotImplemented/broad exception/unsafe defaults and classify findings.

### Task 2: Dec-POMDP observation/state/scenario audit
- [x] Trace actor inputs end-to-end for native and Warp backends.
- [x] Verify local sensing, peer/GCS range gating, masks, link-state, time/battery/report custody semantics.
- [x] Verify centralized state is critic/training-only and evaluation uses decentralized actor inputs.

### Task 3: Communication/reports/reward/termination audit
- [x] Trace report creation, custody, relay, GCS delivery, TTL, duplicate handling, retries and radio energy.
- [x] Check reward components and terminal conditions for consistency between native and Warp.
- [x] Run focused parity/mission/retry/energy tests.

### Task 4: MASAC/MATD3/PQL/replay/checkpoint audit
- [x] Inspect action encoding/masks, critic targets, actor updates, entropy/TD3 noise, target updates.
- [x] Check replay transition semantics, terminal handling, PQL handoff, exact-resume/version guards.
- [x] Run focused algorithm/replay/checkpoint tests.

### Task 5: Training/evaluation/cloud lifecycle audit
- [x] Trace collector -> replay -> learner -> checkpoint -> continuation -> CPU evaluation -> publication.
- [x] Check episode/global-step accounting, completion gates, seed/protocol/source provenance, and W&B axes.
- [x] Run focused lifecycle/evaluation tests without submitting cloud jobs.

### Task 6: Whole-repo verification and fixes
- [x] For each confirmed defect, write a failing regression test, apply the smallest root-cause fix, and rerun focused tests.
- [x] Run `PYTHONPATH=.:tests python3 -m pytest -q tests` and `git diff --check`.
- [x] Update `CURRENT_TASK.md` and evidence with exact findings, limitations, test results, and commits.
