# XYZ GitHub cohort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Queue four seed44/20k XYZ roots through the existing GitHub supervisor.
**Architecture:** Reconstruct per-case state from owned Kaggle control receipts and
stable tracking groups. Reuse the pinned launch API and existing continuation/media.
**Tech Stack:** Python stdlib, current Kaggle/W&B/HF clients and GitHub Actions.
**Spec:** ../specs/2026-10-31-xyz-github-cohort-design.md

## Global Constraints

- Only account_01, authenticated owner kesfljhk. Never rotate accounts or cancel jobs.
- Source481f81f288d8e84a010dba52e4c8e30b5eebd391; E4096/B8192/R1M/FP32/eager.
- Shared actor+critic Hybrid75 XYZ,20T15O/1200steps, seed44,20000episodes.
- Production warmup32; no diagnostic248 override in training.
- Starts disabled; all6full-ring checks must have evidence; selected Hybrid checks must pass.
- Existing full-resume, validation-best and CPU30/publication gates remain mandatory.
- Current-task historical user edits stay unstaged.

## Review Focus

- Unrelated active job must block initial submission, not just managed prefixes.
- Lost initial response must be adopted by group; unknown terminal controls block.
- Partial GPU success or failed CPU publication must not advance the case.
- Low quota must pause on account01, with no alternative selection.
- Wrong source/owner/seed/budget or duplicate case identity must fail closed.

---

### Task1: Queue decision and pinned initial-launch boundary

**Files:** Create uav_network/cloud/cohort_queue.py, configs/xyz-cohort-seed44.json,
tests/test_cohort_queue.py; modify uav_network/cloud/github_supervisor.py.

**Interfaces:**
- Consumes client.kernel_status/run_kaggle/account_quotas/resolve_username and existing
  supervisor read_controls/_validate_job/_visualization_complete/_managed_ref.
- Produces reconcile_cohort(credentials, *, plan_path=None, dry_run=False)->dict.
- Internal _launch_case(plan,case,credentials)->dict returns a real launch receipt.

- [ ] Write tests for disabled plan/no RPC, any active job blocking, dry-run,
  initial submit then repeat/adoption, duplicate group rejection, wrong source,
  low quota, complete GPU without CPU wait, CPU publication failure wait, final
  matching CPU completion advances exactly one next case, all-complete idempotence.
  External backend fixture retains actual control JSON and ownership state;
  assertions target queue outcomes and server ledger, not mock call counts.
- [ ] Run pytest -q tests/test_cohort_queue.py and observe missing implementation RED.
- [ ] Implement bounded plan validation and receipt reconstruction, keeping all
  initial roots serial. Validate complete GPU/CPU checkpoint-parent correspondence.
- [ ] Implement _launch_case using TemporaryDirectory/shared clone at the pinned
  source and a separate Python process invoking existing cloud.cli.launch with
  credentials_from_environment. Public Namespace args only; no credential argv/file.
- [ ] Add optional queue outcome to the current report only when its plan is present;
  respect dry_run and existing supervisor exit-code/failure reporting.
- [ ] Run focused cohort/supervisor/cloud tests, then bare pytest and diff checks.
- [ ] Independent review; fix Critical/Important and rule on declined items.
- [ ] Commit/push queue code with enabled=false; verify Actions dry-run stays safe.

### Task2: Activate after actual readiness and start the cohort

**Files:** Update configs/xyz-cohort-seed44.json and measured evidence/CURRENT_TASK.
**Interfaces:** Consumes Task1 API and the actual6-case readiness index at source481.

- [ ] Read all6case results; require selected Hybrid algorithms passed, exact E/B/R, CUDA gates
  and1M ring wrap. Retain comparator OOM results as measured feasibility outcomes. Preserve any OOM/failure instead of shrinking hidden parameters.
- [ ] Record source/budgets/diagnostic warmup and service proof; enable the four-case
  plan with matching readiness evidence. Pin account01 and check no other active job.
- [ ] Run Actions dry-run and verify the expected first case only.
- [ ] Run normal reconciliation to submit first root; re-run to verify adoption/no
  duplicate. Confirm Kaggle hardware/source/resolved config and actual W&B training.
- [ ] Verify latest-resume/best-policy pointers are distinct when produced; existing
  supervisor owns continuation and final CPU visualize. No claim20k is done early.
- [ ] Commit/push current state, exact URLs and pending multi-seed limitations.

## Self-review

Spec coverage maps to Tasks1/2; no new algorithm/config semantics or dependency.
The five review-focus cases each have a Task1 test. Task2 enables only measured
readiness, so implementation/testing can proceed while GPU diagnostics run.
Native implementation is the existing execution method and the user has opted out
of repeated permission questions; continue within that authorization.
