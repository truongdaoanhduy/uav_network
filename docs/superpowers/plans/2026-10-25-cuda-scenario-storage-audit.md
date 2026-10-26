# CUDA scenario-first and storage audit plan

> Latest scope override (26/10/2026): verification completed; user revoked HF/W&B deletion. Cleanup tasks below are historical and must not be resumed. See CURRENT_TASK and evidence/2026-10-26-checks-only-reverification.md.

> **For agentic workers:** Use superpowers:executing-plans; storage inventories may run as independent read-only domain tasks. Follow checklist order for scenario and algorithms.

**Goal:** Use Kaggle CUDA to verify the scenario before the algorithms, fix confirmed defects with RED/GREEN evidence, and remove demonstrably obsolete W&B/HF data without losing required scientific results or exact-resume state.

**Architecture:** Reuse the pinned repository setup, existing tests, WarpVectorEnv, feasibility oracle and production execution preflight. Run a bounded private Kaggle T4x2 diagnostic with scenario, scenario stress, learner, then full-suite stages. Reconcile both storage services jointly before any deletion; archives require fresh hash restoration.

**Tech Stack:** Python, PyTorch CUDA, Warp1.18, pytest, Kaggle CLI, W&B, huggingface_hub.

**Spec:** User's 25/10/2026 request in CURRENT_TASK.md; scenario logic/modeling first, algorithms second. Existing full audit and CPU verification remain source context, not CUDA evidence.

## Global constraints

- Existing linked worktree/feature branch is already isolated. Preserve unrelated untracked logs.
- Preserve active jobs, run history/events/metrics/configs, all final/best policies and needed exact-resume/controller/media references.
- Source bug: reproduce with failing regression, apply minimal root-cause fix, run focused and full tests, rerun relevant CUDA stage.
- Keep primary report/payload1024 B, buffer3072 B, env4096, batch8192, replay1M, CPU75%, seed44, horizon1800; diagnostic reductions must be explicitly recorded.
- No automatic battery/reward/scenario retune for attractive scores. Feasibility-oracle failure alone does not prove impossibility.
- Never globally squash/rewrite HF history. Preserve immutable revisions and files used by retained manifests. Any intentional retirement must be explicit and auditable.

## Review focus

- Scenario causal locality, hidden receipt information, disconnected placement, finite-horizon/battery consistency.
- Report TTL/custody/admission/retry/energy and native/Warp parity under saturation and long rollouts.
- CUDA devices, streams, replay capacity/layout, full-state resume and nonfinite updates.
- False full-budget completion or evidence inferred from submission rather than downloaded output.
- Deleting a unique checkpoint/raw metric file or breaking retained immutable HF references.

## Tasks

### 1. Fresh preflight and bounded CUDA scenario gate
- [ ] Reconcile Git, W&B live runs, healthy Kaggle account quota/owned kernel states, pinned source and no duplicate diagnostic.
- [ ] Launch private T4x2 diagnostic, verify actual hardware from output.
- [ ] Run all scenario/environment/network/locality/energy/reward/termination regressions before learner stages.
- [ ] Run primary-profile long horizon and4096-world capacity diagnostics plus engineering feasibility controls; record exact scope.

### 2. Scenario logic/modeling investigation and fixes
- [ ] Trace scenario assumptions and examine CUDA/feasibility findings; classify confirmed implementation bugs, model assumptions and calibration risks.
- [ ] For each confirmed bug: add focused RED regression, minimal fix, GREEN, full CPU verification and relevant CUDA rerun. Do not change model assumptions merely to improve metrics.

### 3. Algorithms after scenario gate
- [ ] Run production-shape BASE/PQL execution preflights for MASAC and MATD3 on two actual GPUs.
- [ ] Run four bounded CUDA train/save/load/evaluate checks, inspect finite parameters, update counts, checkpoint protocol and resume tests.
- [ ] Run complete repository test suite on CUDA; download XML/logs and explain every skip/failure. Apply TDD fixes if confirmed, then rerun.

### 4. Joint W&B/HF cleanup
- [ ] Inventory runfiles, artifact versions/aliases/consumers, all W&B HF manifests, HF head/history/LFS and explicit code/controller/media pins.
- [ ] Separate needed final/resume/provenance from obsolete intermediates, duplicate diagnostics and standalone tests. Record an exact retention/deletion list before mutations.
- [ ] For redundant large terminal-success runfiles: consider lossless compressed HF archive, immutable fresh download/decompression SHA verification and persisted source mapping before W&B file removal. Preserve live W&B history/events/config/summary.
- [ ] Retire only proven obsolete HF paths/objects with a retained-reference proof and no global history rewrite; restore-check all affected retained references afterward.
- [ ] Requery logical/physical storage and actual objects after cleanup; report exact reclaimed bytes versus browsing-only changes.

### 5. Verification and continuity
- [ ] Consolidate source hashes, tests, scenario findings, every deletion/retention reason and limits in evidence/CURRENT_TASK.
- [ ] Review any source diff independently, commit/push on working branch and verify remote. Do not claim completion while CUDA or cleanup proof is pending.
