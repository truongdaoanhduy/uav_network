# AMP FP16 + torch.compile activation evidence

Updated 2026-10-26 23:36 +07:00.

## Scope

Implementation commit: `19ffcac` (`feat(training): enable AMP and compile for 1KiB profile`) on `feat/cpu-replay-20261022`.

User requested enabling AMP and `torch.compile` in the UAV training code. The production implementation already existed in `uav_network/algorithms/precision.py`, `uav_network/algorithms/optimization.py`, MASAC/MATD3, BASE/PQL training, cloud execution preflight and checkpoint guards. This change therefore activates that existing path in the primary 1 KiB profile instead of duplicating learner logic.

Changed `configs/comparison-20k-hybrid75-1kb.yaml` only in the intended acceleration/provenance fields:

- `runtime.amp_mode`: inherited `off` -> explicit `fp16`
- `runtime.compile_models`: `false` -> `true`
- `runtime.fused_optimizers`: remains `false`
- tracking group now contains `ampfp16-compile`
- tracking tags add `amp-fp16` and `torch-compile`

A structured old/new YAML comparison verified these are the only config differences. E4096, B8192, replay1M, CPU75/GPU25, 1 KiB report/payload, scenario G3, seed44, 20k episode budget, reward/network/model settings remain unchanged.

The eight existing G3 training/controller lineages are pinned to source `77a1d0f`; this branch edit does not mutate, restart or cancel them. Exact continuation from their off/off full-state checkpoints into this on/on configuration remains rejected.

## TDD activation proof

RED:

```text
HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q tests/test_production_hybrid_replay.py::test_harp_lawn_matched_profile_uses_one_kib_report
1 failed
AssertionError: assert 'off' == 'fp16'
```

GREEN after the minimal YAML change:

```text
1 passed in 3.54s
```

## Focused regression proof

```text
HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q \
  tests/test_precision.py tests/test_production_execution.py \
  tests/test_production_hybrid_replay.py tests/test_baseline_resume.py
65 passed, 17 subtests passed in 24.96s
```

This covers learner-owned autocast/GradScaler state, MASAC/MATD3 real updates, compile graph execution/state-dict stability, BASE/PQL precision checkpoint ownership and baseline exact resume contracts on the available local CPU test path.

Direct guard checks also created off/off BASE and PQL full-state checkpoints and attempted on/on exact continuation:

```text
BASE_GUARD_OK Fixed-schedule resume runtime.compile_models differs from checkpoint
PQL_GUARD_OK PQL resume runtime.amp_mode differs from checkpoint
```

## Repository hygiene

```text
git diff --check && python3 -m compileall -q uav_network
DIFF_COMPILE_OK
```

## Full-suite result

```text
HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONPATH=.:tests python3 -m pytest -q
806 passed, 95 skipped, 8 warnings, 1350 subtests passed in 509.72s (0:08:29)
exit 0
```

The first direct full-suite invocation was interrupted by a Sandbox transport 502 before a test result was returned. It was not counted. The authoritative rerun above completed in a separate local task and its exit-code receipt is `0`.

## Final self-review

No separate reviewer subagent is available in this session, so the author performed the required read-only whole-change review using the Superpowers code-review checklist. Findings: **0 Critical, 0 Important, 0 Minor**. Plan alignment, resume compatibility, production-source pin isolation, test coverage, config provenance and secret hygiene were checked. `wandb/` remains untracked and excluded from the change.

## Boundaries

- No new Kaggle/T4 benchmark was launched in this implementation task.
- No measured speedup, VRAM reduction or learning-quality equivalence is claimed from enabling the flags alone.
- No active cloud job, W&B run, HF artifact/checkpoint, reward, geometry, replay capacity, batch size, environment count or training budget was changed.
- Real CUDA execution remains protected by the existing cloud execution preflight, which checks actual FP16 autocast/applied optimizer groups and compiled graph execution before accelerated training proceeds.
