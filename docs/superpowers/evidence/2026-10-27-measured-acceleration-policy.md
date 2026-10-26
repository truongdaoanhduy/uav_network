# Task 6 — measured per-algorithm acceleration policy

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Goal

Encode the T4/G3 measurements from Tasks 3–5 into the canonical cloud production launch path so future runs cannot accidentally apply `torch.compile` to MATD3 PQL merely because the base YAML default is compile-on.

## Source changes

- `b9d660a feat(training): encode measured acceleration policy`
- `8bedf4c feat(training): apply measured acceleration policy` (updates the existing canonical-profile contract test to the acceleration-neutral W&B group)

The pure policy helper is `uav_network/cloud/acceleration.py`. It applies only when the exact canonical profile `configs/comparison-20k-hybrid75-1kb.yaml` is launched through cloud stage `train` or `validate`.

## Measured policy

| Mode | AMP | torch.compile | fused Adam |
| --- | --- | --- | --- |
| MASAC BASE | fp16 | ON | OFF |
| MATD3 BASE | fp16 | ON | OFF |
| MASAC PQL | fp16 | ON | OFF |
| MATD3 PQL | fp16 | **OFF** | OFF |

The MATD3-PQL exception is supported by n=3 FP16/off and n=3 FP16/compile measurements from Task 5. Compile reduced throughput by 11.51% on the means while saving VRAM, so FP16-only is the measured production candidate for this mode.

## Safety / scope behavior

- Benchmark stage is policy-free, preserving matched experiments.
- Other config files are untouched even if they share the same filename elsewhere.
- Explicit user overrides for `runtime.amp_mode`, `runtime.compile_models`, or `runtime.fused_optimizers` take precedence over measured defaults.
- Canonical PQL `--algorithm both` in train/validate is refused before submission because MASAC and MATD3 need different compile settings. Separate launches are required.
- BASE `both` remains valid because both algorithms share FP16+compile.
- Effective acceleration overrides are written into the normal cloud manifest, so continuation/exact-resume receives the same runtime values; checkpoint compatibility logic was not changed.
- No historical run or checkpoint is mutated or resumed with changed acceleration flags.

## Provenance

The canonical W&B group is now acceleration-neutral: `uav-v3-full20k-e4096-b8192-r1kb-cpu75-accel-policy-20261027`.

`authoritative_tags(config)` removes stale acceleration tags and appends tags from the resolved runtime:
- `amp-fp16` or `amp-off`
- `torch-compile` or `compile-off`
- `fused-adam` or `fused-off`

Replay-layout and generation tags remain authoritative as before.

## TDD / verification

Integration RED before implementation: 4 failures showed the exact missing behavior — MATD3 PQL still compiled, stale acceleration tags, and compile-specific group naming.

After implementation:
- acceleration + identity integration: 20 passed.
- targeted canonical-profile + acceleration/identity: 21 passed.
- focused cloud/continuation/resume suite: **76 passed, 2 skipped, 770 subtests**.
- first full suite exposed one stale old assertion requiring `ampfp16-compile` in the group; this was a test contract incompatible with the new acceleration-neutral provenance design.
- after updating that contract, full offline suite: **820 passed, 95 skipped, 8 warnings, 1359 subtests, RC=0, 260.34 s**.
- `git diff --check`: clean.
- `python3 -m compileall -q uav_network`: exit 0.
- changed-file credential scan: no Kaggle/W&B/HF secret patterns found.

The full-suite warnings are pre-existing W&B Pydantic deprecations and one Gymnasium pending-step close warning; no test failed.

## Design records

- Spec: `docs/superpowers/specs/2026-10-27-measured-acceleration-policy-design.md`
- Plan: `docs/superpowers/plans/2026-10-27-measured-acceleration-policy.md`

Existing untracked `wandb/` is preserved. No new 20k production training job is launched by this task.

## Next task

Task 7 should perform a bounded real-Kaggle launch-policy smoke using the new source: one MASAC PQL manifest must resolve compile ON and one MATD3 PQL manifest compile OFF, then verify the cloud execution preflight on 2xT4 before any new 20k production cohort is started.
