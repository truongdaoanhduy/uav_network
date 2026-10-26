# Measured Acceleration Policy Design

## Intent

Encode the T4/G3 acceleration choices proven in Tasks 3–5 into the canonical production cloud launch path so future 20k launches do not depend on a human remembering the MATD3-PQL exception.

## Goals

- Apply only to `configs/comparison-20k-hybrid75-1kb.yaml` in `train`/`validate` cloud stages.
- Default measured policy: AMP FP16 ON and fused Adam OFF for every measured mode; compile ON except MATD3 PQL, where compile is OFF.
- Preserve explicit user `--set runtime.amp_mode`, `runtime.compile_models`, or `runtime.fused_optimizers` overrides.
- Keep benchmark stages untouched so future matched experiments can set any cell explicitly.
- Keep auto-continuation exact-resume behavior unchanged by baking the effective policy into normal launch overrides.
- Make W&B acceleration tags reflect the resolved runtime instead of stale YAML tags.

## Non-goals

- No fused-Adam activation.
- No scientific config retuning.
- No policy for other config files/hardware/scenario shapes.
- No automatic mutation of existing runs or checkpoints.

## Approaches considered

1. Duplicate one YAML per algorithm/mode. Rejected: high config drift risk.
2. Add per-algorithm override maps to every cloud spec/continuation path. Rejected: more invasive than needed for the current separate-account production workflow.
3. **Chosen:** canonical-profile launcher policy plus a safety guard requiring separate MASAC/MATD3 PQL production launches. Single-algorithm launches get ordinary resolved overrides, so all downstream benchmark/preflight/train/resume code remains unchanged.

## Behavior

For canonical `train`/`validate` launches with no explicit acceleration override:

| Mode | amp_mode | compile_models | fused_optimizers |
| --- | --- | --- | --- |
| MASAC BASE | fp16 | true | false |
| MATD3 BASE | fp16 | true | false |
| MASAC PQL | fp16 | true | false |
| MATD3 PQL | fp16 | false | false |

If an acceleration key is explicitly supplied with `--set`, the policy does not replace that key.

For canonical PQL `train`/`validate` with `--algorithm both`, refuse before submission and require separate launches, because the measured compile choice differs by algorithm. BASE `both` remains valid because both algorithms share the same policy. Benchmark `both` remains valid and receives no automatic policy.

## Provenance

`authoritative_tags(config)` removes stale acceleration tags and appends tags derived from the resolved runtime: `amp-fp16`/`amp-off`, `torch-compile`/`compile-off`, and `fused-adam`/`fused-off`. The canonical W&B group becomes acceleration-neutral so mixed compile policy is not mislabeled.

## Validation

TDD will cover policy scope, explicit-override precedence, PQL-both refusal, BASE-both allowance, benchmark non-interference, and authoritative acceleration tags. Full offline suite must pass before completion.
