# Task 6 — measured acceleration policy in production launch path

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Input evidence

Tasks 3–5 measured the canonical G3/T4 profile. The measured candidate policy is:

| Mode | AMP | torch.compile | Fused Adam |
| --- | --- | --- | --- |
| MASAC BASE | FP16 | ON | OFF |
| MATD3 BASE | FP16 | ON | OFF |
| MASAC PQL | FP16 | ON | OFF |
| MATD3 PQL | FP16 | **OFF** | OFF |

The MATD3-PQL compile decision has three matched observations per mode. FP16+compile averaged 3065.44 transitions/s versus 3464.35 transitions/s for FP16-only, a repeated 11.51% throughput loss, while saving about 3.11% peak allocated GPU1 VRAM.

## Design boundary

The policy is deliberately narrow:

- only `configs/comparison-20k-hybrid75-1kb.yaml`;
- only cloud `train` / `validate` stages;
- benchmark stages remain policy-free;
- noncanonical configs remain unchanged;
- explicit user acceleration overrides win;
- canonical PQL production with `--algorithm both` is refused because MASAC and MATD3 need different compile choices;
- BASE `both` is still valid because both BASE learners use the same measured policy.

This avoids a global claim that MATD3 or PQL should never use `torch.compile`. The result is specific to the measured T4/G3 workload.

## Implementation

Source commits:

- `b9d660a70122145f6cd1e53f79b3d40aa413d10c` — policy helper, launcher integration, runtime-derived acceleration tags, neutral tracking group, TDD coverage.
- `8bedf4cf74190ddd1eb8299bd3cc2c60af5a8f4d` — canonical profile regression updated for the neutral group.

`uav_network/cloud/acceleration.py` is a small pure helper. It appends missing measured acceleration overrides only when the canonical train/validate scope matches. It does not replace explicitly supplied `runtime.amp_mode`, `runtime.compile_models`, or `runtime.fused_optimizers` values.

`uav_network/cloud/cli.py` reloads the resolved config after policy materialization. Those effective overrides are written into the normal cloud manifest, so existing continuation logic copies the exact acceleration settings without a new continuation mechanism.

`authoritative_tags(config)` now removes stale acceleration tags and regenerates them from the resolved runtime (`amp-fp16`/`amp-off`, `torch-compile`/`compile-off`, `fused-adam`/`fused-off`). The canonical W&B group is now `uav-v3-full20k-e4096-b8192-r1kb-cpu75-accel-policy-20261027`, avoiding a group name that falsely says every member uses compile.

## TDD evidence

Before launcher/tag integration, the focused policy + cloud identity run had four expected failures:

- missing `compile-off` authoritative tag;
- old group still contained `ampfp16-compile`;
- MATD3 PQL manifest still resolved `compile_models=true`;
- stale `torch-compile` tag remained when compile was false.

After the minimal integration:

```text
20 passed in 1.30s
```

Focused continuation/resume/execution regression:

```text
81 passed, 2 skipped, 46 subtests passed in 7.25s
```

Command covered acceleration policy, cloud identities, continuation, controller, BASE resume, PQL training and execution-check tests.

## Final verification

Fresh full offline suite at final source HEAD `8bedf4c`:

```text
820 passed, 95 skipped, 8 warnings, 1360 subtests passed in 197.84s (0:03:17)
exit code 0
```

Additional gates:

- `git diff --check`: exit 0.
- `python3 -m compileall -q uav_network`: exit 0.
- changed-file credential/token grep: no matches.
- implementation files remain below the repository 500-line ownership limit.
- existing untracked `wandb/` remains untouched.

## External documentation cross-check

PyTorch documentation states that `torch.compile` can optimize selected regions while other functions remain eager, and that observed speedups vary with architecture/workload. PyTorch benchmarking guidance also emphasizes warm-up and accelerator synchronization. This is consistent with keeping MATD3-PQL eager for this measured profile while retaining compile for the other measured modes, rather than treating compile as universally beneficial or harmful.

## Operational boundary / next task

No existing Kaggle/W&B training lineage was restarted, cancelled, mutated or resumed as part of Task 6. Task 7 must first fresh-check current G3 lineages and account availability before any new accelerated production launch, and must not duplicate a live lineage.
