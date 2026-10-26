# Task 7 — real Kaggle smoke for measured acceleration policy

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Bounded `validate` smoke on the pushed source `906ee68b3ede00908f28e3e00b358c6e16dcb808`. The canonical profile was used with `pql.enabled=true`; no AMP/compile/fused override was supplied. Only the validation budget was reduced (`episodes=1`, `eval_episodes=1`, `max_steps=8`, `max_train_seconds=600`) and checkpoint upload was disabled for the smoke.

## Launch-policy proof

MASAC PQL:
- account_07
- kernel `kjhihgiugy/val-g3-masac-pql-h75-10261902-cadd73`
- W&B run `32e602d6`
- manifest resolved `runtime.amp_mode="fp16"`, `runtime.compile_models=true`, `runtime.fused_optimizers=false`
- tags include `amp-fp16`, `torch-compile`, `fused-off`

MATD3 PQL:
- account_13
- kernel `bjkjbhm/val-g3-matd3-pql-h75-10261902-866b8e`
- W&B run `c1c927e3`
- manifest resolved `runtime.amp_mode="fp16"`, `runtime.compile_models=false`, `runtime.fused_optimizers=false`
- tags include `amp-fp16`, `compile-off`, `fused-off`

Both kernels reached Kaggle `complete`.

## CUDA execution-check

MASAC PQL execution-check:
- status `passed`
- AMP autocast FP16 marker=1
- critic 32 applied / 0 skipped
- actor 32 applied / 0 skipped
- alpha 32 applied / 0 skipped
- compile ON: 3 unique graphs, 88 Inductor kernels, 0 graph breaks, 0 frame fallback
- devices: world/collector cuda:0; learner/replay hot tier cuda:1

MATD3 PQL execution-check:
- status `passed`
- AMP autocast FP16 marker=1
- critic 32 applied / 0 skipped
- actor 16 applied / 0 skipped
- compile OFF: 0 compiled graphs / 0 Inductor kernels, as required by policy
- 0 graph breaks / 0 frame fallback
- devices: world/collector cuda:0; learner/replay hot tier cuda:1

## Validation job result

Both bounded validate jobs completed one requested episode and reported `episode_budget_complete=true`, `stop_reason=episode_budget`, fresh state, saved environment state, and tracking status finished. Because the smoke intentionally ended at 8 environment steps, the validation trainer itself had zero PQL learning updates; learner-update correctness is proven by the dedicated execution-check above, not by the one-episode smoke.

Each kernel also passed:
- CUDA tests 11/11
- scenario CUDA tests 4/4
- mission regressions 8/8
- PQL smoke tests 11/11

## Output hashes

MASAC: hardware `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`; job `1c7d72298efedaf26816be528862112fab9e66599f62790b977f6b54fd451102`; execution `ba7ef2148331d82874333587ad9105729c0ef2e3befe602b3fb705c04cc1f1a7`.

MATD3: hardware `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`; job `a18c957a55881270be5a96d14b5d4b2983d2af203d5c65328e81fae6650d5f29`; execution `beae41cc56e5b1d87001f37c83cbad0cd9f3a96c92d3e04027b5b1a8c393f256`.

## Result

The new launcher policy is proven end-to-end on real 2xT4 Kaggle: MASAC PQL receives FP16+compile, while MATD3 PQL receives FP16 with compile OFF, without explicit acceleration overrides.

## Next task

Task 8 is a read-only production-cohort readiness audit: confirm the final 20k configuration, four single-algorithm launch modes, account quota/availability, continuation/visualization prerequisites, and absence of conflicts before any long-running submission.
