# Overflow capture, numeric evidence and logging fix — 28 October 2026

Follow-up 2026-10-28: [actor arithmetic reproduction and authorized production restarts](2026-10-28-matd3-actor-reproduction-and-restarts.md) resolves the earlier missing actor reproduction on new rollout data. The exact historical failed batch remains unavailable.

Runtime source: `98724b827ce0dd017bb319dea09aa3c31b90282a`. Branch: `feat/cpu-replay-20261022`.

## What is fixed

- Shared idempotent learner telemetry makes Inf/NaN numeric values null, retaining a flag and original nonfinite label. BASE/PQL episode records, resumed history, summaries and W&B scalar channels use it. Finite scientific values are retained.
- AMP computes norms before modifying gradients. Finite updates use equivalent norm clipping; nonfinite updates still skip all owned Adam steps and back off the scaler. Nonfinite forward loss and FP32 gradient errors retain their existing failure guards.
- The first critic/actor overflow attempts one bounded diagnostic snapshot per learner/group: original batch, model, original owned unscaled pre-clipping gradients, optimizer/scaler state, update counters and pre-forward Torch RNG. Metadata reports raw-gradient versus norm-reduction overflow. Optional diagnostic storage/upload failure cannot kill training.
- `python -m scripts.replay_overflow SNAPSHOT --device cuda:1 --output comparison.json` performs independent FP32/FP16 replays and reports parameter gradients. FP32 bypasses AMP counters; its applied count is null rather than a misleading zero.
- The first full recovery backup is due within 1800 seconds. Subsequent spacing and the four-attempt cap remain. Upload completion can occur after the due time; the observed earlier 15.6GB backup took approximately seven minutes to save/upload.

Scientific MASAC/MATD3 loss/action/target source files, model architecture, scenario, seed, reward, batch/replay settings and hybrid layout are unchanged. Old source-pinned running jobs do not gain the patch retroactively. Three failed original runs still lack verified full-state recovery checkpoints; no production relaunch or fabricated exact resume was performed.

## Measured CUDA evidence

All three diagnostic kernels completed on two Tesla T4 devices and their W&B diagnostic runs finished. These are separate diagnostic runs, not production lineages.

| Evidence | Result | W&B |
| --- | --- | --- |
| Four fresh canonical-sized cases | BASE cases did not overflow within the bound; MATD3 PQL and MASAC PQL naturally overflowed at their default loss scale 65536. No injected Inf or corrupted input. | [128d227d](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/128d227d) |
| Historical finite MATD3 model, fresh canonical data | Critic overflow at scale 8192; eight actor batches at scale 32768 did not reproduce the historical actor event. | [c031f202](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c031f202) |
| Same captured snapshots, FP32 reference gradients | FP32 finite, FP16 overflow reproduced in all three critic snapshots. | [9299f2f9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/9299f2f9) |

Canonical dimensions for these probes: 4096 environments, batch 8192, replay capacity 1M, CPU75/GPU25, seed44, scenario-v3/1KiB. Fresh-data probe does not claim to match the production collection/update timing exactly. Probes use eager execution; original compile flags and asynchronous replay limitations are recorded.

Observed examples from the CUDA FP32 reference:

| Case / parameter | Unscaled FP32 gradient abs max | Loss scale | Scaled reference | FP16 maximum |
| --- | ---: | ---: | ---: | ---: |
| MATD3 PQL / critics.0.q2.4.weight | 1.0508334637 | 65536 | 68867.421875 | 65504 |
| MASAC PQL / critics.2.q1.4.weight | 1.0643646717 | 65536 | 69754.203125 | 65504 |

For fresh MATD3 PQL, all six parameter tensors whose scaled FP32 reference exceeds the FP16 range match the six tensors with nonfinite FP16 gradients. For fresh MASAC PQL the same match holds for both affected tensors. Forward losses remain finite and nearly equal between precision modes: about 284.25 and 295.43. FP32 gradient norms are approximately 35.05 and 29.83; FP16 reports Inf and safely skips.

This directly demonstrates scaled-backward overflow in the observed canaries, not just JSON failure or norm-reduction overflow. The reference gradient magnitude itself need not be extreme: scaling a finite value around1.05 can exceed FP16 range. Gradient clipping happens after backward and cannot prevent an overflow that already occurred in that backward computation.

The trained-model critic probe has finite model/input values, loss about1932.15, FP32 gradient norm8542.00 and FP16 raw-gradient overflow. Fifty-two parameter tensors have a scaled FP32 reference above65504, while36 FP16 gradient tensors become nonfinite; the exact boundary differs across parameter computations. We do not claim a universal one-to-one threshold mapping for that probe.

## What remains unproven

The exact original failed batches were never retained. The historical MATD3 BASE actor overflow was not reproduced in eight fresh actor batches; its precise source operation remains unknown. Original history shows actor scale backoff65536 ->32768 ->16384 and a finite model snapshot, which is consistent with AMP overflow but is not an exact historical replay.

Inputs are normalized by source code and all captured canary batches are finite. This does not certify every lost historical sample or prove the entire training distribution is bounded. The evidence does not justify adding LayerNorm/BatchNorm, changing reward/learning rate, or blaming hybrid replay/actions. The first nonfinite parameter seen by a replay hook is an observation boundary, not proof of the first internal arithmetic operation.

Eager replay does not reconstruct compiled kernels or asynchronous global-RNG interleaving. Snapshots are diagnostic evidence, not exact training-resume files. Actor/critic snapshot attempts are bounded; alpha retains existing guard/telemetry but has no separate replay snapshot in this change.

## Verification

- Baseline:64 related tests passed before implementation.
- RED/GREEN observed for strict episode JSON, raw-gradient/norm overflow distinction, capture and storage isolation, BASE/sync/async integration, Torch2.5 fallback, upload isolation, early backup scheduling, repeated telemetry labels and original gradient payload.
- Fresh whole-change reviewer found two material issues (idempotence and missing raw tensor payload); both were reproduced with failing tests and fixed.
- Earlier full suite on `fd5d207`:959 passed,95 skipped,1408 subtests,8 warnings,523.18s. Final source `98724b8`: **959 passed,95 skipped,1408 subtests,8 warnings,525.47s; JUnit zero errors/failures**. Warnings are existing SDK deprecations and the deliberate pending-vector cleanup test.
- Compileall/diff checks pass. Credential scan covers the actual local key values; no values are included in the evidence or source.

## Operational state and next use

At 2026-10-28T06:52Z, MASAC BASE E4096 `4f98f2c9`, MASAC BASE E2048 `a04dbdef`, and MASAC PQL E2048 `8db188ea` were running on both W&B and Kaggle with fresh heartbeats. E4096 has a verified full backup at step10256384/episode4096; E2048 roots had no full backup yet. Preserve those lineages and recheck services before decisions.

Future launches should pin the verified fixed source. Existing failed source-pinned jobs cannot acquire missing replay/optimizer/RNG state retrospectively. Warm start from policy weights, if later requested, must be labeled as a new run rather than exact continuation.

Structured measured evidence: [2026-10-28-overflow-fix-verification.json](2026-10-28-overflow-fix-verification.json). Archived diagnostic recipes: [trained-model probe](2026-10-28-trained-overflow-probe.py), [reference-gradient probe](2026-10-28-overflow-reference-probe.py). Recipes expect the diagnostic spec, credentials environment and Kaggle runtime shown in the corresponding private kernel source; secrets are not embedded.


## Execution decisions

- Execute the proposal directly as requested; preserve the existing isolated working branch and push it under AGENTS.md. No main merge or 20k production relaunch is implied.
- Use bounded, explicitly labeled diagnostic kernels on verified idle accounts; preserve existing production accounts/lineages and report actual Kaggle-returned slugs. This costs a bounded amount of diagnostic GPU quota, not an account migration.
- Treat eager/async replay as diagnostic comparison, with explicit limitations, rather than exact reconstruction of the missing historical execution.
- Correct the FP32 counter and nested-report ambiguity discovered by live probes; keep the measured reference gradients rather than infer updates from unavailable AMP counters.

The fresh reviewer identified no other actionable findings; both material findings were fixed. No deferred reviewer minors.
