# MATD3 actor reproduction and authorized production restarts

Recorded 2026-10-28. Diagnostic and all three new roots pin `0042166fb0408d9efa487ad2ccd31c59b709cd69` on `feat/cpu-replay-20261022`. Runtime/config/source under `uav_network`, `scripts`, `configs` are unchanged from previously verified runtime `98724b827ce0dd017bb319dea09aa3c31b90282a`. This follow-up adds evidence, not a new algorithm or normalization layer.

## What was reproduced

Kaggle diagnostic `egkljwigojwof/diag-matd3-actor-late-1028-fc3fc4` completed. W&B diagnostic [aa6066de](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/aa6066de) finished. It loads the actual finite model from failed MATD3 BASE run `3befcfb8` (`failure_state.pt`) and collects new policy rollout data with the original late exploration step. No learning occurs during collection. The old optimizer and failed batch were not restored because they were not saved.

At rollout step 128, batch index 0, actor backward overflows at loss scale 32768 with both eager and compile-enabled probe models. All batch tensors are finite; observation/state absolute maxima are 1. The active fraction is 1, so inactive agents are not required for this reproduced event. The model is finite. The first eight attempts at rollout step 32 were finite; two attempts at step 128 skipped.

The eager backward-only `TorchDispatchMode` trace records the first nonfinite operation, with zero uninspected operations:

| Item | Shape | dtype | Finite | Absolute maximum among finite entries |
| --- | --- | --- | --- | --- |
| `aten.mm.default` input 1 | 3 x 8192 | FP16 | yes | 22.484375 |
| input 2 | 8192 x 256 | FP16 | yes | 30.125 |
| output weight gradient | 3 x 256 | FP16 | no, 16 nonfinite values | 63776 |

This is the matrix multiplication producing the actor continuous output head weight gradient. The first nonfinite parameter gradient in eager replay is `actors.1.continuous.weight`. Its FP32 reference maximum is 7.690993309; multiplied by the source scale 32768 it becomes 252018.46875, exceeding FP16's finite limit 65504. FP32 replay has finite loss and actor norm (reported mean 3.108659744). FP16 loss itself remains finite, about -136.18024; AMP skips the update and reduces scale from 32768 to 16384.

This establishes scaled-backward FP16 arithmetic overflow in the reproduced actor event. Finite inputs/model and a successful FP32 reference do not support blaming corrupt input data or prescribing a missing normalization layer. They also do not prove that every historical event had this exact first operation. Exact historical replay remains impossible without the lost batch. Compile-enabled reproduction is recorded separately; the arithmetic trace and reference replay are eager, not a reconstruction of compiled kernels or production asynchronous RNG interleaving.

## Applied behavior in the restarted runs

The existing fixed source preserves AMP skip/backoff and original pre-clipping gradient evidence. Shared telemetry encodes handled nonfinite numeric metrics as null plus flags, so strict JSON episode logging can proceed. First actor/critic overflow captures save batch/model/original gradients/scaler/optimizer/RNG context for subsequent replay. The first full recovery checkpoint is due by 1800 seconds; later spacing/cap remain unchanged. The recovery interval does not imply a checkpoint already exists at startup.

The scientific equations, learning rates, rewards, model architecture and HYBRID CPU75/GPU25 layout remain unchanged. This retry intentionally preserves FP16 and the original acceleration policy so the handling fix can be observed with the same scientific setup.

## Three fresh production roots

The user explicitly authorized these restarts after diagnosis. Original roots were failed/error, and each selected account had no active kernel before launch. The temporary actor diagnostic completed before account03 was reused. Credential identity/mount checks and available quota were verified; no credentials are included in this record.

| Case | Account | Old W&B run | New W&B run | Kaggle root |
| --- | --- | --- | --- | --- |
| MASAC PQL | account02 | `03df03cc` | [0650d5b1](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/0650d5b1) | `vuliu123456/gh-tra-g3-masac-pql-h75-1028080-7b56ff` |
| MATD3 BASE | account03 | `3befcfb8` | [44558e6c](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/44558e6c) | `egkljwigojwof/gh-tra-g3-matd3-base-h75-1028-fa6b65` |
| MATD3 PQL | account04 | `4f8f7d7d` | [87a9500e](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/87a9500e) | `gkgijewjf/gh-tra-g3-matd3-pql-h75-10280805-9535a7` |

All three start fresh from seed44, initial episode/global_step 0, with no warm-start artifact or optimizer/replay restoration. Missing old full-state checkpoints prevent exact continuation. They retain the canonical 20000-episode target, 4096 environments, batch8192, replay1M, CPU75/GPU25 and two Tesla T4s. Live W&B readback shows zero differences against each old run across `world`, `sensor`, `network`, `energy`, `reward`, `masac`, `matd3`, `pql` and the entire `runtime` section. Hardware is two T4s. MASAC PQL and MATD3 BASE compile models; MATD3 PQL keeps compile disabled, matching the old policy. FP16 remains enabled for all.

At 08:15:29-08:15:47 UTC all three Kaggle roots and W&B learners were running with fresh heartbeats:

| Case | global_step | train/updates | Actor applied / skipped | Critic applied / skipped |
| --- | ---: | ---: | ---: | ---: |
| MASAC PQL | 278528 | 272 | 264 / 0 | 263 / 2 |
| MATD3 BASE | 688128 | 1088 | 541 / 3 | 1084 / 4 |
| MATD3 PQL | 655360 | 1008 | 501 / 3 | 998 / 3 |

These summary fields arrive from different telemetry events, so they need not form an atomic learner snapshot. All three already captured real production gradient overflows on finite batches and then continued applying updates. MATD3 BASE saved `overflow-base-actor.pt/json` and `overflow-base-critic.pt/json`; MATD3 PQL saved `overflow-policy-actor.pt/json` and `overflow-critic-critic.pt/json`; MASAC PQL saved `overflow-critic-critic.pt/json`. The first captures show scale65536 ->32768 and category `nonfinite_gradients`. Their metadata are retained in the structured evidence. Unlike the lost old batch, these new production batches are now available for replay on W&B.

Sampling W&B history did not include the brief nonfinite-flag events; absence from that sample is not used as evidence of absence. Capture metadata, cumulative skip/applied counters and subsequent positive learner progress provide the evidence above. This verifies handling so far; the 20000-episode runs have not completed.

## Independent supervision and preserved runs

The launch manifests request `auto_continue=true`, `orchestration=github_actions`, `external_supervisor_required=true`, CPU visualization and final CPU evaluation30, seed2000000044, stride5, fps10. Hosted service probe [40653229682](https://github.com/truongdaoanhduy/uav_network/actions/runs/40653229682) authenticated the allowed Kaggle accounts, W&B and HF. Hosted non-dry supervisor [40654732673](https://github.com/truongdaoanhduy/uav_network/actions/runs/40654732673), job122702913311, reports all three exact new roots as `action=active/status=running` at08:09:44UTC. Its structured report is included in the evidence JSON.

That Actions invocation is nonzero because the three OLD roots still report `recovery_blocked/RecoveryUnavailable/checkpoint_missing`; it does not report a failure of the new roots. The supervisor remains active on main with accounts01,02,03,04,06,07 enabled. Continuation requires verified full-state checkpoints and available service/quota; the startup check is not a guarantee of uninterrupted execution.

Original healthy runs were retained without relaunch/cancel/config changes. Fresh 08:16:12-08:16:13UTC W&B and Kaggle readback:

- `4f98f2c9` MASAC BASE E4096: running, global_step18853888, updates36568.
- `a04dbdef` MASAC BASE E2048: running, global_step8783872, updates33800.
- `8db188ea` MASAC PQL E2048: running, global_step10096640, updates39184.

## Reproducibility and verification

- [Structured reproduction/restart evidence](2026-10-28-matd3-actor-reproduction-and-restarts.json), including effective config comparisons, actual snapshot metadata and supervisor report.
- [Unabridged diagnostic result](2026-10-28-matd3-actor-probe-result.json), including per-attempt metrics, input statistics and FP32/FP16 parameter-gradient comparisons.
- [Executed diagnostic driver](2026-10-28-matd3-actor-probe.py). It expects the Kaggle source checkout, two CUDA devices, authorized W&B credentials and `/kaggle/working/diagnostic-spec.json` with the public spec fields at the top of the raw result (`commit`, `entity`, `project`, `wandb_run_id`, `source_run_id`, `kernel_ref`, purpose/submission metadata). Use a new diagnostic run ID when rerunning. It creates new rollout batches; it cannot restore missing historical data.
- Previously completed full CPU suite on runtime98724b8:959passed,95skipped,1408subtests,8warnings in525.47s, JUnit0failures/0errors. That historical full-suite result was not rerun for this evidence-only follow-up.
- Fresh verification: recipe py_compile passed; project structure/scientific source checks 2passed +449subtests in0.22s; diff check and runtime/config equality against98724b8 passed; evidence consistency assertions passed; 12 credential values scanned across6changed files,0hits.

Next action: preserve all six active production roots and inspect current W&B/Kaggle state for future status requests. Investigate any future failure using the captured batch and exact source; do not duplicate roots, reset old histories, infer exact historical replay, or silently change scientific settings.
