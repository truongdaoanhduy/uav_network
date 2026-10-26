# Task 2 — PQL CUDA verification for AMP FP16 + torch.compile

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Bounded real-CUDA verification only for PQL MASAC and PQL MATD3. No 20k production training and no restart/cancel/mutation of existing G3 lineages.

Source commit: `0b73806cc9b97e1addf0b1b8faa78dff0cbb9bd8`
Kaggle kernel: `kjhihgiugy/ben-g3-both-pql-h75-10261706-4de452`
Account alias: `account_07`
Stage: `benchmark`
W&B benchmark runs: MASAC `ca285f70`; MATD3 `905012a9`
Profile: `configs/comparison-20k-hybrid75-1kb.yaml` + `pql.enabled=true`

Resolved settings under test:
- PQL async enabled
- AMP FP16 ON
- `torch.compile` ON
- fused optimizers OFF
- envs=4096
- batch=8192
- replay capacity=1,000,000
- hybrid replay CPU75/GPU25
- report/payload=1 KiB
- generation v3

## Hardware

Downloaded `hardware.json`: PyTorch 2.11.0+cu128, CUDA available, two Tesla T4 devices, 14.562 GiB each.

## Dedicated execution-check

### MASAC PQL

- status `passed`, `pql=true`
- AMP FP16 autocast marker=1
- critic 32 attempted/applied, 0 skipped
- actor 32 attempted/applied, 0 skipped
- alpha 32 attempted/applied, 0 skipped
- compile: 3 unique graphs, 88 generated kernels
- Dynamo graph breaks=0, frame fallback=0
- devices: collector/world cuda:0; learner/replay hot tier cuda:1
- hybrid replay: CPU 9.156 GiB + GPU hot 3.049 GiB
- peak allocated: GPU0 0.275 GiB; GPU1 3.962 GiB
- bounded execution-check time 48.361 s

### MATD3 PQL

- status `passed`, `pql=true`
- AMP FP16 autocast marker=1
- critic 32 attempted/applied, 0 skipped
- actor 16 attempted/applied, 0 skipped; policy-delay behavior preserved
- compile: 3 unique graphs, 88 generated kernels
- Dynamo graph breaks=0, frame fallback=0
- devices: collector/world cuda:0; learner/replay hot tier cuda:1
- hybrid replay: CPU 9.156 GiB + GPU hot 3.049 GiB
- peak allocated: GPU0 0.276 GiB; GPU1 3.843 GiB
- bounded execution-check time 30.225 s

## Actual async PQL benchmark

Both wrapper and inner benchmark reports are `passed`; both report `cuda_pipeline_verified=true` with collector cuda:0, critic cuda:1 and policy cuda:1.

### MASAC

- 40 collection steps, 72 critic updates, 72 actor updates
- 2459.41 joint transitions/s
- 1.081 critic updates/s, 1.081 actor updates/s
- elapsed 66.618 s
- peak allocated GPU0/GPU1: 0.266 / 3.954 GiB
- final AMP metrics: critic 65/65 applied, actor 72/72, alpha 72/72, all skipped=0, scales=65536, autocast marker=1
- reported learning metrics are finite (critic loss 318.3945, actor loss -5.0401, q mean 4.4403, target mean 4.3955)

### MATD3

- 40 collection steps, 72 critic updates, 36 actor updates
- 3127.85 joint transitions/s
- 1.375 critic updates/s, 0.687 actor updates/s
- elapsed 52.381 s
- peak allocated GPU0/GPU1: 0.276 / 3.843 GiB
- final AMP metrics: critic 65/65 applied, actor 36/36, all skipped=0, scales=65536, autocast marker=1
- reported learning metrics are finite (critic loss 293.6984, actor loss -5.0373, q mean 4.0679, target mean 3.6854)

`target_t4_scenario_verified=false` is not a failure for the current G3 profile: the helper still hard-codes the older target tuple `(6 UAV,25 targets,15 obstacles,1000 steps,E4096,B8192,R500k)`. This run intentionally uses G3 `(6,35,25,1800,E4096,B8192,R1M)`. The report's actual CUDA gate `cuda_pipeline_verified` is true for both algorithms.

## Additional gates

- `test_cuda.py`: 11 tests passed.
- `test_cuda_scenario_v2.py`: 4 tests passed.
- `test_training_audit_regressions.py`: 8 tests passed.
- `test_pql_training.py`: 11 tests passed.

No OOM, non-finite failure, AMP step skip, Dynamo graph break or compile frame fallback was recorded.

## Retrieved output hashes

- hardware.json: `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`
- masac-execution-check.json: `9a5dbbf5e5c39fb19c4844c7c614ad3a434fdc194ff939ade4b3518abec7f70b`
- matd3-execution-check.json: `1fff660d34358359f1d7761e395b049b5792a0785c6a2b60944de47fce2d5edc`
- masac-benchmark.json: `6d02e583763d14e2dd8b7a24eb912d78316bce4ca3dc1c9bcc84c81ecd355803`
- matd3-benchmark.json: `abfabd76340f97589debef995e4905e6eec9db0c1c445f9e7c02bde51caf58b2`
- job-result.json: `f2fc313737b27c28121c1fe2578cb969cbf1142bd55d010b3e73651c8bb6e19d`
- pql-smoke-tests.log: `8e16fb2bbe24e8eb5bd8cf6416588ea144289ea9e80d7336f9fbfe87f869b1bc`

## Boundary / next task

Task 2 proves functional/numerical execution for PQL with AMP+compile, but it does not prove acceleration versus off/off. Task 3 is a matched BASE 4-cell benchmark (off/off, FP16/off, off/compile, FP16/compile) before making speed/VRAM claims.
