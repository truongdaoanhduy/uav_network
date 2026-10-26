# Task 1 — BASE CUDA verification for AMP FP16 + torch.compile

Date: 2026-10-26 (Asia/Ho_Chi_Minh)

## Scope

Bounded real-CUDA verification only for BASE MASAC and BASE MATD3. No PQL, no 20k production training, no restart/cancel/mutation of the existing eight G3 lineages.

Source commit: `eab048f94f5c85d5a1a1cb60b4f0fcd772ccfef4`
Kaggle kernel: `kjhihgiugy/ben-g3-both-base-h75-10261650-bbce3e`
Account alias: `account_07`
Stage: `benchmark`
Profile: `configs/comparison-20k-hybrid75-1kb.yaml`
Requested accelerator: `NvidiaTeslaT4`, `gpus_per_run=2`

Resolved production-shape settings under test:
- `runtime.amp_mode=fp16`
- `runtime.compile_models=true`
- `runtime.fused_optimizers=false`
- envs=4096
- batch=8192
- replay capacity=1,000,000
- hybrid replay CPU75/GPU25
- report/payload=1 KiB

## Hardware proof

Downloaded `hardware.json` reports PyTorch 2.11.0+cu128 with two Tesla T4 devices, each 14.562 GiB, compute capability 7.5. CUDA is available. Warp reports peer access supported fully.

## Execution-check result

### MASAC BASE

- status: `passed`
- `amp_autocast_fp16=1`
- critic updates: 32; actor updates: 32
- AMP critic/actor/alpha attempted=applied=32; skipped=0
- scaler remains finite at 65536
- compiled unique graphs: 3
- Inductor generated kernels: 88
- Dynamo graph breaks: 0
- Dynamo frames fallback: 0
- peak allocated: GPU0 0.076 GiB; GPU1 3.590 GiB
- peak reserved: GPU0 0.090 GiB; GPU1 3.826 GiB
- bounded execution-check runtime: 56.84 s

### MATD3 BASE

- status: `passed`
- `amp_autocast_fp16=1`
- critic updates: 32; actor updates: 16 (policy delay preserved)
- AMP critic attempted=applied=32; actor attempted=applied=16; skipped=0
- scaler remains finite at 65536
- compiled unique graphs: 3
- Inductor generated kernels: 88
- Dynamo graph breaks: 0
- Dynamo frames fallback: 0
- peak allocated: GPU0 0.076 GiB; GPU1 3.618 GiB
- peak reserved: GPU0 0.086 GiB; GPU1 3.764 GiB
- bounded execution-check runtime: 36.23 s

## Bounded benchmark observations

These are standalone AMP+compile measurements, not a matched speedup claim against off/off.

- MASAC: 4260.47 joint transitions/s; warmed learner 2.141 updates/s; compile warmup 8.853 s.
- MATD3: 4276.99 joint transitions/s; warmed learner 5.533 updates/s; compile warmup 7.995 s.

The generic benchmark fields `cuda_pipeline_verified=false` and `dual_gpu_verified=false` are not execution failures for this hybrid profile. `benchmark.py` defines those flags as true only when replay is also `cuda:1`; the production profile intentionally reports replay device `cpu` for CPU75/GPU25 hybrid storage. The dedicated execution-check is the authoritative AMP/compile gate and passed for both algorithms.

## Additional CUDA gates

- `test_cuda.py`: 11 tests passed.
- `test_cuda_scenario_v2.py`: 4 tests passed.
- `test_training_audit_regressions.py`: 8 tests passed.

No OOM, non-finite parameter failure, GradScaler skip, compile graph fallback, or graph break was recorded in the bounded BASE execution checks.

## Retrieved output hashes

- hardware.json: `c8acbe15e3b1bc238b9dbbc309aa44d2367d49415f8f899beab39e98258658f2`
- masac-execution-check.json: `3901ee4ae50c51ca6f413b8db00348cd43637f1049bbc0681d2a37c57b8fce7f`
- matd3-execution-check.json: `fb45e1b1c2e92deb1546eabd69e8661dd7abdd23a305028681ba3e93bb0887c9`
- masac-benchmark.json: `c7dea88cce4687633a42fcb11fded73169ea68ebf963bf84d4ba8304b99efe7b`
- matd3-benchmark.json: `4eae30dac2ef2d9474e32f9ce11186397f5ba026b5624011d034cf56a4ef945c`
- job-result.json: `f9e859fba84072d8e69327f81fec5db7cd580bfc1f2d101dff818fcb5eb1f122`

## Boundary / next task

Task 1 is BASE-only. PQL AMP+compile verification is deliberately not started here. A matched off/off vs AMP/compile performance comparison is also not part of this task, so no speedup/VRAM-reduction claim is made.
