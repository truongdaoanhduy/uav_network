# Training configuration comparison: UAV G3, FlashSAC, FastTD3

Comparison prepared 2026-10-26 (Asia/Ho_Chi_Minh). Read-only inspection of training source and existing G3 launch receipts; no live job, learner, environment, or scientific configuration was changed.

## Provenance

- UAV production source: `77a1d0f52ed4e78bc2823c3ada03d661568cd923`; configuration `configs/comparison-20k-hybrid75-1kb.yaml`, defaults `uav_network/configuration.py`, and effective eight-case W&B configuration captured at 2026-10-26T13:15:10.902765+00:00 in `outputs/robust-training-20261026/cohort/observed-latest.json`. This is configuration evidence, not a fresh claim of current run health.
- FlashSAC inspected upstream HEAD: `87edc9061150ae9e962dd84e6544e27a1554b3ab`. GPU/IsaacLab benchmark script `scripts/run_isaaclab.sh` overrides `configs/flashSAC_base.yaml` and `configs/agent/flashSAC.yaml`. Plain base defaults differ from the GPU benchmark.
- FastTD3 inspected upstream HEAD: `229ed59bbf43ea2f7a2d5d90d1076314839944d7`. Use `IsaacVelocityFlatG1Args` -> `IsaacLabArgs` -> `BaseArgs` in `fast_td3/hyperparams.py`, matching the README's standard FastTD3 IsaacLab example. The separate SimbaV2 example has additional overrides.

## Matched GPU-profile comparison

| Field | UAV G3 | FlashSAC IsaacLab script | FastTD3 Isaac-Velocity-Flat-G1-v0 |
| --- | --- | --- | --- |
| Algorithm | MASAC / MATD3; six-agent CTDE, hybrid continuous/discrete action | FlashSAC | FastTD3 |
| Parallel environments | 4096 | 1024 | 4096 |
| Logical sample batch | 8192 joint transitions | 2048 transitions | 32768 transitions |
| Replay capacity, all environments | 1,000,000 joint transitions | 10,000,000 transitions | 4096 * 10240 = 41,943,040 transitions |
| Replay location | HYBRID: CPU75%/GPU25%; FULLGPU: learner CUDA | CUDA | CUDA |
| Critic updates per eligible vector collection | BASE 8; PQL asynchronous target ratio 8 | 2 | 4 (this G1 task; BaseArgs is 2) |
| Learning start threshold | 131072 joint transitions = 32 full vector collections | 100000 stored transitions; n-step filling also matters | learning_starts=10 in vector iterations; code tests global_step > threshold |
| Gamma | 0.99 | 0.99 | 0.99 |
| Target tau | 0.005 | 0.01 | 0.1 |
| Learning rate | 3e-4 fixed | 3e-4 -> 1.5e-4 cosine schedule | actor/critic 3e-4; same end value by default |
| Return horizon | 1-step target | n_step=3 | num_steps=8 |
| Actor update frequency (UAV BASE) | MASAC every critic update; MATD3 every 2 | every 2 | every 2 |
| Network configuration | actor/critic hidden_sizes=[256,256], plain ReLU MLP | actor hidden_dim128/2blocks; critic hidden_dim256/2blocks | actor_hidden_dim512; critic_hidden_dim1024; architecture differs from UAV MLP |
| Critic output | scalar twin Q | categorical distribution, 101 bins | categorical distribution, 251 atoms for IsaacLab |
| Reward normalization | no adaptive reward-normalizer stage in inspected learners | enabled, normalized_G_max=5 | disabled by default in this profile |
| AMP | off | on (FP16 GradScaler) | on, bf16 default |
| torch.compile | off | on, auto compile mode | on, reduce-overhead default |
| Optimizer | Adam, fused_optimizers=false | Adam with CUDA fused enabled | AdamW, weight_decay=0.1 |
| Training budget | 20000 completed episodes across vector environments; horizon1800 | 50000896 aggregate environment steps | 50000 vector steps = 204800000 environment transitions |
| Train seed | 44 in all eight G3 cases | script runs 0,1000,2000,3000,4000 | README example seed1; CLI adjustable |

The eight G3 cases are algorithm/scheduler/replay-layout comparisons, not eight independent training seeds. Native/CUDA scenario seed scans are generator validation, not multi-seed learning-curve replication.

## Unit checks and interpretation

- UAV replay allocates `(capacity, ...joint_transition_shape)`; its capacity is shared across environments, not per UAV or per vector environment.
- FlashSAC TorchUniformBuffer allocates `(max_length, ...)`; the benchmark script overrides the agent's base replay 1M to 10M and its warmup 10K to 100K.
- FastTD3 SimpleReplayBuffer allocates `(n_env, buffer_size, ...)`; the 10240 IsaacLab setting is per environment. Do not compare 10240 directly with UAV's total 1M.
- At full vector width, sampled replay rows per newly collected row = updates * batch / num_envs: UAV BASE16, FlashSAC4, FastTD3 G1 32. These are row-reuse calculations, not a claim that the algorithms have equal compute cost or learning quality. Each UAV row covers six agents and centralized critic data. Asynchronous PQL's realized rate and independent policy-worker update frequency must be measured separately.
- UAV critic_chunk_size1024 bounds activation memory while preserving the logical8192 batch, loss normalization and optimizer semantics. It does not change batch_size to1024.
- A 20000-episode budget and a 20000-vector-step budget have different units. Do not infer equal training budgets from the same numeral.
- FastTD3 README says its defaults were optimized on one A10080GB. G3 launches use two T4s with separate device allocations. The configuration difference alone does not establish which setup learns faster or better.
- GPU-parallel off-policy collection is a shared systems direction. Current UAV learners do not implement FlashSAC/FastTD3's complete algorithm, architecture, normalization, return horizon or optimizations. Copying their numerical hyperparameters alone would not reproduce their methods.

## Primary source links

- [UAV production configuration](https://github.com/truongdaoanhduy/uav_network/blob/77a1d0f52ed4e78bc2823c3ada03d661568cd923/configs/comparison-20k-hybrid75-1kb.yaml)
- [UAV defaults](https://github.com/truongdaoanhduy/uav_network/blob/77a1d0f52ed4e78bc2823c3ada03d661568cd923/uav_network/configuration.py)
- [FlashSAC GPU benchmark launch](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/scripts/run_isaaclab.sh)
- [FlashSAC agent configuration](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/configs/agent/flashSAC.yaml)
- [FlashSAC replay](https://github.com/Holiday-Robot/FlashSAC/blob/87edc9061150ae9e962dd84e6544e27a1554b3ab/flash_rl/buffers/torch_buffer.py)
- [FastTD3 task configuration](https://github.com/younggyoseo/FastTD3/blob/229ed59bbf43ea2f7a2d5d90d1076314839944d7/fast_td3/hyperparams.py)
- [FastTD3 replay](https://github.com/younggyoseo/FastTD3/blob/229ed59bbf43ea2f7a2d5d90d1076314839944d7/fast_td3/fast_td3_utils.py)
- [FastTD3 training loop](https://github.com/younggyoseo/FastTD3/blob/229ed59bbf43ea2f7a2d5d90d1076314839944d7/fast_td3/train.py)

Validation: primary configs and effective G3 receipts inspected; replay allocation dimensions and step/update loop units checked in source. No new source-code tests were needed for this documentation-only comparison. Existing G3 training and continuation remain unchanged. Next action requires a new user request; no retuning, cancellation or duplicate launch is queued.

## AMP / torch.compile recommendation

Follow-up 2026-10-26: both are supported experiment options, not measured G3 production improvements. Prioritize an isolated AMP FP16 comparison on T4; retain torch.compile only if a matched benchmark improves end-to-end performance without unacceptable memory or numerical changes. No benchmark or job mutation was requested or performed for this advice.

- Source inspection: algorithms/precision.py already uses autocast FP16 and learner-owned GradScaler groups; models, optimizer state and replay stay FP32. AMP can reduce selected activation/computation costs, but does not halve the existing replay allocation or guarantee equal FP32 learning trajectories.
- algorithms/optimization.py::compile_models compiles the whole vmap critic boundary plus actor encoders with Inductor/dynamic=True and the default mode. It does not compile the Warp packet environment or the entire pipeline. G3 critic activation recomputation/chunking must be included in any new benchmark.
- PyTorch2.11 documentation: reduce-overhead mode uses CUDA graphs and can cache extra workspace memory; max-autotune enables CUDA graphs by default. Start by measuring the existing default-mode implementation rather than copying another repository's compile mode. Compilation/warmup time and steady-state speed must be reported separately.
- Proposed matched matrix: off/off, FP16/off, off/compile, FP16/compile. Preserve env4096/batch8192/replay1M and scientific configuration; cover MASAC/MATD3 BASE/PQL and the relevant replay layouts. Measure successful applied optimizer updates per second, aggregate collection throughput, CPU/GPU peaks, numerical health/overflow-skip rate and complete checkpoint+GradScaler resume. Compare learning evaluation on matched training seeds separately from short throughput tests.
- Current fixed/PQL resume guards compare amp_mode and compile_models against the saved configuration. An off/off G3 checkpoint is not an authorized or compatible exact continuation with changed flags; use a separately named experiment. Existing G3 cohort/controller settings remain intact.

Official references checked with Context7 and primary documentation:
- https://docs.pytorch.org/docs/2.11/amp.html
- https://docs.pytorch.org/docs/2.11/generated/torch.compile.html
- https://www.nvidia.com/en-us/data-center/tesla-t4/
