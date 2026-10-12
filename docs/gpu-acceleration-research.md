# GPU acceleration for the real UAV packet environment

Primary-source review: 2026-10-12. These are compatibility findings and a
benchmark plan; throughput has not been measured on the user's workstation.
Library APIs and hardware support should be pinned to the versions actually
tested, rather than inferred from the current documentation.

## Recommended architecture

Run independent swarm scenarios in **spawned CPU processes**, with one live
UavNetSim instance in each process. Keep a single PyTorch learner on the GPU,
batch policy inference across scenarios and UAVs, and sample joint transitions
from replay for MASAC or MATD3 updates. This preserves the packet model while
using CPU cores for collection and CUDA for neural-network work.

This follows two concrete constraints: [SimPy processes events sequentially][simpy],
and the [local network bridge][bridge] owns process-global upstream configuration
and rejects multiple live networks in one process. The [upstream adapter][upstream]
pins UavNetSim to `04daafb815eb377409b40b285574eeb62b9a8d58` and advances a
`simpy.Environment`. Moving the actor to CUDA does not move that event loop.

The first implementation should use a small process vector runner or a
Gymnasium adapter plus `AsyncVectorEnv`. Add TorchRL only if measured collection
latency or replay/collector maintenance justifies the integration. Keep eager
FP32 training as the comparison baseline; enable compilation after profiling.
These choices are engineering recommendations derived from the sources below.

## Options and compatibility

| Option | Compatibility with this project | Decision |
| --- | --- | --- |
| PyTorch CUDA | Accelerates tensor inference, critics, gradients and optimizers; existing custom hybrid losses remain usable. [Tuning guide][torch-tuning] | Use one CUDA learner and batched inference. |
| Gymnasium `AsyncVectorEnv` | Processes plus pipes; standard observation spaces support shared memory. Requires adapting the current custom `Action`/environment interface. [API][gym-vector] | Good initial collection layer. |
| TorchRL collectors and replay | Supports synchronous/asynchronous off-policy collection, policy weight synchronization and replay integration. [Collectors][torchrl-collectors] | Useful next step; custom hybrid multi-agent losses still need an adapter. |
| EnvPool | Custom runtime environments must be native C++; its authoring checklist excludes embedding/calling the official Python environment. [Extension guide][envpool] | Requires rewriting and validating the SimPy packet model; not a wrapper speedup. |
| Sample Factory | Efficient PPO/APPO, custom multi-agent environments and hybrid actions are supported. [Overview][samplefactory] [Parameters][samplefactory-params] | Suitable for a separate PPO comparison; not a drop-in MASAC/MATD3 learner. |
| Isaac Lab | GPU-parallel robot-learning framework; current development docs describe Newton/Warp and PhysX backends. [Ecosystem][isaaclab] | A new simulation implementation requiring model validation. |
| NVIDIA Warp | JIT-compiles explicitly authored, statically typed CPU/CUDA kernels; host Python and kernel scope differ. [Programming model][warp] | Possible later motion/sensing kernels; does not automatically execute SimPy events on CUDA. |

The inspected [Sample Factory setup.py][samplefactory-setup] identifies version
2.1.3 and requires `numpy<2.0` and `gymnasium<1.0`. This conflicts with the
project's NumPy 2.x requirement and the Gymnasium 1.1+ autoreset API described
below. Use an isolated dependency set for that comparison.

Isaac Lab/Warp and EnvPool benchmark claims must not be transferred to this
environment. A rewritten packet simulator must reproduce queues, MAC contention,
retries, collisions, packet deadlines, fragmentation/custody and event ordering
before it can serve as the same experimental model. Keeping SimPy on CPU while
moving small geometry arrays to GPU may add transfer overhead; measure first.

## Identify the actual GPU

"RTX 6000" does not identify one architecture or memory capacity:

| Exact product | Architecture | Advertised memory | NVIDIA source |
| --- | --- | --- | --- |
| Quadro RTX 6000 | Turing | 24 GB GDDR6 | [Data sheet][quadro] |
| RTX A6000 | Ampere | 48 GB GDDR6 ECC | [Product page][a6000] |
| RTX 6000 Ada Generation | Ada Lovelace | 48 GB GDDR6 ECC | [Product page][ada6000] |
| RTX PRO 6000 Blackwell Workstation Edition | Blackwell | 96 GB GDDR7 ECC | [Product page][blackwell6000] |

Record the actual model, driver, available VRAM, CUDA build, PyTorch version,
CPU model/core availability, RAM and operating system before choosing worker
count or GPU replay capacity. For example, inspect `nvidia-smi` and
`torch.cuda.get_device_properties(0)`. Neither memory capacity nor supported
mixed-precision modes should be assumed from the short product name.

## CPU process vectorization contract

1. **One worker is one complete swarm**, containing all six UAVs and its GCS.
   Parallelize independent scenarios, not the six interacting agents inside
   one packet event queue. Give each worker its own scenario seed and simulator.
2. Use `context="spawn"`, an import-safe entry point and worker factories that
   create/reset their own simulator. CUDA is not fork-safe after initialization;
   PyTorch documents `spawn`/`forkserver` for CUDA subprocesses. Keep GPU tensors
   and the learner in the parent. [Multiprocessing guidance][torch-mp]
3. With Gymnasium, describe observations as standard `Box`/`Dict` spaces, e.g.
   local observations `[6, O]` and centralized state `[S]`. Include state in the
   fixed-shape observation payload to share it alongside local observations;
   passing it through `info` or repeated `call("state")` uses separate pipe work.
   Adapt actions to `Box([6, 3])` plus `MultiDiscrete([7] * 6)`, then reconstruct
   the existing `Action`. `shared_memory=True` shares observations, not all
   actions/metadata. Start with `copy=True` to avoid replay aliasing. [API][gym-vector]
4. Set `autoreset_mode=AutoresetMode.DISABLED`. Store each returned terminal
   observation **and centralized next state in replay before reset**. Then reset
   only `terminated | truncated` slots with
   `reset(options={"reset_mask": mask})`; the mask must be a Boolean NumPy array
   of shape `[num_envs]` and contain at least one `True`. Do not reset continuing
   slots or replace terminal next states with a new episode. [Autoreset guide][gym-autoreset]
   [Implementation][gym-source]
5. Retain both episode-end flags. Use `terminated` for the Bellman bootstrap
   mask when the step limit is an external truncation. If the finite horizon is
   part of the task itself, time remaining belongs in the observation and its
   end is a task termination. [Time-limit semantics][gym-time-limits]
6. Count collected **joint environment transitions**. A vector step with `E`
   scenarios contributes `E` transitions, each containing six agents. Preserve
   learning-start thresholds and the configured update-to-transition ratio;
   neither multiply nor divide learner updates accidentally when increasing `E`.

`AsyncVectorEnv` parallelizes execution but a normal `step()` still waits for
all workers. Its `step_async()`/`step_wait()` split can overlap outstanding CPU
steps with learner updates on previously collected replay. Packet-heavy scenarios
may create slow-worker barriers; asynchronous collectors are a later option.
The [source][gym-source] implements the waits and shared-buffer writes explicitly.

## CUDA learner, replay and compilation

Batch the homogeneous shared actor over `[E * 6, O]`, then reshape actions by
scenario; do not issue one GPU forward pass per UAV. Keep critics, target
networks and optimizer state on CUDA. Start with contiguous CPU replay and
transfer sampled batches together. Consider pinned staging/prefetching or GPU
replay only after measuring memory and transfer costs. [PyTorch's transfer
guide][torch-transfers] documents that pinning in the hot path can itself be
costly and that asynchronous copies require correct data lifetime/synchronization.

Action readback to the CPU simulator is necessary, but it should occur once per
batch. Avoid converting every loss to Python with `.item()` inside each update;
reduce metrics on device and read them at the logging interval. Such operations
introduce CPU/GPU synchronization. [Tuning guide][torch-tuning]

Compile actor/critic tensor functions first, using stable minibatch shapes.
Do not include simulator stepping, reset handling, NumPy conversions or logging
inside the compiled region. Tensor-dependent Python branching and `.item()`
cause graph breaks; guard changes can trigger recompilation. [Graph breaks][torch-breaks]
[`torch.compile` reference][torch-compile] The MATD3 delayed-update branch can
remain outside separately compiled tensor paths.

Compare eager with `mode="default"` before trying `reduce-overhead`. The latter
uses CUDA graphs, may consume more workspace memory and is not applicable to
every graph; no particular speedup is guaranteed. Separate cold compilation
time from warmed execution. [API][torch-compile] [End-to-end tutorial][torch-compile-example]
Compiler backends can change numerics and RNG behavior relative to eager;
`fallback_random` is a debugging option, not a general correctness guarantee.
[Compiler FAQ][torch-compile-faq] Keep compilation optional and check losses,
gradients and learning curves against eager execution.

## Determinism, battery and MATD3

Seed Python, NumPy generators, PyTorch and each worker's UavNetSim instance.
Use explicit per-worker/per-episode seeds and log them; process start and arrival
order must not determine scenario identity. Keep the worker count and collection
schedule fixed for reproducibility comparisons. Asynchronous collection changes
sample arrival and policy age even when simulator trajectories are individually
deterministic. [TorchRL collection guidance][torchrl-collectors]

Preserve the existing deterministic baseline. `torch.use_deterministic_algorithms`
selects deterministic implementations or raises when unavailable; it does not
make an entire application reproducible. [API][torch-deterministic] NVIDIA
documents `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` as reproducibility controls
for concurrent cuBLAS streams. [cuBLAS][cublas] Set applicable configuration before
CUDA work starts. Record precision settings and dependency versions. PyTorch
does not promise identical results across CPU/GPU, releases or platforms, and
deterministic implementations can be slower. [Reproducibility][torch-randomness]

MATD3's algorithmic basis is twin centralized critics, target smoothing and
delayed policy/target updates, extending TD3 to decentralized execution.
[Original TD3 paper][td3] [Original MATD3 paper][matd3] The MATD3 particle-task
experiments use Gumbel-Softmax, so a categorical next-hop branch is compatible
with a documented estimator approach. The exact hybrid movement/power/destination
loss remains a project adaptation: specify categorical exploration/gradient
estimation and apply Gaussian target noise only to continuous controls.
Adding battery state also changes the task: include remaining energy in local
and centralized observations, reset it per episode, account for each drain once,
and specify whether depletion terminates the team or disables individual UAVs.
These adaptation decisions require their own correctness checks.

## When TorchRL is worthwhile

Current [collector documentation][torchrl-collectors] supports
`Collector(num_collectors=..., sync=False)` for first-available off-policy batches
and background replay population with `start()`/`async_shutdown()`.
`AsyncBatchedCollector` also describes centrally batched inference with process
environment workers. This can overlap simulation and learning and reduce
slow-worker barriers, at the cost of policy lag and collection-order variation.
Pin the selected TorchRL release: current class names differ from older examples
using `SyncDataCollector`/`MultiaSyncDataCollector`.

Use an `EnvBase`/TensorDict adapter with explicit agent, environment and time
dimensions; retain state and episode-end semantics. TorchRL's [multi-agent DDPG
tutorial][torchrl-marl] is a useful layout example, not a ready implementation of
this hybrid learner. Its [ReplayBuffer API][torchrl-replay] offers sample pinning
and prefetching. Do not add a second collector/replay architecture until the
simpler runner demonstrates a measurable limit.

## Benchmark acceptance plan

- Sweep 1, 2, 4, 8 workers up to available CPU capacity, reserving CPU time for
  the learner/driver. Limit worker BLAS/OpenMP/PyTorch threads initially to one;
  measure oversubscription instead of assuming more workers help.
- Keep the real network enabled with identical scene and packet configuration.
  Separate environment-only fixed-action throughput from full training.
  Report joint transitions/s, learner updates/s, time-to-fixed-transition-budget,
  reset latency, CPU utilization, GPU utilization and peak RAM/VRAM.
- Profile simulator step, inference, action transfer, replay sampling, learner
  update and slow-worker waits. Warm CUDA and compilation separately. CUDA is
  asynchronous: use CUDA events or synchronized timing; PyTorch's benchmark
  utility performs warmup and accelerator synchronization. [Benchmark guide][torch-benchmark]
- Compare scalar and process rollouts with identical seeds/actions, including
  terminal observations/state, battery, packet counters and clock advancement.
  Exercise partial resets, worker failure cleanup and no cross-episode replay.
- Compare algorithms and speed modes at the same collected-transition budget
  and update ratio, then evaluate multiple held-out seeds: delivery/search rate,
  mission success, return, packet delay/PDR and energy. Report wall time as a
  second axis. A faster loop alone does not establish better sample efficiency.

[bridge]: ../uav_network/communication/backend.py
[upstream]: ../uav_network/communication/upstream.py
[simpy]: https://simpy.readthedocs.io/en/latest/topical_guides/time_and_scheduling.html
[gym-vector]: https://gymnasium.farama.org/api/vector/async_vector_env/
[gym-source]: https://gymnasium.farama.org/_modules/gymnasium/vector/async_vector_env/
[gym-autoreset]: https://farama.org/Vector-Autoreset-Mode
[gym-time-limits]: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
[torch-mp]: https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html
[torch-tuning]: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
[torch-transfers]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
[torch-compile]: https://docs.pytorch.org/docs/stable/generated/torch.compile
[torch-breaks]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.common_graph_breaks.html
[torch-compile-example]: https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
[torch-compile-faq]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html
[torch-deterministic]: https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
[torch-randomness]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[cublas]: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
[torch-benchmark]: https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
[torchrl-collectors]: https://docs.pytorch.org/rl/stable/reference/collectors_single.html
[torchrl-replay]: https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html
[torchrl-marl]: https://docs.pytorch.org/rl/stable/tutorials/multiagent_competitive_ddpg.html
[envpool]: https://envpool.readthedocs.io/en/stable/content/new_env.html
[samplefactory]: https://www.samplefactory.dev/
[samplefactory-params]: https://www.samplefactory.dev/02-configuration/cfg-params/
[samplefactory-setup]: https://github.com/alex-petrenko/sample-factory/blob/master/setup.py
[isaaclab]: https://isaac-sim.github.io/IsaacLab/develop/source/setup/ecosystem.html
[warp]: https://nvidia.github.io/warp/latest/user_guide/programming_model.html
[td3]: https://proceedings.mlr.press/v80/fujimoto18a.html
[matd3]: https://arxiv.org/pdf/1910.01465
[quadro]: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-6000-us-nvidia-704093-r4-web.pdf
[a6000]: https://www.nvidia.com/en-us/products/workstations/rtx-a6000/
[ada6000]: https://www.nvidia.com/en-gb/products/workstations/rtx-6000/
[blackwell6000]: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
