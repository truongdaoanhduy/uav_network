# MATD3 Speed Audit — W&B / Hugging Face — 23 November 2026

**Scope:** Direct W&B online API reads of three completed runs in `truongdaoanhduy-uit/uav_search_target`, corresponding Hugging Face forensic archives plus *immutable* Actor checkpoint manifests; read-only collection. **Do not modify existing model/checkpoints.** New `FASTGRID` ablation is isolated in `experiment/map800-posterioronly-speed-timing-grid-only-20261123` and needs real T4×2 benchmarking before making claims about training GPU throughput.

**Reproducible source references:**
- Newest W&B: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/602a1e82
- V9 W&B: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e079eace
- Original W&B: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef
- Newest HF archive: https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/602a1e82/diagnostic-archives
- V9 HF archive: https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/e079eace/diagnostic-archives
- Original HF archive: https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/18d756ef/diagnostic-archives
- Immutable Actor snapshots: https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/deferred-gpu-eval
- Official PyTorch CUDA async transfers: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- Official PyTorch torch.compile profiling: https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html

## 1. The three runs — actual W&B metrics

| Measure | 18d756ef baseline | e079eace NoOracle V9 | 602a1e82 Posterior-only MAP800 TX08 |
|---|---:|---:|---:|
| W&B state | finished | finished | finished |
| Completed episodes | 20,000 | 20,000 | 20,000 |
| Global environment transitions | 27,549,696 | 29,392,896 | 30,720,000 |
| `train/joint_transitions_per_s` final | **4,599.758** | **1,981.912** | **2,493.174** |
| W&B `_runtime`, all phases | **8,577s / 2.38 h** | **38,744s / 10.76h** | **18,931s / 5.26h** |
| Estimated train-only time = transitions/final rate | **5,989s / 1.66h** | **14,831s / 4.12h** | **12,322s / 3.42h** |
| MATD3 hidden | 256×512 | 256×512 | 256×256 |
| `torch.compile` | False | False | True |
| E / U / B | 4096 / 4 / 8192 | 4096 / 4 / 8192 | 4096 / 4 / 8192 |
| Map size / belief cell | 1000m / 25m | 1000m / 25m | 800m / 20m |
| Target detection | Historical visible-truth/fine-altitude MDP | NoOracle V9 | Positive Bayes posterior≥.99, no LR/50m gate |
| TX max, learned | .4 W | .4 W | .8 W |
| W&B GPU-validated Actor snapshots | 70 | 74 | **254** |
| Validation episodes/policy | 100 | 100 | 100 |
| Held-out Final Test episodes | 4000 | 4000 | 1000 |

**Training speed rank** (same E/U/B, but not same MDP): baseline first 4,600 > new 2,493 > NoOracle V9 1,982 environment transitions/s. New is **1.84× slower than baseline** and **25.8% faster than V9**; `torch.compile`, observation size, reward or model size cannot individually be credited or blamed in a mixed scenario comparison. The train-only times above are **estimates from terminal average throughput**, not an exact timestamped phase segmentation. W&B `_runtime` includes fullflow validation/test/upload and must never be mistaken for train-only.

## 2. Precise slowdown evidence from source, not guesses

W&B `602a1e82` training summary:

- `timing/collector_wait_seconds = 6,881.264` (host wall waiting for Warp vector collector, including synchronization/queue delays)
- `timing/learner_seconds = 3,034.034`
- `timing/replay_sample_host_seconds = 1,996.615` (**65.8% of learner-side host seconds**)
- `timing/optimizer_update_host_seconds = 1,023.075` (**33.7% of learner-side host seconds**)
- `train/updates = 29,876`; `runtime.replay_cpu_fraction = .75` (CPU 750k transitions / learner GPU 250k). `HybridMeasuredReplay.sample` samples global indices and constructs each minibatch field by `cpu_data[name].index_select(...)` → `.pin_memory().to(cuda, non_blocking=True)` and GPU hot-tier gather → GPU scatter. Potential bottlenecks are per-batch CPU gather, pinned allocations, Host→Device transfer, GPU synchronizations. The instrumented host durations **are not guaranteed exclusive** because collector and learner overlap via async `step_async()/step_wait()` and CUDA ops may be queued. The replay time strongly motivates microbenchmarking replay tier and transfer path, not changing batch/U or ratio blindly.
- W&B GPU system history median `system.gpu.0.gpu = 100%` for new and V9 in sampled training window, baseline about 71.5% (sampling/physical GPU mapping may differ); Warp collector/kernel path has little idle time. Changing `torch.compile` of the Actor/Critic alone cannot address a saturated Warp collector.
- New HF dataset `runs/602a1e82/diagnostic-archives`: **147 immutable archive parts**, ~221 MB diagnostic archives plus23 post-eval entries ~27 MB, all SHA-manifest-backed. V9 120 parts ~159MB, baseline169 parts ~236MB. Upload volume alone does **not** explain baseline speed because baseline uploaded even more diagnostic bytes yet trained faster. Profile uploader separately before blaming HF I/O.
- Both V9 and newest use NoOracle, but scene dimensions/visibility belief, confirmation and policy architecture differ; do not treat a scene-dependent throughput difference as algorithm-only.

## 3. Identified / FIXED huge evaluation overwork: extra best-candidate Actors

Read the remote `deferred-gpu-eval/{id}/deferred-gpu-eval.tar.gz` **streaming only manifest.json** (did NOT download the entire 100+MB Actor bank):

| Run | Unique stored | `periodic` reason | `training_metric_best_candidate` reason | Candidate-only, off-grid |
|---|---:|---:|---:|---:|
| 18d756ef | 265 | 68 | 197 | 196 |
| e079eace | 157 | 72 | 85 | 84 |
| **602a1e82** | **254** | **76** | **180** | **178** |

Counts of reasons overlap: some snapshots have both `periodic` and candidate reasons.

The new run's **entire 254 snapshots** went through held-out validation (`fullflow/validation_required_checkpoints=254`, `fullflow/validation_logged_checkpoints=254`) despite user wanting *every 100 vector steps*. A 7,500-step training run requires only ~76 periodic snapshots: **178 additional off-grid policies / 70.1% of evaluation workload**. This is not 70% of total wall time! W&B older runs validated only selected subsets, while the new run validated everything.

Implemented `UAV_PERIODIC_ACTORS_ONLY=true` as **opt-in** in `uav_network/experiments/training.py` to prevent `TrainingMetricCandidates` from force-saving additional Actor snapshots at every trailing-window score improvement. All initial Actor, actual every-100 vector-step policies and final policy are **still stored**, all periodically stored Actors **still validated on the same100 fixed seeds**, held-out Final Test1000 and best selection still use Validation (no training-score shortcut). No changes to scientific parameters, reward, dynamics or learner update ratio. Existing run unchanged; new W&B run ID.

## 4. Safe speed-profile ablation implementation

New Kaggle notebook generated from SAME tested 3-cell reference (`therwed/notebook5051433b29` style) and same source tar SHA-verification. Separate W&B run, never resumes or overwrites originals. Inline flags only, before child training starts:

```python
UAV_PERIODIC_ACTORS_ONLY=true
UAV_FAST_GRAD_CLIP=true
UAV_BATCH_SCALAR_METRICS=true
UAV_CUDA_EVENT_PROFILE=true
UAV_FAST_TRAINING=false  # preserve forensic accuracy
```

- `UAV_FAST_GRAD_CLIP=true`: Existing `uav_network/algorithms/optimization.py` skips duplicate manual grad norm reductions; uses `torch.nn.utils.clip_grad_norm_` with nonfinite detection. Check test numerical parity and final policy quality before attributing speedup.
- `UAV_BATCH_SCALAR_METRICS=true`: Existing code batches CUDA scalar metric transfers when tensor shapes/device match; reduces GPU-host fences, retains telemetry contents.
- `UAV_CUDA_EVENT_PROFILE=true`: Existing sparse CUDA Event probe logs `timing/probe_update_cuda_ms` at diagnostic cadence (global step65,536), without synchronizing every learner update.
- `timing/unattributed_host_seconds` new diagnostic provides approximate residual host time; it is **not an exclusive critical-path breakdown** due to overlapping kernels/workers. Continue logging actual collector_wait, replay_sample, optimizer host times.
- `UAV_FAST_TRAINING` remains **false**: user requires full HF `forensic-transitions`, `forensic-minibatches`, `root-cause-timeline`, reward/progress and W&B, so we do not trade away evidence.

## 5. Additional potential optimizations requiring real T4 A/B

1. **Replay sampler**: benchmark current `.index_select` + per-field `.pin_memory().to(cuda,non_blocking=True)` vs `UAV_HYBRID_PINNED_STAGING=1` (existing bounded preallocated pinned staging protected by CUDA Event). On CUDA, `.pin_memory()` itself copies CPU memory and can incur overhead; official PyTorch docs warn naive pinning may be slower. Keep ratio75/25, sample RNG/uniform indices and batch8192 unchanged. Check bandwidth, CPU time, GPU VRAM, numerical sample identity and CUDA data race; do not toggle blindly.
2. **Warp collector**: profile GPU0 Warp kernels with Nsight/PyTorch profiler; isolate Bayesian belief/FOV, network/CSMA/CA and dense mission logging. Compare exact same scenario with forensic intervals changed *only in a dedicated logging-ablation*; user-facing full deep log stays default.
3. **Inter-GPU Actor refresh**: compare legacy clone-to-CPU-copy vs `UAV_DIRECT_GPU_COLLECTOR_COPY=1` on T4×2. It uses direct peer-device copy and saves host staging only when P2P topology supports it; verify state buffers (BatchNorm), bitwise policies, walltime and GPU utilization. Do not combine experimental toggles before measuring one-at-a-time.
4. **Inductor compilation**: record compile startup vs steady-state; `TORCH_LOGS=recompiles,graph_breaks`, sparse CUDA Event and torch.profiler for real T4. CUDA graph `mode=reduce-overhead` may benefit but requires static shapes and careful RNG/control-flow semantics; not enabled without regression.
5. **Diagnostics/uploader**: asynchronous shard publication, reduce eager per-step CPU sync while retaining all required log records and HF manifest integrity. Check observable host residual and event cadence; network bytes alone are not causation.
6. **Benchmark correctness**: measure comparable `global_step` interval (e.g. after130k warmup to2M transitions) on identical source/hardware, at least 3 replicates; report steps/s, walltime per vector step, J/s/sample, CPU/VRAM peak and GPU0/1 usage; retain AUC/Delivery/False Confirm and checkpoint hashes, not just speed.

**Do not:** reduce U4, B8192, 4096 envs or eval100 merely to claim an FPS improvement; do not switch to an inaccurate simulator or change Bayes prior, TX max, hidden sizes, seed, or NoOracle safety constraints. Don't call trial-speed results production-verified until actual T4 run.

## 6. Verification status

- TDD RED: fresh speed test missing opt-in functionality failed import; GREEN: 4 tests pass after implementation.
- Real MATD3 Warp CPU episodic smoke with hybrid replay fraction0.75: snapshot manifest confirmed **no training_metric_best_candidate**, periodic-only retained and compiled config validation passed. This was a tiny CPU scenario, not an E4096 T4 throughput measurement.
- Broader selected regression after implementation: **71 passed, 4 subtests passed, 3 forensic uploader tests failed** from inherited W&B env global-state conflict in a combined pytest process; the exact same three forensic tests **passed 3/3 in isolated test process** with clean environment. Not a CUDA issue, not a claimed green aggregate 74/74.
- The 3-cell baseline bootstrap/code-bundle plus live cloud mandatory unittest tests were retained. No actual fresh Kaggle T4 fullflow has been executed from new optimized branch.
