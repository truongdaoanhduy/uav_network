# Native packet environment capacity sweep — 2026-10-14

## Scope

Measure the existing native UavNetSim CPU environments with the current CUDA learners. Requested counts are **128, 256, 512, 1024, 2048, 4096**, in ascending order. At each count, run MASAC then MATD3 on the same physical T4 (GPU 0), sequentially in fresh processes. An actual CUDA OOM stops the entire sweep immediately, including the other learner at that count and all larger counts. CPU RAM/PID guards, worker failures and timeouts also stop the sweep rather than manufacture measurements for an untrainable case.

The experiment uses public code pinned to commit `14187bfbd8b2c386f96fe1d8f6e50396d38ac97e`:
https://www.kaggle.com/code/duytrngoanh/uavnet-envscale-20261014-0149-9b7c

This is a separate private Kaggle job on account_01. No credential datasets are mounted. W&B tracking is disabled for the probe. The previous 500-episode training job on account_02 is not modified or cancelled:
https://www.kaggle.com/code/haibro1234/uavnet-train-20261014-010307-f4376f

## What increasing environments changes

`make_vector` creates one spawned CPU worker per complete UavNetSim environment. The packet scheduler and motion/search environment remain on CPU; only policy inference, learner updates and tensor replay run on CUDA. This experiment does not implement a GPU packet-simulation port.

Use `configs/motion-long.yaml`: six UAVs; hidden sizes 256×256; replay capacity 100,000; batch size 512; learning starts at 4,000 collected joint transitions; one learner update per vector step; vectorized independent critics; FP32; compile and fused optimizers disabled; one Torch/BLAS thread; seed 44. Each worker has its own simulated world and packet events. The profile's 1000-step episode horizon is retained.

The replay allocation is fixed at 642,400,000 bytes (~612.6 MiB), independent of environment count. Enlarging the inference batch therefore need not fill a 16-GB T4. Allocated VRAM and GPU compute utilization are separate metrics.

Changing environment count also changes learning intensity with a fixed batch/update schedule:

| Environments | Replay sample uses / new joint transition |
|---:|---:|
| 128 | 4 |
| 256 | 2 |
| 512 | 1 |
| 1024 | 0.5 |
| 2048 | 0.25 |
| 4096 | 0.125 |

This is a capacity/throughput probe, not evidence of improved convergence. It does not automatically replace the current 2-environment long-training profile.

## Measurement

Each fresh learner/replay process starts its real native workers, collects warmup transitions and executes two real learner warmup updates. Measure at least 16 subsequent stochastic policy/vector steps with real replay insertion and learner updates, and at least 10 seconds. Synchronize CUDA before and after timing.

Record:
- sampled physical-GPU VRAM peak from `nvidia-smi`;
- PyTorch CUDA peak allocated and reserved bytes;
- GPU and total-host CPU utilization mean/peak during the timed interval only;
- actual native worker PIDs, collection rate, update count and startup/warmup phases;
- effective cgroup RAM/PID limits, matching usage, and ancestor pressure.

Utilization samples crossing timing boundaries are excluded. GPU samples reserve a conservative one-second lookback because NVIDIA's sampling window depends on the device. At least three valid timed samples are required for both GPU/CPU means. Missing/incomplete CUDA telemetry becomes an explicit failed measurement, not an invented zero or a successful fully measured case.

The supervisor checks RAM at 85% and PID usage at 90%, and gives each case a 600-second startup/warmup/measurement deadline. Every child case owns an isolated process group so cleanup also terminates partial native worker sets. Case results and the parent report are written atomically before proceeding.

Resources during a failed startup are explicitly startup measurements; they cannot be presented as GPU utilization of a working 128-environment training loop.

## Reproduce

On a CUDA machine with the native dependencies installed:

```bash
.venv/bin/python -m uav_network.experiments.env_scale \
  --config configs/motion-long.yaml \
  --counts 128 256 512 1024 2048 4096 \
  --algorithms masac matd3 --device cuda \
  --steps 16 --min-seconds 10 --case-timeout 600 \
  --memory-fraction 0.85 --output outputs/env-scale-new
```

Use a new empty output directory. The runner refuses to overwrite earlier evidence. `results.json` records the attempted cases and skipped larger counts; each case keeps `case.json`, `resources.jsonl` and `worker.log`.

## Validation

Full local suite after the reviewed resource/timing fixes: **164 tests, 154 passed, 10 CUDA-only skips**, 74.753 seconds. Eight probe tests cover real native two-worker updates for both algorithms, inherited cgroup pressure, timing-boundary exclusion/missing telemetry, child termination, first-OOM stopping and evidence preservation. A separate read-only review has no remaining Critical/Important finding for this execution path.

Cloud verification: **all 10 CUDA tests passed**, 53.575 seconds, with no skips. The Kaggle kernel completed normally because the supervisor recorded its intentional resource stop rather than allowing an uncontrolled OOM. Raw trace summaries were independently recomputed from all three downloaded resource logs and matched the saved results exactly. Worker counts were independently checked; all skipped case directories are absent.

## Actual results

Physical device: Tesla T4 (compute capability 7.5), driver 580.178.04. NVIDIA reports 15,360 MiB framebuffer capacity; PyTorch reports 14.562 GiB usable device memory. Kaggle supplies four available logical CPUs and an effective 30-GiB cgroup RAM limit. Python 3.13.15; Torch 2.11.0+cu128. CUDA_VISIBLE_DEVICES=0 isolates the single measured GPU; the session's second T4 is idle.

| Env count | Learner | VRAM peak, NVIDIA MiB | Mean GPU utilization, timed | Mean CPU utilization, timed | Peak RAM, GiB | Joint env transitions/s | Result |
|---:|---|---:|---:|---:|---:|---:|---|
| 128 | MASAC | 1133 | 0.3125% | 92.90% | 14.73 | 169.42 | Passed |
| 128 | MATD3 | 955 | 0.2857% | 95.26% | 14.69 | 182.16 | Passed |
| 256 | MASAC | 763, warmup only | Unavailable | Unavailable | 25.53 | Unavailable | CPU RAM guard |
| 256 | MATD3 | Not run | Not run | Not run | Not run | Not run | Stopped before sibling learner |
| 512 | Both | Not run | Not run | Not run | Not run | Not run | Skipped after first resource failure |
| 1024 | Both | Not run | Not run | Not run | Not run | Not run | Skipped after first resource failure |
| 2048 | Both | Not run | Not run | Not run | Not run | Not run | Skipped after first resource failure |
| 4096 | Both | Not run | Not run | Not run | Not run | Not run | Skipped after first resource failure |

At 128 environments, each learner completed 32 vector warmup steps (4096 joint transitions), two learner warmup updates, and 16 timed vector steps/updates (2048 new joint transitions). MASAC's timed interval was 12.088 seconds; MATD3's was 11.243 seconds. There were respectively 16 and 14 valid GPU samples, and 18 and 15 valid CPU samples. CPU percentages are aggregate host utilization across the four CPUs. Startup took 83.02 and 80.59 seconds.

The 256-environment MASAC case did start and reset all 256 native workers. It then stopped during replay warmup, before any timed policy/update interval. The triggering cgroup RAM sample was 27,407,511,552 bytes = 25.525 GiB out of 30 GiB (85.084%). The guard threshold is 85%, not the exact OOM limit. PID peak was 265 out of a 38,508-task limit. Its 763-MiB sampled GPU peak is **partial warmup/allocation usage**, not a working 256-environment training measurement; PyTorch's saved peak was 0.614 GiB allocated and 0.627 GiB reserved.

**No CUDA OOM was observed.** The experiment stopped on CPU RAM pressure first. It therefore cannot establish a maximum CUDA environment count, or report measured training VRAM for the four skipped counts. Releasing the case's process group ended its learner/native workers; the private Kaggle benchmark session is complete.

## Interpretation and limits

The observed timed GPU utilization at 128 environments is roughly 0.3%, while CPU utilization is 93–95%. With one update per vector step, the larger env batch supplies many more CPU-simulated transitions per CUDA learner update. The fixed replay and update batch keep VRAM around 1 GiB. More spawned worlds consume CPU RAM; they do not move SimPy/native packet work onto CUDA.

Increasing native environment count alone is therefore not evidence of making better use of the T4 in this configuration. The result motivates optimizing/batching or porting environment simulation and separately choosing a learner update budget. It does not prove 128 envs is the fastest native configuration, or guarantee 128-world 1000-step episodes will fit in RAM for an entire 500-episode run. That longer workload and convergence were not tested here.

Full saved cloud report, triggering/peak resource samples, software/configuration, original training snapshot and trace digests: [environment-scale-20261014-evidence.json](environment-scale-20261014-evidence.json).

## Primary references

- NVIDIA utilization and framebuffer reporting: https://docs.nvidia.com/deploy/nvidia-smi/index.html
- Linux process cgroup membership and hierarchical limits: https://docs.kernel.org/admin-guide/cgroup-v2.html
- PyTorch reserved allocator memory: https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_reserved.html
