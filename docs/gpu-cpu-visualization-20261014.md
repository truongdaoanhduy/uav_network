# GPU training followed by a separate CPU visualization session

## Selected longer trial

`configs/motion-long.yaml` uses fresh learners, seed 44, evaluation seed 10044,
500 completed episodes total per algorithm, and at most 1000 steps per episode.
An episode can finish sooner when the mission succeeds or the swarm runs out of
battery. Two vector workers can finish together, so the reported total can exceed
the requested budget by at most one episode.

The execution settings are inherited from `motion-study-fast.yaml`: two CPU
environments per learner, six independent actors and centralized twin critics,
256×256 hidden layers, batch 512, replay capacity 100000, CUDA replay, 4000 fresh
transitions before updates, one update per vector collection, idle-poll
coalescing, and vectorized critics. Model compilation and fused Adam remain off.

On two T4 GPUs, MASAC uses GPU 0 and MATD3 uses GPU 1 independently. There are four
complete swarm environments in total, each containing six UAVs. This is not one
learner distributed across two GPUs.

The learner has a 36000-second soft training budget; the Kaggle kernel has a
39600-second ceiling. The final summary records whether the episode budget was
completed. A time-budget stop still saves and uploads a policy for CPU evaluation.
Increasing the episode budget supplies more experience; it does not guarantee
convergence or mission success.

## Autonomous handoff

Launch the selected profile with:

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/motion-long.yaml --account account_02 \
  --required-gpu-hours 22 --timeout-seconds 39600 \
  --credential-dataset haibro1234/uav-wandb-credential-masac50k-d59860c \
  --cpu-visualize
```

1. The GPU job runs native mission/CUDA regression gates and a configuration
   benchmark before opening the two independent learners.
2. During training it logs online W&B metrics and uploads policy checkpoints
   every 10 episodes. Inline evaluation is disabled for this mode.
3. Once successful learner summaries exist, it pins the actual uploaded W&B
   artifact version (`vN`), writes the training result first, and submits a
   private child kernel with `enable_gpu=false` and no accelerator flag.
4. The GPU runner exits. The new CPU session downloads the pinned policies and
   evaluates each on CPU, with one episode at seed 10044. It produces PNG,
   metrics JSON, and MP4 (GIF when ffmpeg is unavailable).
5. The CPU job URL is recorded in the GPU output file
   `cpu-visualization.json`; the CPU output includes
   `visualization-result.json` and `hardware.json`.

Frames are sampled every five simulation steps at 10 fps, including the first
and terminal states. With the default one-second simulation step, playback is
50× simulated time. The video shows UAV paths, obstacles, camera footprints,
battery, target discovery, and reports delivered to the GCS. Evaluation does not
update policy weights.

The GPU job mounts a private dataset containing only the selected Kaggle token
needed for the child submission. The CPU child mounts only a private W&B
credential dataset, reusing an existing verified single-file dataset when
possible. Neither script nor job manifest embeds token bytes. The CPU child does
not receive the Kaggle launcher token or unrelated HF credentials.

Checkpoint transfer uses the existing W&B artifact flow; HF is unnecessary for
this handoff. Training and uploaded checkpoints are retained if child submission
or rendering fails; postprocessing failures are reported explicitly.

For a downloaded checkpoint, CPU visualization also works locally:

```bash
.venv/bin/python -m uav_network evaluate \
  --checkpoint /path/to/final.pt --episodes 1 --seed 10044 \
  --animate --frame-stride 5 --fps 10 --video-format auto \
  --output-dir outputs/cpu-preview
```

## What vmap and a GPU packet port mean here

`torch.func.vmap` adds a model dimension to the computation over the six
independent critics. Each critic retains its own live parameters, gradients,
optimizer state, and checkpoint format. This batches GPU matrix operations; it
does not put the SimPy event loop or packet simulator on CUDA.

A full GPU packet port would replace Python/SimPy event handling with GPU kernels
and fixed or bounded arrays for packet queues, timers, contention/backoff, ACK,
retries, TTL, PHY/SINR, custody, and RF/battery accounting. Ordering simultaneous
events and checking behavior against the native backend are part of that work.
Installing a framework or moving observations to CUDA is insufficient.

WarpDrive provides a framework for environments with CUDA C or Numba step
functions and GPU training. It cannot automatically translate the current
UavNetSim event simulation. Its public repository was archived on 2025-05-01.
The current project retains the tested native packet simulator and uses the
measured collector and critic optimizations.

Primary references:

- https://docs.pytorch.org/docs/2.11/generated/torch.vmap.html
- https://github.com/salesforce/warp-drive
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md

## Measured historical resources

These measurements come from the completed 100-episode, max-800-step runs of
2026-10-13. They are historical observations, not predictions for the new trial.

| Learner | Assigned physical GPU | Mean / peak GPU utilization | Peak allocated GPU memory |
| --- | --- | --- | --- |
| MASAC `119fc307` | T4 0 | 27.45% / 42% | 0.940 GiB |
| MATD3 `35837919` | T4 1 | 16.99% / 25% | 0.923 GiB |

Each T4 exposed 14.562 GiB, and the host exposed four logical/available CPUs.
W&B's sampled host CPU metric averaged 21.92% during the MASAC observation window
and 20.88% during the MATD3 window; these are overlapping whole-host observations,
not separate allocations to each learner. W&B can report both physical GPUs in
each run despite its learner's CUDA device mask.

Aggregate CPU usage does not rule out a bottleneck in a serial event loop.
GPU utilization is not a throughput target: batching work can reduce wall time
without raising average utilization. See the prior paired throughput evidence in
`training-acceleration-20261013.md`.

Historical runs:
- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/119fc307
- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/35837919

## Verification

The final CPU suite after active-Python CLI selection: 156 tests, 146 pass, 10
CUDA skips, 76.918 seconds. Native checkpoint evaluation, decoded GIF frames, pinned
artifact handoff, private credential boundaries, partial-result preservation,
and rejection of GPU-visible visualization are covered.

The real autonomous GPU-to-CPU smoke completed successfully. Machine-readable
evidence: [gpu-cpu-visualization-20261014-evidence.json](gpu-cpu-visualization-20261014-evidence.json).

The first real GPU smoke passed all 10 CUDA tests and eight native mission
regressions, and both algorithms uploaded immutable policies (four episodes,
160 transitions, 77 updates each). Its CPU handoff exposed a runtime dependency
issue: the image-level Kaggle 2.0.2 CLI lacks the required JSON format option,
while setup installed 2.2.4 in the project venv. The launcher now invokes
`sys.executable -m kaggle` so its CLI and SDK use the same Python environment.
An actual subprocess regression covers operation without a global executable.

The next smoke reached CPU kernel submission. Kaggle rejected the generated
52-character validation-child title; its API response explicitly limits titles
to 50 characters. The child now uses the shorter `-cpu` suffix; a regression
checks the actual generated validation title and slug consistency. A separate
CPU rendering check of the real uploaded policies completed successfully.

Final autonomous smoke, source `60e18c08df88108bbc4cda3a93b52ffe9ca5b065`:

- GPU: https://www.kaggle.com/code/haibro1234/uavnet-validate-20261014-005513-3ce5a9
- CPU: https://www.kaggle.com/code/haibro1234/uavnet-validate-20261014-005513-3ce5a9-cpu
- Both learners: four episodes, 160 joint transitions, 77 real CUDA updates;
  uploaded artifact versions match the policies downloaded by the CPU child.
- The actual Kaggle child metadata has `enable_gpu=false`, private visibility,
  and only the W&B credential dataset. Hardware reports CUDA unavailable.
- Both child evaluations use CPU and seed 10044 and produce PNG plus valid MP4
  (800×800, 10 fps, nine frames including terminal step 40).

The completed small smoke verifies execution and postprocessing, not convergence.
