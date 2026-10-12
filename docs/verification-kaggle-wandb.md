# Kaggle CUDA and W&B verification

Verified at **2026-10-12 12:42 UTC**. Both full-profile learners are running
on Kaggle and their transition/update counters are increasing. Training is
still in progress; final convergence and evaluation results are pending.

## Active training

- Kernel: https://www.kaggle.com/code/bduy2378/uavnet-train-20261012-123636-259be1
- Published source: `99025e71e5cf49831aa2ff62ff0226732c33ea52`.
- Account: `account_07` (`bduy2378`), selected from the copied `key.txt`.
- Verified private credential source: `bduy2378/uav-wandb-credential-arch-ablation`.
- Hardware: 4 CPU cores, two Tesla T4 cards with 14.56 GiB each;
  Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8.
- Budget: 1,000 episodes per algorithm, 11-hour kernel ceiling.

| Learner | Physical GPU mask | Joint transitions | Updates | Episodes completed |
| --- | --- | --- | --- | --- |
| MASAC | 0 | 5,202 | 601 | 6 |
| MATD3 | 1 | 6,202 | 1,101 | 7 |

These are dated startup snapshots, not final totals. W&B API verification found
both runs `running`, actor/critic loss fields, system GPU/VRAM samples and
explicit visibility masks 0/1. Each process sees its own GPU as CUDA device 0.

- MASAC: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/60d32e75
- MATD3: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/48a5b5c4

Each learner has two spawned CPU environments, six UAVs per scenario, ten
targets, eight obstacles, 800 maximum steps, seed 44, batch 512, CUDA replay
capacity 100,000 and one update per collection. Warmup collects 4,000 joint
transitions before optimization. The requested/actual episode budget and
effective sample usage are recorded in the training summary.

Before learner startup, the full job passed all CUDA regressions and warmed
benchmarks of both actual model/batch configurations. Its private metadata and
attached credential dataset were read back from Kaggle and independently checked.
Only the learner initializes W&B; simulator workers do not.

## Verified smoke and checkpoint path

Successful validation source: `907ecd0c9c110e141b788360429284ec2bf3810e`.
All **three CUDA tests passed** in 11.741 seconds, including actual MASAC/MATD3
updates from spawned native UavNetSim environments and both replay transfer paths.

Each online smoke learner completed four episodes, 160 joint transitions and
77 updates with CUDA learner/replay. MASAC elapsed 20.97 s and MATD3 15.86 s;
these small-run timings include startup and tracking.

Independent W&B API checks found `finished`, 13 history rows, early progress,
monotone global steps and one model artifact per algorithm. Both checkpoints
completed CPU evaluation and contain none of the copied credential values.

- MASAC smoke: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/66f7ec82
- MATD3 smoke: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/480bb79c

The temporary validation notebook was removed to free a queued GPU reservation.
All downloaded logs, hardware metadata, evaluation outputs and checkpoints
remain in `outputs/kaggle/uavnet-validate-20261012-121025-5b8179/result/`,
and the two W&B artifacts remain available. Generated outputs are Git-ignored.

## Performance evidence

Exact full-size config, two environments, batch 512 and 20 measured learner
updates after warmup, with native networking enabled:

| Algorithm | Local CPU learner updates/s | Kaggle T4 learner updates/s |
| --- | --- | --- |
| MASAC | 1.61 | 9.19 |
| MATD3 | 8.26 | 19.60 |

Replay storage is 642,400,000 bytes per learner. GPU collection benchmarks
measured 90.28/86.65 joint transitions/s for MASAC/MATD3. Machines and PyTorch
versions differ. Timings measure the warmed learner separately; total training
speed also includes simulation, sampling, transfers, logging and resets.
GPU utilization and memory usage are recorded online, alongside mission metrics.
Full benchmark records and GPU masks are also stored in each W&B run config.

## Credentials, local checks and fixes

Copied `/home/aduy/Documents/NCKH/uav_search_target/key.txt` in full to
`/home/aduy/Documents/NCKH/uav_network/key.txt`: **1,461 identical bytes**,
mode **0600**. The source file is untouched; the destination is Git-ignored.
W&B uses `truongdaoanhduy-uit/uav_search_target` from the copied file.
The complete key file and unrelated HF fields remain local.

Local suite: 69 tests discovered, 66 passed and three explicit CUDA skips.
The three CUDA tests subsequently passed on actual Kaggle hardware.
The source/document size check passes; each owned file is below 500 lines.

Independent review's two cleanup issues were reproduced and fixed: partial
W&B initialization finishes the failed run, and SIGTERM executes env/tracker
cleanup before a process-group fallback. The real two-worker shutdown test
passed five consecutive runs and the complete local suite passed afterwards.
The readiness marker is published once using atomic rename.

The launcher parses Kaggle's canonical push URL and rejects server-side errors
even if the CLI exits zero. Kaggle Python 3.13 could not bootstrap ensurepip;
`--system-site-packages` now creates a venv with `--without-pip` and uses the
image's installed pip. The corrected setup passed real CUDA execution.

Account 02 encountered a batch-session limit and remained queued. Only this
session's obsolete queued notebooks were removed; training moved to account 07
after verifying its quota, dataset ownership/privacy and matching W&B credential.

## Output and limits

Kaggle output stores CUDA tests, hardware, benchmarks, per-algorithm logs/config,
periodic policy checkpoints, final summary/checkpoint and CPU evaluation output.
W&B records progress, reward, search/delivery, battery, safety, packet/delay/
throughput, learner losses and system metrics; the final policy is uploaded
as a model artifact when training completes.

Checkpoints currently support evaluation. Optimizer/replay resume is not
implemented. If the time ceiling is reached before the episode budget, retain
the available periodic checkpoints and distinguish them from a completed run.
Smoke verifies execution and tracking; policy quality requires the full
training and held-out evaluation results.
