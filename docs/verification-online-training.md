# Online training repair and restart

Checked on **2026-10-12 at 16:30 UTC / 23:30 Vietnam time**. The replacement
MASAC and MATD3 runs are online and running. These are startup measurements;
the 1,000-episode training and policy-quality evaluation are still pending.

## Old job diagnosis

Old kernel: https://www.kaggle.com/code/bduy2378/uavnet-train-20261012-123636-259be1

| Old run | State | Logged transitions | Updates | Completed episodes |
| --- | --- | --- | --- | --- |
| MASAC `60d32e75` | failed | 13,202 | 4,601 | 16 |
| MATD3 `48a5b5c4` | failed | 20,402 | 8,201 | 25 |

MASAC produced NaN from masked categorical entropy: multiplying float32.min
by alpha > 1 overflows, then multiplying by probability zero produces NaN.
An unchecked optimizer update poisoned weights before CUDA categorical sampling
asserted. The old fail-fast runner then sent SIGTERM to MATD3. Laptop shutdown
did not cause either failure. Old logs remain under `outputs/kaggle/`.

The fix clears zero-probability log terms before multiplication, checks gradient
norms before every Adam step and lets independent learners finish. Training
manifests are saved before evaluation; one evaluation failure does not prevent
the other evaluation or destroy completed results. Kaggle policy checkpoints
now save every 10 episodes, rather than every 100.

## Verified code and CUDA execution

Training source: `61240a65e553f37bff94d36a2645be586d38f97a`.
Independent review's result-preservation and paginated-file findings were fixed.
Local full suite: **76 tests, 72 passed and four explicit CUDA skips**.
The same four CUDA tests passed on real Kaggle hardware in **12.551 seconds**,
including alpha 1.01, 2 and exp(5) with scalar metrics disabled.

Validation kernel:
https://www.kaggle.com/code/bduy2378/uavnet-validate-20261012-161147-81ee29

- MASAC: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/40ae4b6e
- MATD3: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/315da96d

Both validation runs finished with four episodes, 160 joint transitions and
77 updates on CUDA learner/replay. Each has 13 online history rows and one model
artifact. Both downloaded checkpoints have finite model tensors and completed
one CPU evaluation episode. Outputs, CUDA log, hardware and evaluations are in
`outputs/kaggle/uavnet-validate-20261012-161147-81ee29/result/`.
These short runs verify execution; their search/delivery scores are not evidence
of a trained policy.

## Live replacement job and workspace

Kernel: https://www.kaggle.com/code/bduy2378/uavnet-train-20261012-162400-bd7488

| New learner | Physical CUDA mask | State | Joint transitions | Updates | Episodes |
| --- | --- | --- | --- | --- | --- |
| MASAC `f8524504` | 0 | running | 5,802 | 901 | 7 |
| MATD3 `971920c8` | 1 | running | 7,402 | 1,701 | 9 |

- MASAC: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f8524504
- MATD3: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/971920c8
- Saved project view: https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=gmcixx5iveu

Workspace name: **UAV Network 2D - MASAC and MATD3**. API read-back confirmed
six sections, 26 charts, the `uav-network-2d` group filter and both new runs
pinned/enabled. Histories have increasing global steps and finite logged losses.
Actual GPU utilization/VRAM samples are available under each run's System tab.
Both old IDs are present in the project API; the user's former browser view/filter
was not observed, so its exact reason for hiding them is not established.

Hardware: four CPU cores, Tesla T4 x2 (14.56 GiB each), Python 3.13.15,
PyTorch 2.11.0+cu128. Each learner runs two spawned CPU swarm environments,
six UAVs per swarm, batch 512 and CUDA replay of 100,000 joint transitions.
Full-size warmed benchmarks measured 9.09 MASAC and 18.59 MATD3 updates/s;
end-to-end training also includes simulation, transfers, tracking and resets.

The submission reused this session's successfully validated private credential
dataset after confirming owner, privacy and unchanged W&B file metadata. Repeated
local dataset download timeouts prevented a redundant download. No credential was
embedded in source or job manifests; the copied full `key.txt` remains local.

Kaggle and online W&B logging execute remotely after submission. The laptop can
be powered off. This job has an 11-hour ceiling; checkpoints support evaluation,
not optimizer/replay resume. Final convergence is not yet claimed.
