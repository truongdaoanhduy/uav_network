# Diagnostics verification — 2026-10-13

Source tested on Kaggle: 5f939a1928ba3ec7aa66f67f4bb6ec477f48c14e.
Native full suite: 96 tests in 36.781 seconds, 91 CPU passed, 5 CUDA skipped.
Actual Kaggle CUDA suite: 5 tests passed in 24.601 seconds.

## Online proof

Private job completed:
https://www.kaggle.com/code/bduy2378/uavnet-validate-20261013-065747-7a816a

Account bduy2378 (account_07), submitted 2026-10-13 06:57:47 UTC
(13:57:47 Vietnam). Both learners started about 06:59:17 UTC (13:59:17).
This was four-episode wiring validation, not a convergence experiment.

| Algorithm | W&B run | State | Critic grad norm | Mean absolute TD error | Fixed-seed eval return |
|---|---|---|---:|---:|---:|
| MASAC | ff3a027e | finished | 0.24921 | 0.10226 | -10.96096 |
| MATD3 | 9f1c79f7 | finished | 0.21175 | 0.14422 | -9.96376 |

W&B API confirmed real reward/action/rolling/elapsed metrics, evaluation at
episodes2 and4, resolved config, diagnostics.jsonl, output.log and evaluation
logs/JSON uploaded to Files. Console API returned4/6 captured lines respectively.
MATD3 had2 actual diagnostic console events; MASAC had none. A zero warning count
is not a guarantee of correctness.

The CPU failure-injection regression runs a real learner with a nonfinite model:
the optimizer rejects it, failure.json names nonfinite parameters and records the
actual seed/step/update context, failed_batch.pt preserves the sampled replay
tensors and failure_state.pt preserves the model after failure.
A missing checkpoint and simulated worker-start boundary failure also retain
local evidence. Numerical fault injection is explicitly a test, not an observed
failure in the user's long training runs.

## Saved views

72 panels across11 sections were saved using the real W&B workspace API.

Long experiments, current metrics and metadata:
https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=gmcixx5iveu

Diagnostics validation, populated new plots and Logs:
https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=zsc7ey5p85b

- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ff3a027e
- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/9f1c79f7

The long runs9c9309b3/afdad4a9 remain on their immutable old sourcea855862.
They were observed running, created 2026-10-13 06:24:08 UTC (13:24:08 Vietnam),
on account bduy2378, with snapshots530/990 episodes and runtime about34.5minutes.
Their new diagnostic-only panels can be empty because these processes cannot be
patched while running. Existing metrics continue; future launches use the new
default diagnostics. No long job was cancelled or silently restarted.
