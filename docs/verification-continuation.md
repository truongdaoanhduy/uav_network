# Recovery and continuation — 2026-10-13

Project: uav_network, fixed 2D/FOV, six homogeneous UAVs, native UavNetSim.
The previous Kaggle job uavnet-train-20261012-162400-bd7488 was cancelled.
Its launched hard ceiling was 39,600 seconds. Both W&B runs are crashed.
Turning off the laptop does not stop a submitted Kaggle server job.

## Recovered evidence

| Algorithm | Saved episode | Global step | Updates | Parent W&B |
|---|---:|---:|---:|---|
| MASAC | 500 | 399360 | 197680 | f8524504 |
| MATD3 | 940 | 728026 | 362013 | 971920c8 |

Logs contain 504/946 completed episodes; only the saved prefixes 500/940 are
restored. All recovered model tensors are finite. Original files stay untouched.
Copied progress derives from actual episodes.jsonl and excludes unsaved episodes.
A separate finished recovery run 46239a6e published immutable inputs:
- truongdaoanhduy-uit/uav_search_target/uav-network-masac-f8524504-recovered:v0
- truongdaoanhduy-uit/uav_search_target/uav-network-matd3-971920c8-recovered:v0

This is policy warm-start: all model/target/temperature tensors and update counters
are retained. Adam/replay/RNG/simulator state was absent and is recreated.
New runs identify their input artifact and parent; old histories are not rewritten.

## Evaluation of recovered policies

Actual CPU evaluation uses matching seeds 10044–10046.

| Algorithm | Mean return | Search % | Delivery % | Success % |
|---|---:|---:|---:|---:|
| MASAC 500 | -534.6178 | 76.6667 | 0 | 0 |
| MATD3 940 | 32.8307 | 93.3333 | 76.6667 | 0 |

These results do not establish convergence. Metrics and previews are in ignored
outputs/recovery-evaluation/{masac,matd3}; no generated checkpoints enter GitHub.

## Verification and safeguards

A fresh review reproduced final-upload failure losing summary.json. New regression
tests fail before the fix and pass after: summary is saved before upload, failures
remain explicit, and cloud aggregation preserves valid summaries from failed uploads.
The safe default/cap is 39,600 seconds, with smaller explicit budgets supported.
Zero-episode deadlines save finite models without fabricated episode metrics.
Online periodic checkpoints wait for artifact commitment before training continues.
Native suite: 91 tests, 86 CPU passed and 5 CUDA skipped, 31.270 seconds.
Kaggle validation bduy2378/uavnet-validate-20261013-061540-d44fc1 is complete.
All 5 real CUDA tests passed in 26.435 seconds on two Tesla T4 cards.
Both online smoke runs finished: MASAC 9f591334, MATD3 154e2ae9, 4 episodes,
160 transitions, 77 updates each, CUDA learner and replay.
Each run committed four periodic artifacts plus the final v4 model, confirmed by
the W&B API. This validates wiring and restoration, not convergence.
## Live continuation

Submitted job:
https://www.kaggle.com/code/bduy2378/uavnet-train-20261013-062015-15b379

Pinned source a85586284c554d1368651af582bd9418938ceaa4, account_07,
verified remaining GPU quota 11.76 hours before submission.
Hard timeout 39600 seconds; soft training budget 36000 seconds.
Goal is 1000 cumulative episodes per algorithm.

| Algorithm | Run | Snapshot episodes | Global step | Updates | Physical GPU |
|---|---|---:|---:|---:|---:|
| MASAC | 9c9309b3 | 506 | 404162 | 198081 | 0 |
| MATD3 | afdad4a9 | 946 | 733628 | 362814 | 1 |

Both are running, with the expected immutable input artifacts linked.
Update counters increased after collecting 4000 new transitions. Replay sizes
4802/5602 confirm fresh refill; each learner uses CUDA and two spawned CPU envs.
W&B system samples show GPU0 at 28–30%, GPU1 at 19–20%, about 1 GB allocated
on each card. CPU-based UavNetSim collection remains part of the workload.

- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/9c9309b3
- https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/afdad4a9
- Saved project view, 26 panels, both new runs pinned:
  https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=gmcixx5iveu

These are dated startup snapshots, not a claim that 1000 episodes are complete.
Kaggle automatically saves/uploads/evaluates at normal segment completion;
it continues when the local laptop is off.

