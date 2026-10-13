# Receiver admission and repeatable single-seed study

Training root seed is 44; evaluation uses exactly one map seed, 10044.
Training episode map seeds follow the existing deterministic sequence. This is
one training seed, not an average over multiple independently trained seeds.

## Fix

The old mask advertised peers using range/liveness alone, even when all sender
reports were already known by that peer or its buffer was full. The backend
then rejected these actions. The new mask and backend share custody/TTL/buffer
admission. Obstacle loss/SINR is deliberately not used by the mask.

This models an ideal reliable admission handshake with no separate airtime or
energy cost. It does not expose target coordinates or channel outcomes.
GCS still ACKs duplicates to clear sender custody, as in the original backend;
duplicate reports do not grant another delivery reward.

Set `network.receiver_admission_mask=false` for the original baseline. Legacy
checkpoints without this field load with false so historical evaluations retain
their original semantics. Fresh training is required for the controlled study.

## Actual completed-checkpoint probe

MATD3 checkpoint 1000, deterministic evaluation seed 10044, before/after mask:
return -315.98482340115527, search 20%, delivery 20% in both cases.
2051 obstacle blocks, 51 peer blocks, 3404 commanded stalls over 800 steps and
6 UAVs. 55.7396% of movement components have zero float32 tanh derivative;
74.5208% have derivative below 1e-4; maximum absolute raw logit 166.9786.
These show saturation and physical blocking, not proof that a particular
regularizer will improve performance. Receiver masking alone did not improve
this old policy on this map. Dynamics and reward weights were not changed.

## Controlled diagnostic training

Compare fresh baseline, admission fix, and an exact repeat of the fix using
MASAC and MATD3 separately, seed 44, 100 episodes, 2 CPU environments per
learner, batch 512, replay 100000, warmup 4000 transitions, 1 update per vector
step, maximum 800 steps per episode. Evaluate only seed 10044. This short study
is for diagnosis and reproducibility, not a convergence claim.

Console logging stays enabled through W&B `console=wrap`. Episodes include
`motion/*` obstacle/peer/stall counts and `routing/*` accepted/rejected actions;
learner losses, gradients, policy saturation, network and battery metrics remain
available. Runs are ordinary unpinned runs. Private credentials are not stored
in Git or public job manifests.

## Exact repetition scope

Repeatability requires identical source, settings, software versions, GPU/CPU
platform, environment count and a completed episode budget. Time-cutoff runs or
policy-only warm starts are not exact repeats. Model/episode/settings SHA256
fingerprints in summary.json and W&B expose differences instead of promising a
particular percentage. Native CPU repeat tests pass for both algorithms. Actual Tesla T4 CUDA
validation ran all six CUDA tests successfully (32.447s), including exact model
and episode hashes for both algorithms in short repeated training. The online
smoke learners finished four episodes, 160 joint transitions, 77 updates each.
Longer cross-job repetition remains pending.

PyTorch does not guarantee identical outputs across releases or platforms:
https://docs.pytorch.org/docs/stable/notes/randomness.html

## Submitted online jobs (13 October 2026)

All study jobs use source commit `56fd8e3`, fresh initialization, seed 44 and
100 episodes per algorithm. Training soft limit is three hours per learner;
Kaggle kernel hard limit is eleven hours. A timeout is reported as incomplete,
not an exact-repeat result. Runs are unpinned and console logging was verified
on the completed CUDA smoke runs.

| Condition | Account | Kaggle | MASAC W&B | MATD3 W&B |
|---|---|---|---|---|
| Baseline | riujick | [Job](https://www.kaggle.com/code/riujick/uavnet-train-20261013-101522-0176c3) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/5630eb8f) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/463fe9f0) |
| Fixed | vuliu123456 | [Job](https://www.kaggle.com/code/vuliu123456/uavnet-train-20261013-101524-45a5e6) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2fdea1f8) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/cb9cd52b) |
| Fixed repeat | haibro1234 | [Job](https://www.kaggle.com/code/haibro1234/uavnet-train-20261013-101703-e62802) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ffb90702) | [Run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f2706cb2) |


## Completed MATD3 study: verified metrics, console and exact repetition

All three MATD3 learners completed 100 episodes. These are means across the
episodes of one training root seed (44), not means across independent seeds.

| Condition/account | Return | Search | Delivery | Success | Mean blocked rate |
|---|---:|---:|---:|---:|---:|
| Baseline / riujick | -503.113 | 39.2% | 27.7% | 1% | 67.82% |
| Admission fix / vuliu123456 | -624.613 | 26.4% | 21.7% | 2% | 76.83% |
| Exact repeat / haibro1234 | -624.613 | 26.4% | 21.7% | 2% | 76.83% |

No episode in these runs exhausted a UAV. Mean remaining battery was 55.92%
for baseline and 57.44% for the fixed runs; minimum UAV fraction in the whole
history was 22.08% / 35.92%, respectively. Low battery is not the failure
mechanism observed here.

Both fixed runs have identical model, episode and settings hashes after the
full episode budget. Hardware/software also match: Tesla T4, Python 3.13.15,
NumPy 2.5.4, PyTorch 2.11.0+cu128. This verifies this repeated 100-episode
experiment; it does not promise identity after changing source/platform/config.

Model hash: 7655288669a6f69095769a222c458ba96877ed9af26277ba88c09873167edfee

Episode hash: 5b0c5407f058e91db9d384007d13b472b50453edad47e62b050f11bd16f646ab

Downloaded W&B output.log and episodes.jsonl for all three runs. Console is
nonempty (fixed runs each 37,558 bytes), no traceback was found in these
captured console files. Logs contain blocked-motion and report-loss warnings;
absence of a traceback in captured console is not a claim to have read every
Kaggle stderr file. Fixed episode 100 has 3,146 blocked opportunities,
blocked-motion reward -629.2, three report expirations and zero delivery.

Admission masking by itself did not improve aggregate search/delivery on
this short single-seed experiment. See
[the motion safety study](obstacle-avoidance-single-seed.md) for the separate
frozen-policy intervention and new training job.
