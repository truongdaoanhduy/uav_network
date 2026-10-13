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
particular percentage. Native CPU repeat tests pass for both algorithms; CUDA
and longer repeats must be verified with the actual cloud jobs.

PyTorch does not guarantee identical outputs across releases or platforms:
https://docs.pytorch.org/docs/stable/notes/randomness.html
