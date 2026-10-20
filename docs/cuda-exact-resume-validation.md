# CUDA exact continuation validation

These offline integration fixtures exercise real Warp CUDA state and local full
checkpoints without credentials, tracking uploads, private inputs or online runs.
Use a checkout pinned to the reviewed commit and install the project's GPU
extras (including pinned Warp). From the repository root, on two visible GPUs:

```bash
UAV_CUDA_FULL4096=1 PYTHONPATH=.:tests python -m unittest discover -s tests -p test_cuda_exact_resume.py -v
```

The command discovers 11 tests. Six small tests cover a serialized world and future
action RNG on each of the first two visible CUDA devices, live packet sessions and
MAC state on both devices, and four real active-learning save/resume cases
(MASAC/MATD3, BASE/PQL). The small learner fixtures use two worlds and two updates
per collection; async PQL must advance critic and actor updates after restoration.
They check exact environment continuation without requiring identical async
learner scheduling or learning quality.

The five production tests require both `UAV_CUDA_FULL4096=1` and two visible GPUs.
One serializes/restores a real 4096-world vector and compares every next-step world
array and future action RNG on CUDA0. Four cases each run two fresh `train()`
calls across a zero-completed-mission full-checkpoint boundary, for eight production
training segments total. BASE and async PQL must use collector CUDA0 and
learner/replay CUDA1. They load
`configs/comparison-20k-steps1800-scenario-v3.yaml` unchanged for world and
algorithm settings: 4096 worlds, 20,000 target episodes, 1800 steps, seed44,
35 targets, 25 obstacles and 1,000,000-byte reports. Only architecture/algorithm
selection, temporary local output/resume paths, disabled tracking and
`runtime.max_train_seconds=1e-9` differ. Each segment collects one vector step
before its time stop. Warmup tests must retain active seeds, increase world lengths
and global-step prefix, complete zero missions, and make zero learner updates.

CPU discovery honestly skips all 11 tests before allocating a CUDA world; it does
not establish CUDA correctness. With one GPU the small tests run there and the
production tests skip. Without full opt-in the production tests also skip.
A two-GPU opt-in validation should report 11 executed tests and no skips; a skip,
OOM or timeout is incomplete production evidence, not an equivalent smaller run.

The fixtures print JSON evidence with world array bytes, serialized checkpoint
bytes, step/update prefixes, elapsed segment time, host memory and CUDA memory.
Production snapshots are large (roughly 2.2 GiB of world arrays); checkpoint
deserialization temporarily requires host memory as well as device memory.
The vector test stores its start and next-step oracles sequentially on disk and
compares one current array at a time. The training cases retain only small metadata
between calls. Temporary files are removed and CUDA caches released between cases.
Allow disk space for two full snapshots/checkpoints per case plus replay/model
data. Do not invoke the production fixture with CPU device overrides.

## Verified Kaggle execution (2026-10-20)

The reviewed branch `fix/v3-session-resume-relay-audit-20261020` at commit
`da6cc466c2e9df740b90047150c4a5f61f4dfa29` was executed in the private Kaggle
kernel `duytrngoanh/uav-resume-1800-cuda-check-20261020-da6cc466`. A fresh
read-only status/output check at `2026-10-20T03:22:59Z` reported `complete` and
validated two visible Tesla T4 GPUs. The exact production-profile fixture used
`configs/comparison-20k-steps1800-scenario-v3.yaml` with 4096 environments,
20,000 target episodes, `max_steps=1800`, and seed 44. All 11 CUDA tests executed
with 0 failures, 0 errors, and 0 skips.

This is a CUDA simulator/collector/checkpoint correctness validation, including
full-4096 warmup/resume segments. It is **not** evidence that the 20,000-episode
scored training budget has completed, and this validation did not submit a
production training run.
