# Current task: live W&B audit and requested hybrid training

Observed 2026-10-23T05:45:16Z. User explicitly requested Kaggle for CUDA.

## Verified state
- Production hybrid fix is published on `feat/cpu-replay-20261022` at `b32d18ef1077371a3603664b974328744c031db5` (`fix: release staged resume checkpoint before save`). GitHub branch read-back confirmed the same SHA.
- Fresh verification for this hotfix: focused `65 passed, 4 skipped, 48 subtests`; full suite `667 passed, 74 skipped, 8 warnings, 1219 subtests` in 556.59 s; `git diff --check` and `compileall` passed.
- Requested config `configs/comparison-20k-hybrid75-1m.yaml`: replay 1M, CPU fraction .75, env4096, batch8192, 20k episodes, seed44, Scenario V3. Production hybrid constructor is connected to BASE/PQL; cloud no longer overrides placement. Keep all scientific settings unchanged.
- Actual two-T4 gate PASSED: Kaggle `ewkljfelweow/uavnet-benchmark-20261023-044345-84cefe`, W&B `31831ab2`. Full replay CPU 9,831,000,000 bytes + GPU 3,274,000,000 bytes. Checkpoint size 15,514,514,885 bytes; exact RNG/environment restore and resumed learning to 32 updates passed. Gate scope is memory/execution, not policy quality.
- Root requested 1M MASAC PQL job exists: Kaggle `ewkljfelweow/uavnet-train-20261023-051420-773792`, account_21, planned learner W&B `fc14508c`. Fresh Kaggle state is `running`; learner W&B is still absent because a separate preflight W&B run `9a716926` (`MASAC-PQL-benchmark-E4096-B8192-seed44`) is currently `running`, with env4096, replay1M and PQL enabled. This is positive evidence that the kernel is executing preflight rather than proof the learner has begun training.
- Root 1M job is pinned to pre-hotfix commit `66f4d402be2648be8514b2ea0173e80e4cdce40e`. Its first segment has no staged resume input, so the cleanup bug does not affect initial startup. However automatic child continuation inherits the same old commit and therefore does not contain the staged-checkpoint cleanup fix.
- Root cause fixed by `b32d18e`: cloud continuation marks only its downloaded resume input as disposable; after verified restore BASE/PQL release replay/environment/RNG/optimizer mmap references, preserve the source SHA for reproducibility, unlink the staged input before the next save, and never delete normal local/manual warm-start files.
- Measured disk reason: the gate checkpoint is ~15.5 GB while free working disk before save was ~20.9 GB, so keeping both the downloaded input and a newly written full checkpoint cannot fit.
- Existing MASAC PQL 500k replacement W&B `b88d7df8` remains running and reached global_step 225280 at the last fresh check. Do not stop or duplicate it.
- Old 500k root W&B runs `29e6add6`, `3070f38e`, `bda3bccd` are finished with `time_budget`, not final 20k completion. Interim CPU MASAC BASE evaluation `9a9895ea` is finished and is not final 20k evaluation.
- account_21 quota at last fresh check: 18.96 GPU hours remaining. account_25 was separately verified idle with sufficient quota if a bounded CUDA verification is later needed; do not use active accounts 21/27/28 for duplicate work.
- Evidence: `docs/superpowers/evidence/2026-10-23-staged-resume-cleanup.json`.

## Next action
1. Keep the existing root 1M Kaggle kernel; do not cancel/relaunch solely because planned W&B `fc14508c` is not visible. Fresh-check Kaggle first, then W&B.
2. Only claim actual 1M learner training once `fc14508c` exists and `global_step` advances.
3. If the root segment later creates a checkpoint but its old-code automatic continuation fails, recover from that immutable checkpoint using source `b32d18e` or newer; do not retry an unchanged old-commit child that can hit the disk-space bug.
4. When a segment reaches the full 20k budget, allow the separate CPU-only visualization/evaluation stage; do not label interim evaluation as final.
5. Preserve BASE/PQL scientific settings, RNG, environment state, checkpoint identity, and all currently active production jobs.

## Continuation rules
Read `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and the newest evidence JSON before acting.
Use credentials only through trusted helpers; never print tokens or raw secrets. A W&B missing run is not proof that Kaggle failed. A Kaggle timeout is not proof no kernel was created. Always check side effects before retrying.
