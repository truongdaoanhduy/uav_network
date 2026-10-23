# Current task: live W&B audit and requested hybrid training

Observed 2026-10-23T06:46:03Z. User explicitly requested Kaggle for CUDA.

## Verified state
- Production hybrid fix is published on `feat/cpu-replay-20261022` at `b32d18ef1077371a3603664b974328744c031db5` (`fix: release staged resume checkpoint before save`). GitHub branch read-back confirmed the same SHA.
- Fresh verification for this hotfix: focused `65 passed, 4 skipped, 48 subtests`; full suite `667 passed, 74 skipped, 8 warnings, 1219 subtests` in 556.59 s; `git diff --check` and `compileall` passed.
- Requested config `configs/comparison-20k-hybrid75-1m.yaml`: replay 1M, CPU fraction .75, env4096, batch8192, 20k episodes, seed44, Scenario V3. Production hybrid constructor is connected to BASE/PQL; cloud no longer overrides placement. Keep all scientific settings unchanged.
- Actual two-T4 gate PASSED: Kaggle `ewkljfelweow/uavnet-benchmark-20261023-044345-84cefe`, W&B `31831ab2`. Full replay CPU 9,831,000,000 bytes + GPU 3,274,000,000 bytes. Checkpoint size 15,514,514,885 bytes; exact RNG/environment restore and resumed learning to 32 updates passed. Gate scope is memory/execution, not policy quality.
- Requested 1M MASAC PQL root is actively training: Kaggle `ewkljfelweow/uavnet-train-20261023-051420-773792`, account_21; W&B `fc14508c` is `running` and reached global_step 163840 at the last fresh check. Config read-back matches env4096, batch8192, replay1M, CPU fraction .75 and PQL ratio8. Replay allocation is CPU 9.831GB + GPU-hot 3.274GB, with no overwrite/unsampled eviction yet.
- Its preflight W&B `9a716926` finished `passed`, global_step163840, `cuda_pipeline_verified=true`, about63.73 joint transitions/s. The old `target_t4_scenario_verified=false` flag is a legacy hard-coded 500k/25-target benchmark label and is not the production gate used by cloud launch.
- Runtime health read-back: no non-finite summary metrics observed, W&B system OOM kills=0, process RSS peak observed ~11.54GB with >=16.95GB available, GPU0 allocated <=17.92% and GPU1 <=73.70%, application disk free ~20.91GB. A 75-second no-step observation was rechecked and the learner advanced to163840; it was one long collection, not a deadlock.
- Root 1M job is pinned to pre-hotfix commit `66f4d402be2648be8514b2ea0173e80e4cdce40e`. Its first segment has no staged resume input, so the cleanup bug does not affect initial startup. However automatic child continuation inherits the same old commit and therefore does not contain the staged-checkpoint cleanup fix.
- Root cause fixed by `b32d18e`: cloud continuation marks only its downloaded resume input as disposable; after verified restore BASE/PQL release replay/environment/RNG/optimizer mmap references, preserve the source SHA for reproducibility, unlink the staged input before the next save, and never delete normal local/manual warm-start files.
- Measured disk reason: the gate checkpoint is ~15.5 GB while free working disk before save was ~20.9 GB, so keeping both the downloaded input and a newly written full checkpoint cannot fit.
- Existing MASAC PQL 500k replacement W&B `b88d7df8` remains running and reached global_step 319488 at the last fresh check. Do not stop or duplicate it.
- Old 500k root W&B runs `29e6add6`, `3070f38e`, `bda3bccd` are finished with `time_budget`, not final 20k completion. Interim CPU MASAC BASE evaluation `9a9895ea` is finished and is not final 20k evaluation.
- account_21 quota at last fresh check: 18.08 GPU hours remaining. account_25 was separately verified idle with sufficient quota if a bounded CUDA verification is later needed; do not use active accounts 21/27/28 for duplicate work.
- Evidence: `docs/superpowers/evidence/2026-10-23-staged-resume-cleanup.json`.

## Next action
1. Keep the healthy 1M root running; do not cancel or duplicate it while Kaggle/W&B continue advancing.
2. At segment end, inspect Kaggle side effects, W&B summary and immutable checkpoint artifact before any retry.
3. If the root segment later creates a checkpoint but its old-code automatic continuation fails, recover from that immutable checkpoint using source `b32d18e` or newer; do not retry an unchanged old-commit child that can hit the disk-space bug.
4. When a segment reaches the full 20k budget, allow the separate CPU-only visualization/evaluation stage; do not label interim evaluation as final.
5. Preserve BASE/PQL scientific settings, RNG, environment state, checkpoint identity, and all currently active production jobs.

## Continuation rules
Read `specs/2026-10-23-training-handoff-replay-design.md`, `plans/2026-10-23-training-handoff-replay.md`, and the newest evidence JSON before acting.
Use credentials only through trusted helpers; never print tokens or raw secrets. A W&B missing run is not proof that Kaggle failed. A Kaggle timeout is not proof no kernel was created. Always check side effects before retrying.

## Fresh W&B cohort audit — 2026-10-23T06:52Z
- Correction: there is NOT a four-run production cohort at replay1M/hybrid75. The only production learner with `configs/comparison-20k-hybrid75-1m.yaml` is MASAC-PQL W&B `fc14508c`; it is running and reached global_step 196608 at the latest check. Gate `31831ab2` and preflight `9a716926` are finished diagnostics/benchmarks, not extra training cases.
- The original four-way env4096/batch8192 fair cohort is replay500k on CUDA, group `uav-network-v3-full20k-env4096-steps1800-seed44-20261020`: MASAC BASE `29e6add6` finished by time budget and has replacement `b9d73bd3` running; MASAC PQL `f5cfb3f9` crashed and has replacement `b88d7df8` running; MATD3 BASE `3070f38e` and MATD3 PQL `bda3bccd` finished by time budget and have no replacement/continuation currently running.
- No expected `c01` continuation kernel exists for the original MASAC BASE, MATD3 BASE, or MATD3 PQL roots. Those roots were pinned to `71e11e9`; the subsequent `08f7f3a` cloud-handoff fix added exact regression coverage for the Kaggle CLI outdated-version warning contaminating JSON plus fixed W&B cleanup ordering so W&B cannot report finished before learner resources close. The unavailable parent wrapper logs prevent assigning one post-run exception to each root, but the handoff path on the old commit is conclusively obsolete and no child was created.
- Original MASAC-PQL `f5cfb3f9` is different: W&B itself is `crashed` at global_step 196608 after ~0.99h, with no checkpoint. Its old account_22 token currently cannot authenticate, so exact Kaggle crash output cannot be retrieved from that account.
- Current valid quota >12h: account_27 22.62h (active 500k PQL), account_21 17.78h (active 1M PQL), account_25 16.85h (verified idle), account_28 16.80h (active 500k BASE). Therefore only account_25 is presently both idle and suitable for another full ~10h segment without stopping a healthy job.
- Do not claim the current 1M MASAC-PQL pilot is a complete four-way comparison. A true four-way 1M/hybrid75 comparison still needs MASAC BASE, MATD3 BASE and MATD3 PQL launches (and consistent infrastructure-fix provenance recorded), while preserving the active jobs.
