# Current task: live W&B audit and requested hybrid training

Observed 2026-10-23T04:56:33Z. User explicitly requested Kaggle for CUDA.

## Verified state
- Code fix commit `9499964d643a7d00972c15046873cd90a179c749`, published on `feat/cpu-replay-20261022`, GitHub read-back verified.
- Full tests: 659 passed, 74 skipped, 8 warnings, 1218 subtests in 812.73s (`/tmp/uav-hybrid75-final-suite.log`). Focused visualization 9 passed. Three Important independent review findings reproduced RED and fixed GREEN.
- Requested config `configs/comparison-20k-hybrid75-1m.yaml`: replay 1M, CPU fraction .75, env4096, batch8192, 20k episodes, seed44, Scenario V3. Production hybrid constructor is connected to BASE/PQL; cloud no longer overrides placement. Keep all scientific settings unchanged.
- Actual two-T4 gate PASSED: Kaggle `ewkljfelweow/uavnet-benchmark-20261023-044345-84cefe`, W&B `31831ab2`, source 9499964. Full replay CPU 9,831,000,000 bytes + GPU 3,274,000,000 bytes. 15.5GB checkpoint saved, exact RNG and environment restored, learning resumed to32 updates, checkpoint unchanged. Result in `evidence/2026-10-23-hybrid75-4096-1m-kaggle-result.json`. Scope is memory/execution with duplicated replay, not quality training. Minimum measured effective host headroom about1.02GB during restore; production should retain monitoring.
- Account21 (`ewkljfelweow`) fresh idle, gate complete, 19.47 GPU hours remaining. No requested hybrid production job submitted yet.
- Existing 500k production remains untouched and running: MASAC BASE `29e6add6` step7004160; MASAC PQL replacement `b88d7df8` step126976; MATD3 BASE `3070f38e` step5308416; MATD3 PQL `bda3bccd` step6410240. Never cancel or duplicate them.
- Old 2048-env sessions stopped at2048/20000 episodes and failed continuation due insufficient quota (~.45h), rather than finishing final evaluation. Their checkpoints remain available. W&B summaries corrected to show failed GPU continuation. Key.txt W&B credential valid.
- Interim CPU MASAC BASE evaluation FINISHED: Kaggle `vuliu123456/uavnet-validate-20261022-104625-33-cpu`, W&B `9a9895ea`; one episode fromcheckpoint483a83a5, media uploaded to HF/W&B. This is not final20k evaluation.

## Next action
1. Commit/push this gate evidence, then launch exactly one actual MASAC PQL training on idle account21 using the standard cloud CLI, requested config, auto-continuation and CPU visualization. Use Python `main([...])`; `python -m uav_network.cloud.cli` has no entrypoint and does nothing.
2. Capture actual manifest/kernel/planned W&B ID, distinguish preflight from actual training; only claim training is active once W&B steps advance. Never launch a duplicate following a timeout.
3. Recheck existing production at session end; do not assume continuation succeeded from W&B finished. Preserve checkpoint/config/algorithm identity for any recovery.
4. Update this record, evidence and plan with actual live status, commit/push and read back GitHub before final response.

## Continuation rules
Read `specs/2026-10-23-training-handoff-replay-design.md` and `plans/2026-10-23-training-handoff-replay.md` for this task. Audit evidence: `evidence/2026-10-23-training-handoff-audit.json`.
Use existing credentials through trusted helpers; never print secrets or store arbitrary exception strings. Do not change quotas/accounts/keys. For account lookups use subprocess-backed `run_kaggle`; `resolve_username` mutates process environment and is unsafe concurrently.
Frozen legacy comparison intentionally rejects hybrid; normal BASE/PQL supports hybrid. Existing500k cohort stays unchanged. Onlyone free account had sufficient quota for a new12h session; do not claim four new1M runs or guarantee20k completion within current quota.
