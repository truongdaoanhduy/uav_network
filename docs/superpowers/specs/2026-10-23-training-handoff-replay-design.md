# W&B training handoff and hybrid replay correction

User request 2026-10-23: verify live W&B, explain missing replay 1M (75% CPU/25% GPU), env 4096/batch 8192, fix real failures using existing credentials without exposing them.

## Evidence and scope
- Fresh W&B and Kaggle show three active production runs using 500k CUDA replay. MASAC PQL f5cfb3f9 crashed and account_22 is disabled; an already submitted replacement b88d7df8 is in GPU preflight. Do not duplicate or stop these runs.
- 1M hybrid75 exists only in cpu_replay_probe.HybridRing; production construction does not use it. Cloud remote appends replay_device=cuda and therefore overrides CPU choices.
- Old 2048-env MASAC sessions 483a83a5 and c77ded1d saved exact checkpoints after 2048/20000 episodes. Kaggle failed during continuation, not final evaluation. Their selected accounts currently lack enough quota for another safe segment. Existing failure records discard the reason and W&B looks finished.

## Design
Reuse the tested HybridRing as a production replay implementation behind explicit runtime.replay_cpu_fraction (default 0). Preserve single global ring, RNG, sampling, learner and scientific environment. Pure CPU/CUDA behavior is unchanged. CPU replay base storage remains pageable. Add a separate config for replay 1,000,000, CPU fraction .75, env 4096, batch 8192; keep existing cohort config unchanged. Honor config replay placement in every cloud entrypoint. Exact save/load and environment resume must pass before a production-shaped T4 gate. No guessed success, no silent fallback or batch reduction.

For continuation, retain structured safe failure codes and publish next-stage status to W&B after the training tracker closes. Do not copy arbitrary exception messages that could contain secrets. Distinguish session completion from episode-budget completion and scheduling completion. Preserve training results on handoff failure.

## Approaches
Use existing HybridRing and shared replay factory: selected, smallest change retaining measured semantics. A fresh cache implementation would duplicate tested logic. Merely changing YAML would not work because production and cloud currently ignore the hybrid layout.

## Verification
Regression RED/GREEN for placement propagation, production hybrid sample/checkpoint parity, continuation safe reason and status. Run project full pytest suite once changes stabilize. Verify live Kaggle and W&B before any launch. Use only an idle quota-verified account for the requested layout; if hardware gate fails, report the measured failure and preserve current jobs.
