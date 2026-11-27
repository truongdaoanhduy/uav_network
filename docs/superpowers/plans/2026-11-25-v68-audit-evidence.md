# v6.8 Reproducible Academic QA Evidence Plan

> Execution: RED → GREEN → full regression → immutable artifact; do not claim GPU readiness.

**Goal:** ship a single portable full-source ZIP that includes *actual* test logs and an explicit, source-grounded scientific bug ledger.

**Input:** user-provided v6.7 `code.zip`; no credentials or Kaggle account accessed. **Architecture:** preserve current folder packages, put test runner in `scripts/maintenance`, generated QA logs in gitignored `outputs/qa_evidence`, and source-only SHA256 manifests in their existing locations.

## Gates

- [x] Inventory source ZIP and verify missing QA logs (original contains zero `.log`/JUnit artifacts).
- [x] Execute untouched v6.7 test suite: 198 pass, 2 skip, 55 subtests.
- [x] RED regression for 8-UAV seven-hop path-vector advertisement; GREEN via instance-specific hop budget.
- [x] RED regression for missing QA log tool; GREEN via `run_qa_bundle.py`, `setup.sh --qa` and precise failure/blocked statuses.
- [x] RED regression for UQ8.8 saturated effective evidence at eta=0/0.001; GREEN with conservative finite-horizon bound.
- [x] RED regression for non-1s step misleading AoI clock; GREEN with fail-closed configuration guard.
- [x] Full regression, manifest and policy after changes.
- [x] Capture actual pytest JUnit+console, map, learner smoke, GPU blocker exit status, environment versions.
- [x] Write science audit with cited external sources and evidence/limitations by subsystem.
- [x] Package from clean source and verify extracted ZIP independently.

## Global constraints

1. ≤500 lines per source file, named source folders, config-driven changes and portable `setup.sh`.
2. Never include secrets, HF/W&B/Kaggle credentials, training checkpoints or bulky datasets in archive.
3. CPU PASS is not UavNetSim event parity, GPU Warp rollout or MATD3/MASAC convergence.
4. Keep evidence complete in ZIP, but do not commit generated logs to Git.
5. No destructive changes to user's Git worktree or original ZIP.
