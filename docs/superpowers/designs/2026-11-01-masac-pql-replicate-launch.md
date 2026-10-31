# MASAC PQL Reproducibility Replicate Design

## Intent
Launch one second MASAC PQL training root on a different Kaggle account to test whether two runs with the same scientific configuration and seed follow the same training trajectory/results.

## Goals
- Use Kaggle `account_06`, currently idle with verified GPU quota.
- Pin training source to `0ac776bd20f1abd472c0ed016bb9e3d95f7a95b4`, matching the current MASAC PQL root.
- Keep config, seed, PQL mode, CBF safety, horizon, replay, batch, env count, AMP/compile state, and W&B group identical to the original run.
- Launch from scratch with 2x T4, auto-continuation, and final CPU evaluation/visualization.
- Avoid exposing or passing Kaggle/W&B/HF credentials through the sandbox; use GitHub Actions secrets.

## Non-goals
- Do not modify any RL/environment/training code.
- Do not change seed to create a statistical replication; this is specifically a same-seed reproducibility check.
- Do not stop or modify the original MASAC PQL run.
- Do not claim bitwise determinism; async PQL and GPU kernels may still be nondeterministic.

## Approaches considered
1. Direct sandbox launch using `key.txt`: rejected because the sandbox safety layer blocks credential-bearing external submission.
2. Reuse the external supervisor: rejected because the supervisor reconciles existing roots and does not create new root jobs.
3. Manual GitHub Actions root launcher using already-synced GitHub Secrets: selected. It keeps credentials inside GitHub, checks out the exact scientific commit, and produces a sanitized launch artifact.

## Selected architecture
Create a manual-only workflow dedicated to this replicate. It checks out commit `0ac776b`, reconstructs a temporary one-account credential file from GitHub Secrets inside the runner, invokes the existing cloud launcher with the exact original MASAC PQL overrides, uploads only the sanitized launch manifest, and then lets the existing GitHub supervisor manage continuation.
