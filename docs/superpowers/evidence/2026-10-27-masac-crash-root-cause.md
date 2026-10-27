# MASAC G3 crash diagnosis — external hard termination

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Observed terminal state

Four 26-Oct MASAC G3 learners are W&B `crashed`: `25eecfd1`, `63202ad9`, `3b39b0c8`, `6f0bcde3`. Their four controllers `45ff19d1`, `8e3786a8`, `16f5a5f3`, `4f61b7cd` are also `crashed`. The four MATD3 G3 lineages had already reached 20,000 episodes and remain `finished`.

W&B documents `crashed` as a run that stops heartbeating unexpectedly, distinct from `failed` (nonzero clean exit) and `finished` (successful exit).

## Evidence against an algorithm / host-OOM failure

- All 8 MASAC learner/controller heartbeats stop inside 11 seconds: 19:31:57..19:32:08 UTC.
- Last sampled W&B system metrics still show active GPUs and finite resource readings.
- `system.proc.memory.oomKills=0` on all sampled learners near the end.
- Run summaries retain finite learner metrics and checkpoint metadata; no fatal-error summary or Python traceback was produced.
- BASE training is wrapped in the repository SIGTERM lifecycle handler. A catchable scheduler SIGTERM/Python exception would execute failure reporting and `tracker.close(failed=True)`, producing a W&B `failed` run rather than an unannounced `crashed` heartbeat loss. That did not happen.
- All four launch receipts specify `timeout_seconds=39600`; the configured learner soft limit is 36,000 s. The runs stopped after roughly 6h23m, so the repository's own 10h/11h deadlines were not reached.
- Kaggle's current official Notebook documentation states 12h CPU/GPU execution time, so ~6h23m is not the documented normal maximum session boundary.

## Credential/account evidence

The worker accounts for the crashed MASAC cohort were accounts16–19. Fresh Kaggle CLI quota/auth calls for accounts16–23 all fail with Kaggle's authentication-required response. The same credential file had successfully launched the cohort earlier; before today's operational edit its filesystem mtime was 2026-10-24, so the observed post-run failure was not explained by a local token-file rewrite during training.

This does not expose Kaggle's private enforcement reason. Therefore the narrowest supportable conclusion is:

**The immediate W&B `crashed` state was caused by an external hard termination or connectivity/session/account event, not by a caught MASAC learner exception. The strongest correlated evidence is post-run Kaggle credential/account invalidation. Exact Kaggle-side reason (revocation, account restriction, platform termination, etc.) is not recoverable from the now-inaccessible private kernels and must remain unproven.**

## Applied mitigation

`/home/aduy/Documents/NCKH/uav_network/key.txt` was atomically updated without printing secrets:

- `account_01` and `account_02` remain enabled; both freshly authenticate and have ~23.42h / ~25.03h GPU quota.
- `account_03`..`account_24` are disabled because their current credentials did not pass the production account gate.
- file permission remains 0600.

This prevents the launcher from selecting known-dead credentials. It does not bypass Kaggle restrictions and does not revive terminated jobs.

## Recovery / hardening proposal

1. Add/refresh only legitimately authorized Kaggle credentials, then require auth + quota + idle-kernel + private credential-dataset access before launch. Do not rotate accounts to bypass a platform restriction or quota.
2. Keep the existing acceleration policy; no evidence links FP16/compile to these old off/off MASAC crashes.
3. Do not classify a W&B `crashed` row alone as an algorithm failure. Use failure report + fatal metric + system telemetry + synchronized controller state.
4. Existing periodic HF artifacts are light policy checkpoints. A hard kill can occur before a final resumable replay/optimizer checkpoint is uploaded. If exact unattended recovery from hard preemption is required, the next implementation task should add **bounded periodic resumable HF snapshots** with an explicitly measured storage/upload interval; do not blindly upload the multi-GB replay every 128 episodes.
5. A controller colocated on the same Kaggle account shares the account/session failure domain. For true account-failure recovery, place supervision on a separate compliant persistent control plane. Do not implement cross-account quota/bannable-account rotation as an evasion mechanism.
