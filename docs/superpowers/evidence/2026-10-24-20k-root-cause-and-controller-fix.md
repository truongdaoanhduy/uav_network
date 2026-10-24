# 20k episode root-cause and continuation fix — 2026-10-24

## Scope

Fresh W&B/Kaggle/code audit for the canonical 20k × max-1800 training lineages. Scientific scenario settings are unchanged. Healthy production kernels were not cancelled or restarted.

## Root cause

The canonical root segments did not finish the 20,000-episode budget. They terminated at the configured time budget, not because the episode counter, MASAC/MATD3 learner, replay capacity, or T4 hardware failed.

Fresh W&B root summaries:

| run | algorithm | episodes_completed | global_step | episode_budget_complete | stop_reason |
|---|---|---:|---:|---|---|
| 63faeffb | MASAC BASE | 650 | 6,983,680 | false | time_budget |
| fc14508c | MASAC PQL | 1,066 | 7,045,120 | false | time_budget |
| 2dcbf36c | MATD3 BASE | 0 | 3,825,664 | false | time_budget |
| 278634f6 | MATD3 PQL | 0 | 3,465,216 | false | time_budget |
| b9d73bd3 | MASAC BASE 500k | 4,096 | 10,309,632 | false | time_budget |

The canonical config explicitly has `runtime.max_train_seconds: 36000`. Kaggle's notebook execution ceiling is larger, but the trainer intentionally stops earlier to reserve time for checkpoint/upload/continuation.

Current continuation timing still shows collector dominance rather than learner dominance. Examples from fresh W&B summaries:

- MATD3 BASE c01 `7a57e2a5`: collector ≈31,025.9 s vs learner ≈1,760.7 s.
- MATD3 PQL c01 `bdb9a503`: collector ≈30,748.5 s vs critic ≈1,683.3 s + policy ≈501.1 s.
- MASAC BASE c01 `a8ac9ffc`: collector ≈30,453.9 s vs learner ≈2,788.6 s.

The packet simulator is therefore the dominant runtime path. A 1,000,000-byte report at ~1,024-byte payload is 977 fragments. With 35 targets, the workload is amplified further by CSMA/CCA, ACK, retries, collision, interference, PHY, relay and propagation. Current cohort counters contain hundreds of millions of DATA attempts and tens/hundreds of billions of CCA checks.

Zero completed episodes in an early vectorized segment is not by itself an episode-counter bug: 4,096 synchronized environments can accumulate millions of transitions before the first complete wave of long episodes reaches termination.

## Correct resolution

A single 10-hour training segment is not a valid mechanism for a 20k budget at the measured simulator throughput. The correct infrastructure fix is exact, same-account multi-segment continuation:

`time_budget -> full checkpoint -> immutable upload -> verify prefix -> same-account continuation -> repeat -> episode_budget -> CPU evaluation/visualization`

Completion remains gated on all of:

- `episodes_completed >= 20000`
- `episode_budget_complete == true`
- `stop_reason == episode_budget`

No scientific setting is reduced to make the run appear faster.

## Controller regressions fixed locally

Two orchestration failure modes were reproduced against pre-fix HEAD `44fdd43`:

1. A credential/pre-push failure left `submission_intent`, causing later controller iterations to wait for a child that was never submitted.
2. A Kaggle push receipt alone was treated as unattended readiness; there was no requirement for an actual running controller heartbeat.

New behavior distinguishes `preparing` from `pushing`, clears retry-safe pre-push intent, preserves ambiguous post-push intent for reconciliation, and requires an actual Kaggle ACTIVE state plus a fresh matching W&B heartbeat before `unattended_ready=true`.

A third malformed-control regression was found by the focused suite: a non-object HF media file record could be treated as recoverable incomplete output and trigger an unnecessary Kaggle lookup. The validator now rejects malformed media records before recovery decisions.

### RED evidence

The three controller regression tests fail on detached pre-fix HEAD `44fdd43`:

- pre-push credential failure leaves a phantom `submission_intent`;
- launch readiness test cannot find `wait_controller_ready`;
- controller health test cannot find `wait_controller_ready`.

### GREEN evidence

Current focused controller/continuation suite: `56 passed, 38 subtests passed`.
Critical regression subset after the malformed-control fix: `4 passed, 6 subtests passed`.
`git diff --check`: clean.

## Cloud state during audit

Fresh Kaggle state preserved the live lineages:

- MASAC BASE c01: running
- MASAC PQL c01: error; recovery c02: running
- MATD3 BASE c01: running
- MATD3 PQL c01: running
- MASAC BASE 500k c01: error; recovery c02: running

Account 35 was checked for the isolated cloud smoke: about 29.92 GPU hours remained and no active kernel was found at that check.

## Full local verification

Full suite after the controller fixes: `707 passed, 74 skipped, 8 warnings, 1250 subtests passed in 702.57s`, exit 0.

## Pending before COMPLETE

- immutable commit + push;
- real isolated Kaggle cloud E2E smoke with controller heartbeat and final CPU/HF/W&B handoff;
- only after that, attach/adopt controllers for the production lineages without restarting healthy GPU jobs.
