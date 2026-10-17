# Diagnostics v1: completed outcomes and fixed CPU checkpoints

The matched profile is `configs/comparison-25k-diagnostics.yaml`: 6 UAVs,
35 targets, 25 obstacles, 1,800 steps per mission, 25,000 completed episodes,
4,096 environments, seed 44, replay 500,000, batch 8,192 and 8 updates per
collection. Toggle only the algorithm and `pql.enabled` for the four cases.
Physics, reward, observations, actions, 1 KiB reports and 3 KiB buffers are
unchanged. Energy still uses the existing propulsion and RF power × airtime
model. This instrumentation does not add radio hardware, idle receiver, fading
or a new energy objective.

## Observed radio and report quantities

Native and Warp capture the same fixed-size schema in
`network.diagnostics`, `report_diagnostics` and `energy`. Episode and evaluation
JSON retain the raw fields and target first-discovery/first-delivery timestamps.
W&B receives scalar keys, including per-UAV battery joules/fraction,
propulsion/RF/total joules, buffer peaks and oldest currently held report age.
Missing legacy diagnostics have availability 0; absent statistics are not
invented measurements.

| Quantity | Population and units |
|---|---|
| DATA/ACK attempts and TX power | Actual emitted frames, including failed attempts; ACK is separate. Power is W, with count/sum/mean/min/max. Selected actions that never emit are excluded. |
| DATA SNR/SINR/RSSI/distance | Actual PHY decode decisions; dB/dB/dBm/3D metres, with individual counts. |
| DATA PHY success rate | Successes / decode count. Emissions still pending at termination need not have a decode yet. |
| Weak-signal/interference failures | Noise-only SNR fails, versus SNR passes but SINR fails: exclusive primary causes. Overlap/collision counters are additional observed populations. |
| LOS/NLOS and peer/GCS rates | Successes divided by the corresponding actual decode population. |
| Queue/lifetime/retry/energy denials | Actual native/Warp decision-site counts, not inferred permanent report losses. |
| CCA/backoff | Actual checks/busy decisions, random draws and freezes. Capture makes no new random draws. |
| Custody relay count | Successful whole-report peer custody transfers; distinct from ACK reception. |
| Direct/relayed GCS reports | First unique target deliveries with zero/positive prior peer hops; duplicates are excluded. |
| Hops and report age | Conditional on unique delivered reports; count 0 gives null mean. |
| First-discovery-to-delivery | First camera observation to first unique GCS completion, including loss and regeneration; seconds. |
| Action opportunity | Pre-radio interval after motion, propulsion charging, expiry and sensing; eligibility does not promise queue space or emission energy. |

Power histogram bins are <0.1, [0.1,0.2), [0.2,0.3), [0.3,0.4),
[0.4,0.5), ≥0.5 W; DATA and ACK are separate. Hop bins are 1 through 7,
then ≥8. Fixed-bin counts are logged as `.../bin_0` onward, not approximated
from action choices. Existing legacy network metric names remain available.
Existing actor observations retain their prior end-of-step timing; telemetry
does not change sensing/radio discretization.

## Completed-outcome windows

`cohort/*` pools completed outcomes in absolute transition windows of
`world.max_steps * runtime.num_envs` (7,372,800 in the new profile). All
completions in one collection are consumed before closing or comparing a
window. Records are ordered by seed/worker for reproducible summation.

Every window includes episode count, source start/end interaction steps,
partial flag and source label. It is a population of **completed outcomes in
that time window**, not a guarantee that all worlds started or completed
together. Resuming mid-window, final shortened windows, and completion batches cut by
the episode budget are partial. An
empty window has n=0 and null outcome means. `cohorts.jsonl` retains each
window, including when W&B is disabled.

Packet PDR is pooled received / generated packets; energy per GCS delivery is
pooled total UAV energy / unique deliveries. Energy includes failed missions
and excludes GCS RF. Raw numerators, denominators and availability episode
counts accompany ratios. Conditional hop/delay means are delivery-weighted.
Stat objects pool sum/count and min/max only over nonempty populations;
histograms add binwise. Per-UAV summaries show their contributing counts.

Legacy `episode/*`, `outcome/*`, rolling metrics and all-history `final_*`
retain their meanings. Rolling episode order is descriptive. A return-drop
warning compares only two closed, distinct, nonpartial windows with at least
`tracking.diagnostic_window` episodes each. Same-step neighboring workers
cannot create a training-regression claim. Warnings remain symptoms without
a confirmed root cause.

The main dashboard shows completed-outcome window means and fixed-checkpoint
validation, with explicit n/partial/source-step columns. Success/PDR fractions
are labelled as fractions; discovery/delivery percentages remain percentages.
In-flight `progress/*` continues to describe currently running worlds.

## Asynchronous CPU validation

The legacy episode interval defaults remain compatible. The new interval
`tracking.evaluation_interval_steps` defaults to 0, and the new profile sets
7,372,800 **joint transitions**. Every vector collection checks its absolute
threshold, even when no mission finishes. A resumed segment starts at the
next threshold above its saved global step.

There is one exec'd CPU child and no work queue. Busy thresholds are skipped
and counted rather than retried every collection. Thresholds crossed together
use the latest threshold, with the others counted as skipped. Lifecycle
events, failure/timeout/cancellation and busy/deadline counts are in
`evaluations/events.jsonl` and `evaluation_status/*`. Ordinary child exit,
missing/malformed/nonfinite results and launch errors are visible and
nonfatal. The child uses one BLAS/OpenMP thread and no visible CUDA device.

Each job gets a unique atomic evaluation-only actor checkpoint, source
global step, exact actor-update version, SHA-256, scheduled threshold,
completed-episode counter and seed bank. BASE copies actors at a stable
main-thread boundary. PQL reads its immutable complete mailbox snapshot and
version; it does not serialize live mutable critics or freeze replay.
The parent never constructs a learner for evaluation and consumes no
Python/NumPy/Torch RNG. The child constructs a CPU learner and strictly loads
actors; its unused fresh critics are never represented as trained.
Evaluation-only files cannot be used for normal training resume.

The timeout uses terminate, a 0.25 s wait, kill and a bounded 0.5 s reap wait.
If bounded termination has not yet reaped the child, its slot stays reserved;
no replacement is launched, and later polling records the confirmed reap.
Normal shutdown drains at most 10 s, also capped by remaining training soft
budget; failure shutdown cancels immediately. New submissions stop in the
last 10 s of that budget. This preserves the cloud artifact/continuation
reserve. An unfinished evaluation is explicitly cancelled at shutdown and
is not called completed. CPU capacity and the 300 s default timeout can limit
coverage; skipped/cancelled jobs are not replacement scores.

Arrivals log at the current training `global_step`. `evaluation/*` and legacy
`eval/*` use the custom `evaluation/checkpoint_global_step` W&B axis; status
uses arrival global step. Result JSON contains scenario/protocol, source
checkpoint identity, exact seeds, episode records, comparison summaries and
pooled telemetry. Mini validation does not animate; final evaluation's
existing visualization route remains enabled.

The new mini-bank is seeds 1,000,000,044–1,000,000,048 (5 scenarios).
The final bank is unchanged at 2,000,000,044–2,000,000,073 (30 scenarios).
They are disjoint. Protocol v3 accounts conservatively for both configured
legacy episode validation and new transition validation exposure once their
threshold is reached; this does not assert every scheduled job completed.
Resumed history with an unknown earlier bank remains unknown. Repeated mini
validation is validation exposure, not an untouched final test.

## Replay and local checkpoint storage

BASE and PQL use the same measured replay sampler, preserving the original
sample RNG draw and CPU pinned-memory behavior. Collection-age moments and
unsampled evictions have measured-use/eviction counts. Old BASE resume rows
without metadata remain unknown until replaced, never born-now or
assumed-unsampled. New BASE and existing PQL resume payloads retain
measurement metadata.

Replay adds 9 bytes per row (int64 birth collection + bool sampled), about
4.29 MiB for 500,000 rows, plus bounded scalar counters. Reported sample age
is in **collection iterations**, not policy versions or seconds. True
sampled policy age is explicitly unavailable. Existing PQL collector/policy/
critic lag metrics remain differences in update versions; no invented action
version history is added.

`tracking.local_checkpoints_to_keep` defaults to 2 and must be a positive
integer. After completed upload handling, only older generated
`episode_[0-9]{6,}.pt` policies in the current output directory are removed.
A vector batch writes one episode policy after its complete outcome batch,
rather than many identical-weight files for simultaneous checkpoint
boundaries. Final/latest/resume, evaluation snapshots, unrelated files and
warm-start inputs are preserved; remote artifacts are never deleted.
A new policy can transiently coexist with the retained two until upload
handling completes.

Failed atomic policy writes clean the temporary file while preserving the
last good checkpoint. The redacted original failure reaches stdout before
writing its failure report, so a second full-disk error cannot hide it.
Disk free/used bytes and checkpoint file bytes are logged and included in
the training summary. Atomic resume still needs old + new temporary full
replay headroom; retention does not reduce replay or weaken atomicity.

Private credential downloads now always select the exact verified filename
with `-f`, including a one-file dataset. Private owner, exact allowed file
set, downloaded content and credential equality verification remain intact.

## Verification scope

CPU tests exercise real native BASE and CPU Warp PQL training/evaluation for
both MASAC and MATD3. With evaluation enabled versus disabled, synchronous
model and episode hashes are identical. Additional tests cover immutable
snapshot/RNG/source semantics, process failure/timeout/reaping, resumed
schedule/busy bounds, null/count pooling, worker-order permutation, real
continuation and atomic storage/credential regressions. CPU tests do not
constitute local CUDA verification or a production throughput measurement.

Integration verification on 2026-10-17: `.venv/bin/python -m pytest -q -rs tests`
reported 422 passed, 47 CUDA-dependent skipped and 377 subtests passed in
283.25 s, with 8 existing dependency/controlled-failure warnings. Independent
reviews covered passive network capture and training/evaluation integration.
Kaggle CUDA startup and live telemetry require separate deployment verification.
