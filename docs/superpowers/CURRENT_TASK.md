# Current task and session handoff

## Read this first

User intent: preserve the work across chats, continue from the unfinished task,
and use Superpowers with evidence rather than repeating completed experiments.
The next engineering task under discussion is **replay buffer capacity 1,000,000
stored in CPU RAM, with collection and learning on the GPUs**.
“Replay buffer” here is the RL experience store, distinct from UAV packet relay.

**Implementation state: IN PROGRESS. CPU placement, batch-only pinning, RNG guards and checkpoint RAM paths passed focused local tests.
No CPU replay full-buffer Kaggle memory gate has completed.**
The user explicitly requested execution on 2026-10-22 after reading the handoff.
This is a bounded change to the existing replay/device flow. Preserve production.
The CPU placement proposal is technically viable in PyTorch; its stability and
speed in this project remain unverified.

Repository: https://github.com/truongdaoanhduy/uav_network
Active implementation branch: `feat/cpu-replay-20261022`
Stable handoff branch: `docs/session-handoff-20261022` (fast-forwarded at checkpoints)
Handoff worktree at capture:
`/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`
Base commit: `ab60690b76aeaf7d0a72cb10e782480be0cd1df0`
Use `git rev-parse HEAD` for the handoff commit; it cannot contain its own SHA.

## First actions in a fresh chat

1. Fetch this branch, read `AGENTS.md`, this file, and `git status --short` /
   `git log -5 --oneline`. Preserve any existing edits. Inspect actual commits
   before deciding whether an unchecked task is still unfinished.
2. Read the snapshot in [training-snapshot.json](evidence/2026-10-22-training-snapshot.json).
   It is historical, not a live dashboard. Read the running monitor's latest
   output or W&B/Kaggle before launching, stopping, or diagnosing a job.
3. Continue **one task**: read [CPU replay next task](CPU_REPLAY_NEXT_TASK.md).
   Resolve the current design/review stage using the user's actual instructions
   and applicable Superpowers skills. This handoff does not invent a design
   approval or an execution-method selection.
4. Record each completed step with command, result, UTC time, evidence path and
   commit. If interrupted, record the exact next command and existing job ID.
   Commit/push updates before handing off. A task is complete only when its
   required verification has run and its result was read.

A new chat needs access to this repo and the relevant connected tools.
A chat does not start itself after this one ends. Submitted Kaggle jobs and
the existing server monitor have independent lifetimes.

## Completed work — use its evidence, do not repeat

- W&B audit of 20–21 October: 29 runs, 24,849 history rows plus console and
  failure records. Full findings and seven video links are in the
  [audit report](evidence/2026-10-22-wandb-audit.md).
- Hugging Face `duy95/uav-search-checkpoints` is public. Seven available
  checkpoint families and an additional MATD3 BASE policy were recovered.
  Anonymous download and checksums were verified in the previous session.
  Public-default fix is the base commit above, on branch
  `fix/public-checkpoint-storage-20261022`.
- Seven intermediate videos completed, each one episode at seed 10044.
  They are observational evaluations, not a matched learning-quality ablation.
  MASAC PQL 1M had no saved policy before its OOM.
- All four 500k training segments were resumed from full state, with optimizer,
  replay, RNG and exact in-flight environment. Increases in transition counts
  **and learner updates** were verified. New run IDs and timestamps are in
  the snapshot. Preserve these jobs while working on CPU replay.
- Gen1/rxindex tasks 1–5 and the short follow-up probe are finished.
  Keep Gen1. The short probe measured about +1.40% in a single small A/B,
  with only 10,240 occupied replay rows; it did not validate full replay.
  The long probe timed out at its own 3-hour limit before learning.
  Do not restart it as the next task.

## Established diagnoses and limits

- W&B `finished` means a normal process exit. Check
  `episode_budget_complete`, episodes and stop reason before claiming
  the 20,000-episode goal was completed. Diagnostic runs are not training runs.
- The old launcher allowed 10 hours of training inside an 11-hour job,
  reserving upload/continuation time. A Kaggle session ceiling is not a
  guaranteed duration for every job; distinguish configured timeout from quota.
- Three original 1M runs failed on Hugging Face private-storage exhaustion.
  MASAC PQL 1M failed on GPU1 CUDA OOM. Replay alone was approximately
  12.21 GiB (13,105,000,000 bytes including measured-replay metadata).
  The public storage fix addresses the former cause, not the OOM.
- Collector/simulator waiting dominated the old MASAC BASE 500k segment:
  about 9.24 hours waiting versus 45 minutes learning. CPU replay is a
  proposed VRAM fix; it does not fix that simulator bottleneck and may
  add transfer overhead.
- Existing 500k and 1M cohorts differ in AMP, compilation and code revision.
  Neither their training summaries nor seven one-seed videos prove a
  replay-capacity advantage. All seven videos had mission success zero.
- MATD3 BASE's later child `cb43b1a0` reached 138 episodes but failed before
  saving a matching full resume. Its public version-2 policy is model-only.
  The new exact continuation uses the older valid full state at 1,927,168
  transitions; do not combine its optimizer with the later policy.

## Running work and where to inspect it

At the captured observation all four runs were running and progressing:

| Algorithm | Source run | Current recovery run | Kaggle kernel |
| --- | --- | --- | --- |
| MASAC PQL 500k | 1e0c6017 | [b55009f1](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/b55009f1) | libiu123/uavnet-validate-20261022-104443-fe5080 |
| MASAC BASE 500k | 3efe3a0b | [143c7acc](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/143c7acc) | vuliu123456/uavnet-validate-20261022-104625-33f02c |
| MATD3 PQL 500k | 925d10c4 | [18e2823f](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18e2823f) | ergewh54/uavnet-validate-20261022-104712-566c39 |
| MATD3 BASE 500k | dc16b47e | [47bd6da9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/47bd6da9) | nhtbrg/uavnet-validate-20261022-104752-461dfb |

Production recovery source is frozen at
`316ac1ae38d987eccb9e41bfd416955a01d47a62`, with replay 500k, env 2048,
batch 4096, FP32, compile=false, and public checkpoint override.
Do not substitute the AMP/1M branch when interpreting these runs.

Existing server monitor: job `5e55cfb663fd`, named
`uav-500k-recovery-monitor-retry`.
Its files are under
`/home/aduy/Documents/NCKH/uav_network-full4-20261021/outputs/public-resume-20261022/`:
`current-state.json`, `monitor-history.jsonl`,
`all-four-training-progress-verified.json`, `recovery-monitor-v2.log`.
The monitor observes jobs; it does not itself restart training.
Training's existing auto-continuation is separate and depends on valid state,
successful storage and remaining quota. Check for a child before relaunching.

Other remote worktrees:
- `uav_network-full4-amp1m-20261021`: frozen AMP/1M source at `6de210f`.
- `uav_network-macgen2-20261021`: experimental rxindex at `f87c2d3`;
  keep it out of this task.
- `uav_network-public-hf-20261022`: this documentation branch, based on
  public-storage fix `ab60690`.

## Working constraints

- Use one task at a time, with fresh evidence and public Hugging Face storage.
- Preserve active production runs. A new CPU-replay benchmark is a separate
  job; inspect current jobs/quota first and reuse an existing test job.
- Preserve scenario, reward, MASAC/MATD3 equations, and native UavNetSim
  semantics. Oracle revision:
  `04daafb815eb377409b40b285574eeb62b9a8d58`.
  MARL selects each packet next hop; do not introduce automatic rerouting.
- Actual audited V3 run configs had **22 MHz** bandwidth. The older pasted
  prompt says 10 MHz and is stale on this point. Read each pinned run config
  before making a scientific comparison.
- Keep replay RNG/resume semantics explicit when changing CPU/CUDA placement.
  Never label a model-only warm start or device RNG reseed an exact resume.
- Use existing credential loaders; keep credentials, tokens and credential
  file contents out of git, logs and chat.
- Sandbox is the remote machine with the project. Scratch chat paths are
  temporary and are not the project checkout.
- If a tool is unavailable, state the blocker and save the next action.
  Do not claim code execution or launch a duplicate to hide uncertainty.

## Suggested skills

Use `superpowers:using-superpowers` when resuming; classify the CPU replay
change using `superpowers:brainstorming` and recover any actual approval
from the conversation. Use `using-git-worktrees` for isolation, `test-driven-development`
for code, `systematic-debugging` for failures, and
`verification-before-completion` before claiming success.
Use `executing-plans` if an approved plan is executed inline, or the selected
execution skill if the user chose another method.
Use Ponytail/full to reuse the replay/device paths and avoid new dependencies.
Use Context7 for PyTorch transfer/load semantics and GitHub for durable state.

## Progress ledger

| Task | State | Evidence / next action |
| --- | --- | --- |
| Audit, public storage, seven videos, 500k recovery | Completed in previous session | Audit report; snapshot for timestamped progress |
| Cross-chat handoff | Complete | Docs verified locally and read back through GitHub at commit `750ac02`; working tree was clean after push |
| CPU replay behavior and tests | Local placement and checkpoint tests passed | 48 passed, 2 CUDA skipped; full suite and 1M peak-RAM gate pending |
| Full 1M CPU RAM / 2-T4 checkpoint-resume gate | Not started | No CPU replay Kaggle job has been submitted |
| Production adoption of CPU replay | Not started | Requires measured stability and throughput evidence first |

Keep historical evidence timestamped. Replace this ledger's current state
after each task; add the actual verification command and commit.

Handoff verification: commit `750ac02bc6c8371c638d53fdf1598f9dac238f06` was pushed; GitHub read-back of AGENTS/current task and a base/head comparison confirmed the seven documentation files. See [verification evidence](evidence/2026-10-22-handoff-verification.json). This ledger update is a follow-up documentation commit; use branch HEAD for the latest record.

Execution started 2026-10-22T11:38:45.166191+00:00. User instruction: implement and mark progress as work proceeds. Next: baseline suite, then CPU replay placement RED→GREEN. Test logs: `outputs/cpu-replay-20261022/`.

Live implementation checkpoints: [CPU replay progress](evidence/2026-10-22-cpu-replay-progress.json). Baseline job `704f81c9cf80`; placement/pinning/RNG RED tests observed 10 expected failures and 2 CUDA skips. No production code modified yet.

Checkpoint 2026-10-22T11:48:39.396466+00:00: baseline 601 pass/72 skip; focused placement/pinning/RNG 24 pass/2 CUDA skip. PQL honors runtime.replay_device; CPU storage stays pageable; replay device/RNG is validated before copying rows. Next: low-RAM checkpoint save/load tests and implementation.

Checkpoint 2026-10-22T11:53:43.732665+00:00: checkpoint RAM RED→GREEN (5 expected failures; then 48 pass/2 CUDA skip including existing resume/environment tests). Restore uses mmap; full CPU save borrows storage only at synchronous drained boundary; default frozen snapshot remains independent, partial CPU PQL storage is compact. Commit before this checkpoint: `3d51fd3`. Next: bounded CUDA probe harness, its tests, full suite, then one Kaggle job.

Checkpoint 2026-10-22T12:12:57.602891+00:00: post-change full suite job `ddfa97777c82` completed with exit code 0: `622 passed, 74 skipped, 8 warnings, 1209 subtests passed in 493.37s`. The suite discovered `tests/test_cpu_replay_probe.py`. Existing 500k monitor `5e55cfb663fd` remains alive and all four observed recovery runs were still `running` with training progress verified at ~12:12 UTC. Two probe files remain untracked and preserved. Next single step: inspect probe harness and run focused probe checks; no Kaggle diagnostic submitted yet.

Checkpoint 2026-10-22T12:26:16.255023+00:00: CPU replay probe harness local gate completed. Review found two missing preflight constraints and fixed them TDD-first: explicit PQL topology `cuda:0` collector / `cuda:1` learner+policy, and exact 2×T4 hardware validation before large allocation. RED: `tests/test_cpu_replay_probe.py` = 3 failed, 2 passed. GREEN focused: `36 passed, 2 skipped, 12 subtests passed`. Fresh full suite job `5502835ff02e`: `623 passed, 74 skipped, 8 warnings, 1209 subtests passed in 485.06s`, exit code 0. `git diff --cached --check` passed. Next: commit/push probe harness and verify remote HEAD; only then begin Kaggle job/quota preflight.

Checkpoint 2026-10-22T12:29:26.154968+00:00: Kaggle preflight completed without submitting a CPU replay diagnostic. Fresh quota via the repo cloud CLI: `account_27` OK with 26.23 GPU hours remaining. Recent known account_27 diagnostic/profile kernels were checked: latest rxindex minimal = complete, long rxindex probe = cancelled, older task4/task2/task7/task6 = complete. No local manifest/result matching `DIAG-CPU-REPLAY-1M-MASAC-PQL` or `full_buffer_memory_execution` exists. Production monitor process remains alive; at ~12:28 UTC all four 500k recovery runs were Kaggle/W&B `running` with `training_progress_verified=true`. Next single action: prepare and submit exactly one bounded CPU replay 1M MASAC-PQL two-T4 diagnostic on an idle verified account (account_27 is eligible by quota), then capture its manifest/kernel/W&B IDs.

Checkpoint 2026-10-22T12:53:00.468273+00:00: standard redacted cloud path gained a bounded `--diagnostic cpu-replay-1m` mode because the ad-hoc launcher was blocked before side effects by credential safety. TDD RED: 3 failures (CLI flag missing, diagnostic spec fell into HF checkpoint storage, remote helper absent). GREEN: 3/3 new tests; focused cloud/probe regressions 39 passed + 2 subtests; fresh full suite job `375ac5b6f768`: `626 passed, 74 skipped, 8 warnings, 1210 subtests passed in 450.90s`, exit code 0; `git diff --check` passed. Diagnostic mode is constrained to benchmark+MASAC+pinned Scenario V3/no overrides, skips irrelevant HF checkpoint credentials, rechecks the selected account has no active kernel immediately before push, and dispatches the reviewed `uav_network.experiments.cpu_replay_probe` with pinned commit/W&B run ID. Next: commit/push this cloud path, verify remote head, then launch exactly one account_27 two-T4 diagnostic via `uav_network.cloud.cli`.

Checkpoint 2026-10-22T12:54:35.365871+00:00: cloud diagnostic code is committed/pushed at `af4a52b1f0eb54c0c7d0bddbcc11999bd7d52416` and verified through GitHub. Two attempts to perform the actual credentialed Kaggle submission were blocked by the tool safety layer before any Kaggle push request was sent: first the ad-hoc launcher, then the standard `uav_network.cloud.cli` launch invocation. Do not bypass this credential safety boundary. A local search found no CPU-replay diagnostic launch manifest/result under `outputs/kaggle`; no new diagnostic kernel ref or W&B run ID exists. The temporary ignored ad-hoc launcher scratch directory was removed to avoid false duplicate detection. Next action in an environment permitted to use the existing credential source: invoke the committed standard CLI once for `--stage benchmark --diagnostic cpu-replay-1m --account account_27 --required-gpu-hours 1.5 --credential-dataset ioeufwejewlfl/uav-cpu-wandb-credential --algorithm masac --config configs/comparison-20k-steps1800-scenario-v3.yaml --gpus-per-run 2 --accelerator NvidiaTeslaT4 --timeout-seconds 5400`; the CLI itself rechecks quota/idle status and refuses active-account duplicates.
Checkpoint 2026-10-22T16:29:23.162308+00:00: hybrid replay 1M 50/50 CPU/GPU diagnostic completed on Kaggle account_30. Kernel `jwkjfwejfeio/uavnet-benchmark-20261022-155902-445596` completed cleanly; W&B run `3716659c`; source commit `4926dc2b2ab5961cae09cd5b5b064415cde56f72`. Layout is 500k rows CPU + 500k rows GPU1 with one global ring/RNG. Result JSON status=`passed`; replay rows=1,000,000; replay CPU bytes=6,557,000,000; replay GPU bytes=6,548,000,000; exact RNG next-sample parity, environment restore, resumed learning to 64 updates, finite metrics, and checkpoint immutability all passed. Fresh local verification: focused hybrid/PQL/critic suite `43 passed, 2 skipped, 18 subtests`; full suite `632 passed, 74 skipped, 8 warnings, 1210 subtests` in 1090.62s, exit 0. Compared with CPU-only 1M: total probe time 426.79s vs 444.95s (-4.08%), host RSS peak 20.85 vs 24.53 GiB (-15.01%), restore 60.95s vs 65.24s (-6.57%), but GPU1 peak allocated 12.33 vs 1.41 GiB and peak reserved 14.22/14.56 GiB (~97.6% of T4 VRAM); effective available host/cgroup headroom fell to 0.39 GiB vs 1.03 GiB. Conclusion: 50/50 is feasible but materially riskier than CPU-only because GPU1 and total memory headroom are nearly exhausted. Do not adopt 50/50 for production without a lower GPU fraction or another measured margin-improvement step. Production 500k runs were not modified.
