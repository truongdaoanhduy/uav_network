# NoOracle V9 CT095 MATD3 — CPU training-speed A/B and T4×2 pilot

## Scope and historical evidence

The primary goal is faster **actual training**, not post-training Eval.

The historical W&B runs are distinct scientific scenarios:
- [18d756ef](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef): training 20,000 episodes complete, `train/joint_transitions_per_s=4599.7581`, fullflow complete. This is pre-NoOracle cell reporting.
- [e079eace](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e079eace): training 20,000 episodes complete, `train/joint_transitions_per_s=1981.9121`, W&B phase was `gpu_validation` and not marked complete when inspected. NoOracle cell hypotheses enabled.

The difference of approximately 2.32× is **not a controlled A/B**: the scenario hashes, observation/report semantics and networking work differ. Both checkpoint best.json and diagnostic archive manifests were inspected via [HF checkpoints](https://huggingface.co/duy95/uav-search-checkpoints) and [HF diagnostics](https://huggingface.co/datasets/duy95/uav-search-diagnostics).

Previous GPU environmental profiling at E4096 (random actions, no learner): original V9 ~4237 transitions/s, indexed-sensor sparse-MAC baseline ~7924 transitions/s. In the optimized version packet_step accounted for ~70.46% and Bayesian sensing ~22.36% of profiled Warp kernel time. This is **NOT full MATD3 training throughput**.

## CPU training experiments on source identical in science

Source baseline: `/home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123`.

Source candidate (isolated worktree): `/home/aduy/Documents/NCKH/uav_network_train_speed_opt_20261123`.

1. Hoisting report-independent RF precheck outside the per-report loop had bitwise WorldState parity (4 worlds × 80 steps). However CPU env 3-pair median only +0.35% with outlier noise, and 8-env, 24-episode **real MATD3 training was -2.12%** when combined with replay optimization. MAC change **rejected and reverted**; scientific MAC code is unchanged.
2. For hybrid replay, `self.seen[indices.unique()]=True` was changed to `self.seen[indices]=True`. Duplicated indices are harmless idempotent writes to the same True value. Exact RNG state, replay metadata and all sampled B8192 tensors **matched SHA256** between reference and candidate. Microbenchmark 3 pairs B8192 median replay sample wall time 0.38178 vs 0.37614 s, +1.50%.
3. With **replay-only** optimization, real 8-env CPU MATD3 training with horizon12, B16, U1, 24 episodes, 3 pairs was old146.2394 vs new149.9781 transitions/s, **+2.56%**. SHA256 of final model weights and all mission episode records identical.
4. With replay-only optimization, real 8-env CPU MATD3 training horizon150, 16 episodes, 2 pairs old377.8286 vs new378.5371 transitions/s, **+0.19%**. Same final model/mission episode checksum. This is within likely benchmark noise; no claim of substantial speedup.
5. New **opt-in** `UAV_DIRECT_GPU_COLLECTOR_COPY=true` avoids an extra clone of every Actor tensor when synchronizing learner GPU1 → collector GPU0. The source has a backward-compatible false default. CPU synchronization microbenchmark of 250 copies per repetition, 5 pairs: baseline median0.0784724 seconds vs direct0.0544585 seconds (**44.10% faster copy operation**), all Actor weight and BatchNorm buffer state identical. This result does NOT establish overall training throughput or actual P2P GPU benefit.

## Real code/logic safety

- Full NoOracle CT095 scenario: 6 UAV / 35 targets / 25 obstacles / 1000m / horizon1500, Warp PHY/MAC untouched, 0.95 confirmation, H256×256 shared ActorBN and plain Critic, E4096/U4/B8192, replay R1M with 75% CPU, seed44, deterministic flags.
- Candidate scientific code changes limited to `uav_network/algorithms/hybrid_replay.py` replay seen-index marking and opt-in copy in `uav_network/experiments/training.py`. No changes to target ID/no-oracle, energy/rewards, physics, observation, MAC/PHY, checkpoint rules, Critic updates, policy delay, actor normalization or RNG seeds.
- `tests/test_training_direct_actor_sync_20261123.py` checks all actor parameters **and BN running buffers** match. `tests/test_train_replay_skip_unique_20261123.py` checks direct replay seen assignment.
- CPU trial worker `scripts/trial_real_matd3_train_gpu_worker_20261123.py` ran on baseline and candidate under bounded 8-env x24 vector-steps test, B16/U1/horizon12, **model hash, episode hash, science hash equal** and budget stopped exactly at 192 environment transitions. The CPU-only development smoke uses reduced replay capacity to avoid local 7.5 GiB RAM; the GPU notebook **does not reduce R1M**.
- Full regression `64 passed, 1 skipped, 7 warnings, 9 subtests passed` after linking local UavNetSim source for source bundling.

## Kaggle real training-speed A/B notebook

File: `notebooks/MATD3-NOORACLE-V9-CT095-H256x256-TRAIN-SPEED-AB-E4096-U4-B8192-T4X2-PILOT-20261123.ipynb`.

This is **not production**. On Kaggle T4×2, runs 2 paired A/B (baseline,candidate,candidate,baseline); each trial uses GPU0 for 4096 Warp worlds and GPU1 for learner/replay, training a fresh Actor from seed44 for **64 vector steps** (~262,144 transitions), learning starts after131,072. Physics and horizon remain1500; learner U4 B8192 and R1M unchanged. Parent process sets `UAV_TRAIN_ENV_STEP_LIMIT` and verifies termination reason `env_step_budget`. W&B/HF and posttrain validation/test explicitly disabled in this speed test. Both model SHA256 and episode SHA256 are compared at end; any mismatch is a rejection. Output `/kaggle/working/v9_train_speed_ab_report.json`.

**GPU throughput, full CUDA env memory, and cross-GPU Actor transfer not yet measured**: user will run pilot and share JSON. Do not merge/promote until both same-science exact model parity and speed are demonstrated on T4, and until full Eval/Test1000 is revalidated in later non-production run. The original and final 20k notebooks remain unmodified.

## Kaggle notebook version 372716239 bootstrap failure and fix (2026-11-23)

Exact user Kaggle link: https://www.kaggle.com/code/therwed/notebook43d7d397be?scriptVersionId=372716239

The Kaggle status was ERROR. Pulled its Kaggle execution log from `kaggle kernels output therwed/notebook43d7d397be` and notebook source with `kaggle kernels pull therwed/notebook43d7d397be`. The pulled source cells matched the local producer notebook byte for byte. Kaggle detected both Tesla T4 GPUs, but cell [2] failed before install or training:

`tarfile.open(fileobj=io.BytesIO(lzma.decompress(blob)), mode="r:xz")` raised `LZMAError: Input format not supported by decoder` and `tarfile.ReadError: not an lzma file`.

**Root cause:** `blob` was already a valid, SHA-pinned `.tar.xz` archive (XZ magic `fd377a585a00`); applying `lzma.decompress(blob)` converted it to a plain TAR, then the `r:xz` reader tried to decode that TAR a second time. This was a notebook packaging bug, NOT a CUDA, Warp, NoOracle, network or MATD3 problem.

**Correction:** use `tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz")`. The new Kaggle cell explicitly validates the archive XZ magic. New regression test `tests/test_kaggle_train_ab_archive_unpack_20261123.py` invokes the **actual generated Kaggle unpack cell** (only omitting Kaggle-local venv installation) and verifies 500+ extracted files and every baseline/candidate patched byte and the pinned training worker. It first reproduced the original `ReadError`, then passed after correction.

The same regression now runs **both baseline and candidate actual CPU MATD3 training** from those exact extracted source directories, checks capped 192 transitions, optimizer updates, independently generated source-science and final-model SHA256, plus episode event logs. Both pass with identical hashes. GPU memory/speed has NOT yet been tested on Kaggle after the fix. The failed Kaggle version cannot be repaired in place: upload the newly generated fixed notebook as a *new Kaggle version*.
