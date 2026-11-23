# NoOracle V9 — exact 8192 physical worlds per T4, 100 seeds/checkpoint

## Intent and science lock

A pre-training candidate derived from validated CT095 H256x256 Test1000 source, **not** a 20,000-episode training launch. Keep NoOracle sensor (Bayesian 0.95 threshold), 6 UAV, 35 targets/25 obstacles, 1,000m scene, 1500-step horizon, RF network, MAC packet code, reward, energy, shared actor, actor BN/plain Critic, replay mix, E4096/U4/B8192, seed44 and Test1000 unchanged. This branch alters only the after-training CUDA Eval placement and evaluation pool logic. All historical checkpoints and runs must remain untouched.

## Deterministic full-pool Eval

- Every T4 worker allocates **8192 physical WarpVectorEnv** once and reuses across checkpoint batches. 81 checkpoints × 100 scenarios = 8100 real evaluation environments per full batch; 92 additional physical worlds are initialized with deterministic **unused filler seeds**, run with zero actions, and **are never scored as valid episodes**.
- Checkpoint 1 (0-based index0) → world slots [0,100), checkpoint2 → [100,200), checkpoint81 → [8000,8100). 92 filler worlds follow.
- GPU0 takes checkpoint indices 0..80, GPU1 takes indices 81..161; subsequent rounds GPU0 162..242, GPU1 243..323, etc. Final partial block consists only of true checkpoint count, with unoccupied pool slots excluded from metrics.
- Every checkpoint uses identical seed bank [1000000044,1000000143] and is evaluated in deterministic Actor-only mode. Per-world beliefs, GPS/targets, obstacles, PHY/MAC and mission state are initialized by WarpVectorEnv.reset(seed=full_list). Actor policies are evaluated via torch.vmap; no replay, Critic, optimizer, training update or true target ID injection.
- Completion requires one result per snapshot, checkpoint SHA validation and exhaustive manifest; rerun skips only SHA-matched checkpoint records. Validation selects best checkpoint according to unchanged selection logic. **Final Test** continues to use 1000 independent held-out episodes on two T4s, 500 per GPU, seeds [2000000044..2000001043]. No Test-to-checkpoint-selection leakage.
- The original strided eval assignment remains default for unrelated experiments. The candidate explicitly sets `--checkpoint-assignment contiguous_batches` and `--eval-world-pool-size 8192`.

## GPU physical capacity caveat

Warp state footprint estimated at **1,021,308 bytes/world**, so 8192 worlds require approximately **7.79 GiB/GPU for Warp state only**, excluding PyTorch, the secondary observation/state buffers, CUDA kernels, allocator fragmentation and actor inference. A Kaggle GPU test is **required** to establish T4 16GiB feasibility and true throughput. Never mark an unrun test as passed.

## A/B evidence and rejection gate

Existing GPU random-action environment benchmark measured V9 original 4236.71 vs V1 7924.34 transitions/s (+87.04%), not full training. CPU V1 vs V2 median 2543.79 vs 2766.98 (+8.77%) in a prior run. A new dynamic-fragment MAC loop candidate, although bitwise WorldState-hash-equivalent over 4 worlds ×80 steps, measured **2574.98 (V2) vs 2363.77 (V3)** median CPU transitions/s over 3 paired repetitions — **-8.20%**. **Rejected and reverted** from this candidate branch. Raw A/B report: `/home/aduy/Downloads/V9-V2-V3-DYNAMIC-FRAGMENT-CPU-AB-20261123.json`.

## Acceptance criteria

1. CPU tests pass for exact slot assignment across 0..600 checkpoints, seed isolation, old defaults and padded eval metric equality.
2. CPU NoOracle, radio packet, resume, MATD3 train mock fullflow and W&B/HF mock tests all pass.
3. GPU T4x2 real 8192-world allocation and 1-step smoke and CUDA 4096 env V2 A/B measured successfully. If GPU OOM/failed, *do not* promote to approved notebook or silently reduce to 200 worlds.
4. No CUDA optimizer/sensor changes without independent CPU & T4 parity+speed evidence. Production training waits for user approval.
