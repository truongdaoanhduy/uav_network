# V9 `e079eace` independent Eval100 / Test4000, baseline-sized Warp batches

This job is **evaluation-only**. It does not train or modify the source W&B
run `e079eace`, Hugging Face checkpoint repo, or an active Kaggle kernel.
It creates a **new W&B run** and optionally publishes results under a unique
Hugging Face dataset path.

## Kaggle

Notebook: `notebooks/MATD3-NOORACLE-V9-e079eace-REVAL100-TEST4000-BASELINE-BATCH81-T4x2-KAGGLE.ipynb`.

Enable **GPU T4 x2** and Internet. Grant **WANDB_API_KEY** and **HF_TOKEN** Kaggle Secrets.
Run all three cells; there is no need to supply `key.txt` or start training.

The notebook embeds the exact Python source archive from the original NOORACLE-V9
Kaggle notebook (source archive SHA256
`00e2897f4e554f4cd1c48cf6ed6abee998eb654e5c1c3c70188e0a8378c93f5d`).
It downloads `deferred-gpu-eval/e079eace/deferred-gpu-eval.tar.gz` from the
public Hugging Face **dataset** `duy95/uav-search-diagnostics`, unpacks it
safely, and verifies per-snapshot SHA256 before evaluating.

At inspection, the true archive has 157 distinct Actor snapshots, 0 pruned.
The same fixed 100-vector-step checkpoint grid as the source job plus its
extra recorded candidates selects 74 Actor snapshots.

## GPU allocation

- **Val100:** 74 checkpoints, each with the **same fixed 100 validation seeds**.
  GPU0 handles indices 0,2,...,72; GPU1 handles 1,3,...,73.
  Default `CHECKPOINTS_PER_GPU_BATCH=81`: each GPU allocates 37×100 = **3700
  Warp worlds in one batch**, not 8192. The 8192 knob is a maximum guard.
- **Best Actor:** exactly the V9 validation rule: prioritize success rate,
  GCS delivery, discovery, then mean episodic return.
- **Test4000:** deterministic 4000 held-out seeds. GPU0 does the first 2000,
  GPU1 the remaining 2000; **one 2000-world Warp batch per GPU**.
- Uses original Actor-only `torch.vmap`, Warp physics, 1500-step max episode,
  no Critic, no replay buffer, no backward, and no training.

## Where results are stored

The Kaggle output folder is
`/kaggle/working/v9-e079eace-baseline-batch81-evaltest/`:

- `benchmark_summary.json` and `benchmark_status.json`: elapsed validation,
  test, total duration; simulated world steps/s; peak allocated PyTorch GPU
  memory; seed/run/archive hashes; source and benchmark run provenance.
- `gpu_batch_eval/eval_*.json`: true 100-episode metrics/checkpoint, with
  per-batch elapsed time and peak allocated memory.
- `gpu_batch_eval/validation_results.json`, `.csv`, `best.pt`, `best.json`,
  `final_test.json`, `report.json`.

Benchmark W&B run is generated with a fresh ID and the name
`MATD3-V9-VAL100-TEST4000-BASELINEBATCH-e079eace-<new_id>` under
`truongdaoanhduy-uit/uav_search_target`. Validation curves use actual Actor
checkpoint global-step x values; Test4000 is a held-out scalar summary, not
appended to the convergence line.

If HF uploading is enabled, result JSON/CSV files are written to
`duy95/uav-search-diagnostics/evaltest-benchmarks/e079eace/<new_id>/`.
The original `deferred-gpu-eval/e079eace/` archive is never modified.

## Comparing to current V9 and baseline `18d756ef`

Read **`benchmark/validation_wall_s`**, **`benchmark/test_wall_s`**,
**`benchmark/validation_simulated_steps_per_s`**,
**`benchmark/test_simulated_steps_per_s`** and
**`benchmark/gpu_peak_allocated_mib_observed`**. Compare with the original V9
2-checkpoint, 200-worlds-per-GPU schedule and `18d756ef` 35-checkpoint schedule.
Do not compare single-batch seconds alone, because batch sizes and numbers of
rollouts differ.

**Known limitation:** T4×2 VRAM feasibility and wall-time speedup at 3700
V9 worlds per GPU have not been GPU-tested. If OOM occurs, change
`CHECKPOINTS_PER_GPU_BATCH` in cell 2 to 32, 16, 8 or 2 and run an isolated
new benchmark; the notebook never silently downshifts or claims success.

## Tests performed before sharing

- Real 118,839,735-byte source archive fetched from HF; SHA256-verified,
  157 snapshots unpacked and verified. Original 74-checkpoint shortlist
  recovered without dropping/fabricating checkpoints.
- CPU `BatchedSharedActor` with real V9 snapshot weights and a real `torch.vmap`
  inference forward pass passed (2 policies, 6 UAVs, ActorBN).
- Offline pytest covers allocation, safe extraction, tar symlink/traversal,
  corrupt SHA rejection, independent benchmark identity, notebook structure,
  checksum and Kaggle size. Related regression tests passed.
- No Kaggle GPU benchmark has been run for the newly generated notebook.
