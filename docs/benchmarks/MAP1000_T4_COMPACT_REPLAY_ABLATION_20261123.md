# MAP1000 T4 Replay optimization — optional, 2026-11-23

## Baseline and branch

- Parent corrected science commit: a8da8e9e1912dcd0f599ae108b1758824ac24faf.
- Performance branch: perf/map1000-t4-parity-safe-20261123.
- Original YAML and 75% CPU / 25% GPU Replay are NOT modified.
- Separate experimental YAML: configs/matd3-nooracle-map1000-t4x2-gpu-fp16-replay-ablation-20261123.yaml.
- No training or final held-out test on Kaggle T4 has been performed for this experimental storage layout.
- No old replay checkpoint may be reloaded into the fp16 layout without an explicit conversion/verification.

## Implemented

1. MeasuredReplay.sample: mark sampled rows as seen with direct idempotent assignment rather than sorting/unique. Duplicate sampled indices are intentionally permitted; exact sample data, RNG and ages remain equal.
2. New config runtime.replay_storage_precision defaults to fp32. Valid fp16 only with replay_device=cuda and replay_cpu_fraction=0; leaves default training unchanged.
3. Optional compact Replay stores observations, next observations, centralized state, next state, and continuous actions as FP16; destination as int8 for up to 127 UAVs, reward and terminated as FP32. On sample, tensors are converted back to learner-facing float32 / int64. This is lossy storage even though algorithm tensors retain original training dtypes.
4. For 6 UAV, observation dimension 228, state dimension 830, continuous dim 4: 1M transitions require about 8.246 GiB for compact tensor storage, compared with 16.518 GiB for dense FP32/long. MeasuredReplay metadata adds ~9MB, plus temporary minibatches and CUDA allocator overhead.
5. The original hybrid 75:25 Replay and full packet-level Warp environment stay unchanged.

## Evidence

- Direct seen marking vs unique: 300 CPU batches × 8192 transitions: 0.020448 s vs 0.165231 s (8.08× faster for isolated marking only), exact semantic parity.
- Compact CPU tests: sampled tensors remain float32/int64, checkpoint round-trip retains quantized rows, under 60% of dense storage bytes, and scientific YAML sections are unchanged.
- NOT VERIFIED: CUDA memory allocation on T4 GPU1, end-to-end Replay throughput, learner update throughput, 20k episode training convergence, or held-out validation/test.

## T4 acceptance criteria

- Confirm >=2 GPUs, device 1 free VRAM BEFORE allocating capacity=1M.
- Record torch.cuda memory_allocated/reserved/peak and genuine GPU0/GPU1 placement.
- Compare 75/25 and FP16 GPU replay over same 8,192 batch ×4 update workload, same seed, same warmup, fixed stream sync and CUDA events.
- Ensure no OOM under optimizer/snapshot/eval peak. Avoid disabling validation/logging/network.
- FP16 storage is a numerical model change: measure maximum observation/state/continuous quantization error and policy metric deltas across >=5 independent seeds before claiming equivalence.
- Reject optimization if GCS delivery, confirmation precision, discovery AUC, coverage, or mission success deteriorate beyond pre-registered tolerances.
- Do not claim the 8.08× isolated marking improvement accelerates entire training by 8.08×.

## Quick tests

python -m pytest -q tests/test_compact_gpu_replay_ablation_20261123.py tests/test_pql_replay_no_unique_20261123.py
python -m scripts.bench_seen_unique_20261123 --device cpu
