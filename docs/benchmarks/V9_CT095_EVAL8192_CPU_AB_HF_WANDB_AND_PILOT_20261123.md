# NoOracle V9 CT095 Eval8192 — paired CPU evidence and HF/W&B audit

## Historical runs (read-only W&B + Hugging Face)

- 18d756ef: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef
  - 20,000 training episodes, train/joint_transitions_per_s=4599.758087294835, fullflow/complete=true.
  - Val100 per checkpoint, 81 checkpoints/GPU, max 8192 worlds/GPU, Final Test 4000 complete.
  - HF best: https://huggingface.co/duy95/uav-search-checkpoints/blob/main/deferred-gpu-eval/18d756ef/best.json
  - HF best validation delivery 32.9714%, search 36.8%, 100 seeds, step 24166400.
- e079eace: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e079eace
  - 20,000 training episodes, train/joint_transitions_per_s=1981.9121298981177.
  - At inspection: fullflow/complete=false, phase=gpu_validation. Val100, 2 checkpoints/GPU, cap 256 worlds/GPU.
  - HF best: https://huggingface.co/duy95/uav-search-checkpoints/blob/main/deferred-gpu-eval/e079eace/best.json
  - HF best validation delivery 15.5143%, search 22.4%, 100 seeds, step 13107200.
- 18d756ef and e079eace have different scenario_hash values and target-report semantics. The 2.32x historical throughput difference is NOT an A/B evaluation of equivalent physics.
- HF diagnostic logs were inspected from https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/18d756ef/diagnostic-archives and its e079eace counterpart. Their diagnostic manifests assert full_reconstruction_verified=true. This is archive integrity evidence, not speedup evidence.

## Matched CPU validation A/B on exactly the same CT095-H256x256 source and seeds

- 12 checkpoint x 3 seeds x 24 steps, alternating 3 pairs: reference median 2.081974530 s; Eval batch-pool candidate 1.908744530 s; speedup +9.0756%. Entire episode record SHA256 identical: 640f697d23a718013c3a9010998539fc41d9f429ba0b522bf34e3da105e165b1.
- 24 checkpoint x 4 seeds x 60 steps, alternating 3 pairs: reference median 4.111852156 s; candidate 3.605386672 s; speedup +14.0475%. Entire episode record SHA256 identical: 95afc1cf0389f83c7ee2ac7536f4b759880fd4edc64c7154bd247d29cc84fece.
- Source comparison: original worktree /home/aduy/Documents/NCKH/uav_network_v9_ct095_h256_test1000_20261123 (commit 61f667a) versus new isolated worktree /home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123. Both match NoOracle, confirmation=0.95, 6 UAV, 35 targets, 25 obstacles, 256x256 shared Actor BN, same fixed validation seeds and enabled Warp PHY/MAC.
- CPU tests scale checkpoint pack 2 vs 6, with 2 filler worlds in a 20-world pool; they do not allocate real 8192 GPU worlds or run production 1500-step horizons.
- Raw A/B reports in Ubuntu Downloads: V9_CT095_CPU_EVAL_AB_3PAIRS_12CKPTS_3EP_24STEPS_20261123.json and V9_CT095_CPU_EVAL_AB_3PAIRS_24CKPTS_4EP_60STEPS_20261123.json.

## Full-flow CPU and GPU trial

- tests/test_v9_eval8192_full_cpu_handoff_20261123.py: 7 genuine serialized Actor-only snapshots -> manifest -> two worker assignments -> grouped Validation -> actual BestCheckpoint publication and CSV/JSON -> held-out CPU final seed bank. Resume skips existing SHA-matched records. Production 1000-episode Test planning is validated for two 500-world GPU shards.
- Recommended trial: notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-GPU-SMOKE-NO-TRAIN-T4X2-20261123.ipynb.
- The trial has NO train calls or W&B/HF credentials. GPU0/GPU1 each allocate 8192 worlds and evaluate 81 deterministic random Actors with 100 seeds/checkpoint for only 12 smoke steps; 92 filler worlds excluded. Results are written to eval8192_smoke_report.json.
- Production training notebook is a DIFFERENT FILE: notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-CONTIGUOUS-TEST1000-CANDIDATE-T4X2.ipynb. It is deliberately NOT authorized/used during GPU trial.
- Repaired the stale bootstrap CHECKPOINTS_PER_GPU_BATCH==2 assertion; new production candidate checks ==81, with a regression test.
- No claim of T4 memory feasibility, production 1500-step equivalence, or increased training-loop speed until GPU results are independently measured.
