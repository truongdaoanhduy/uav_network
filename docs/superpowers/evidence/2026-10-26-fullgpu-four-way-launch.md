# Additional full-GPU replay cohort: two training, two OOM

Observed 2026-10-26T09:41:59.948575+00:00; independently checked 2026-10-26T09:42:52.981766+00:00. User authorized four OTHER Kaggle accounts for the same MASAC/MATD3 BASE/PQL, changing only replay storage to GPU. **The requested four-way launch was attempted, but four healthy production runs were NOT achieved.** No active cloud job was cancelled/restarted. No HF/W&B cleanup/deletion.

| Case | Account | Kernel | Actual outcome | W&B |
| --- | --- | --- | --- | --- |
| MASAC BASE | account_18 | fowieofweoi/uavnet-train-20261026-093208-d80760 | failed at warmup131072;0updates;CUDAOOM | b18d6a0d |
| MASAC PQL | account_19 | kjkjhhfghjjkn/uavnet-train-20261026-093213-22e669 | mandatory preflight benchmarkOOM;production never started | benchmark8b988c64;reservedproduction003e327e absent |
| MATD3 BASE | account_20 | kjhjvnvjgyu/uavnet-train-20261026-093210-d2dbd8 | RUNNING;step835584;updates1376 | 8c599ca4 |
| MATD3 PQL | account_21 | bhbjbnmku/uavnet-train-20261026-093209-a779b5 | RUNNING;step745472;updates1200 | cfad2583 |

## Identity and unchanged scientific settings

Pinned sourcef39f45682b0b8a387cf8d95906c6730bf4337742, same clean detached worktree as hybrid cohort. New group `uav-g2-o3-full20k-e4096-b8192-r1kb-fullgpu-20261026`. Account18-21 each30GPUhours with no owned kernels and distinct owners at fresh preflight. Source/scientific config unchanged. Config layout diff is exactly `runtime.replay_device: cpu -> cuda` and `runtime.replay_cpu_fraction: 0.75 -> 0.0`.

Same20kepisodes,1800steps,4096env,batch8192,replay1M,seed44,1KiBreport,observation3/generation2. No warmstart,AMP/compile/fusedoff. Newtrackinggroup/output/extra fullgpu tag distinguish cohort. The inherited cpu75-gpu25 tag remains in submitted metadata; runtime.replay_cpu_fraction=0 and replay_device=cuda are authoritative. Do not interpret that inherited tag as actual replay placement.

All four actual execution hosts report2TeslaT4,14.56GiB/device. Replay transition estimate13,096,000,000B (12.197GiB), plus bookkeeping; MATD3 actual logged replay/bytes=13,105,000,000B. Source make_replay selects single CUDA learner-device buffer when fraction0/devicecuda. GPU0/1 VRAM is not one pooled32GB allocation.

## Confirmed failure and boundary

MASAC BASE traceback: training.learn -> MASAC.update_critic -> _expected_q_all -> vmap critic -> linear allocation. Requested336MiB with244.81MiB free onGPU1;14.32GiB already used. MASAC PQL mandatory40stepbenchmark returns statusoom, requested336MiB with176.81MiB free,14.39GiB used. This is capacity pressure at the destination-enumerating critic operation, not evidence of a Dec-POMDP leak or invalid reward/scenario logic. Original failure artifact root_cause_confirmed=false is preserved; these direct OOM observations establish the memory failure, not a claim every possible memory optimization has been tested.

PQL's benchmark W&B run8b988c64 differs from the reserved production003e327e because the preflight stopped execution before production. Do not report003e327e as running or assume it exists. BASEfailure traceback and PQLbenchmarkerror/hardware are in adjacentstatusJSON. Full downloaded logs remain under outputs/cohort-fullgpu-g2-20261026/kaggle-failures/. No failed state warm-started.

## Controllers and preservation

All four controller kernels were submitted; both MATD3 controllers have fresh independent running/phase training/matching source-root-current GPU/heartbeat<=1800s checks. Auto-continuation and finalCPUeval30episodes seed2000000044 configured. This is current readiness, not guaranteed platform/quota/final-completion proof. MASACcontrollers may remain in training until next900secondpoll observes failed parent; those phases do not prove MASAC training is healthy or that visualization can run after failure.

The original account14-17 hybrid runs were freshly confirmed RUNNING and progressing; their records are in adjacentstatusJSON. No tuning/batch reduction/replay reduction/critic-vectorization change was silently applied. Original four and active MATD3 jobs are preserved.

## Next action

Continue monitoring the two fullGPU MATD3 lines and original four hybrid lines. Both fullGPU MASAC lines are blocked by proven VRAM failure with this unchanged configuration. Before a new MASAC attempt, validate an explicit memory strategy on2T4 (e.g. mathematically equivalent sequential/chunked critic execution), then document any execution-setting/source difference. Do not blindly retry the unchanged failing setup, lower scientific batch/replay budgets, duplicate active jobs, or claim this cohort completed. No source fix was made in this task.
