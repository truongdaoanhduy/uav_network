# Verified W&B storage, GPU training and CPU visualization

Checked: 2026-10-15T07:42:16.865866+00:00.

Main workspace: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/workspace

Online comparison: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3cd22d52

## Confirmed cause and fix

The project inventory contains 42,119,446,756 logical artifact bytes. The original 20.7 GB warning preceded completion of the four pinned production jobs; each final full-state replay adds about 5 GB. Periodic policy versions also accumulate. A latest alias does not remove previous versions. Run histories are a small fraction of this total. Logical artifact totals and the billing banner may use different accounting.

New jobs default to tracking.checkpoint_store=huggingface. Policy and full replay/optimizer/RNG state live in the existing private checkpoint repository; W&B contains a small checkpoint.json with a full immutable HF commit, file sizes and SHA256. Legacy binary readers remain available. Authentication, privacy or checksum errors fail rather than upload bulk state back to W&B. MP4/PNG stay in Kaggle CPU outputs.

Every online job validates its main entity/project and reconciles the default Personal Workspace: no group, ID, search or selection restrictions; maxRuns=1000; running runs sorted first. Existing chart content is preserved.

## Actual completed 10k runs

| Variant | GPU job | W&B | CPU output | Episodes | Return | Discovered % | Delivered % | Success % |
|---|---|---|---|---:|---:|---:|---:|---:|
| base-masac | [complete](https://www.kaggle.com/code/fasfwejlklfv/uavnet-train-20261015-025317-f238b6) | [finished](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/b97d9f62) | [CPU MP4/PNG](https://www.kaggle.com/code/fasfwejlklfv/uavnet-train-20261015-025317-f238-cpu/output) | 10,000 | 1297.01 | 97.73 | 93.47 | 30.00 |
| base-matd3 | [complete](https://www.kaggle.com/code/nhtbrg/uavnet-train-20261015-025317-7c91b1) | [finished](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/6c1b317d) | [CPU MP4/PNG](https://www.kaggle.com/code/nhtbrg/uavnet-train-20261015-025317-7c91b1-cpu/output) | 10,000 | 335.69 | 52.93 | 27.60 | 0.00 |
| pql-masac | [complete](https://www.kaggle.com/code/nyegerge/uavnet-train-20261015-024259-32b9d7) | [finished](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bcbd5a81) | [CPU MP4/PNG](https://www.kaggle.com/code/nyegerge/uavnet-train-20261015-024259-32b9d7-cpu/output) | 10,000 | 948.94 | 94.27 | 78.53 | 16.67 |
| pql-matd3 | [complete](https://www.kaggle.com/code/ergewh54/uavnet-train-20261015-024301-df18fe) | [finished](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e0c7b655) | [CPU MP4/PNG](https://www.kaggle.com/code/ergewh54/uavnet-train-20261015-024301-df18fe-cpu/output) | 10,000 | 303.14 | 42.93 | 23.20 | 0.00 |

All four downloaded CPU hardware reports have cuda_available=false and no CUDA devices. Each output contains final_episode.mp4, final_episode.png, metrics.json and 30 evaluation records. The original pinned GPU and CPU job sources were not restarted or modified.

## What the comparison supports

PQL did not improve final policy quality in this single-seed run. Separate learning quality from throughput. Both variants use the same experiment settings and a 10,000-episode target; MASAC interaction counts differ because batched environments finish different trajectories. Production BASE exposes one GPU to its learner; production PQL exposes two. Neither the episode target nor these production wall times establish equal-GPU efficiency.

Paired bootstrap intervals below describe the 30 evaluation scenarios conditional on these checkpoints, not uncertainty across independently trained seeds. No overall winner is inferred.

| Algorithm | Metric | BASE minus PQL | Paired 95% CI | Equal training joint-step budget |
|---|---|---:|---|---|
| masac | return | 348.07 | [238.44, 454.69] | False |
| masac | gcs_delivery_fraction (percentage points) | 14.93 | [7.87, 22.00] | False |
| masac | mission_success (percentage points) | 13.33 | [-6.67, 33.33] | False |
| matd3 | return | 32.55 | [-11.78, 79.21] | True |
| matd3 | gcs_delivery_fraction (percentage points) | 4.40 | [1.47, 7.47] | True |
| matd3 | mission_success (percentage points) | 0.00 | [0.00, 0.00] | True |

Matched GPU microbenchmark: 4,096 environments, batch 8,192, capacity 500,000, 40 collection iterations, 72 completed critic updates; startup excluded, learner drain included. It is a short throughput measurement, not a convergence test. Torch allocator peaks below exclude Warp allocations; physical-board telemetry is retained in the benchmark output.

| Algorithm | Mode | Joint transitions/s | Torch peak GiB per visible GPU |
|---|---|---:|---|
| masac | BASE | 3427.23 | 7.407, 0.000 |
| masac | PQL single GPU | 3282.87 | 8.572, 0.000 |
| masac | PQL dual GPU | 3868.83 | 0.204, 8.761 |
| matd3 | BASE | 7289.89 | 5.245, 0.000 |
| matd3 | PQL single GPU | 7444.37 | 5.749, 0.000 |
| matd3 | PQL dual GPU | 9172.56 | 0.204, 5.656 |

The GPU benchmark completed successfully: https://www.kaggle.com/code/trteht/uavnet-benchmark-20261015-023236-779b9a/output

## Verification and safe cleanup

- Full unittest discovery: 362 tests, 315 passed and 47 CUDA-only skips; no failures (162.332 seconds).
- Actual first maintenance attempt caught W&B nested-summary flattening and retained all originals. Scalar-per-version mappings now pass real server readback and a simulated missing-original lookup; the resolver also accepts older flattened/nested mappings.
- Both review findings fixed: queued/unregistered children require a bound completion proof; W&B producer reads explicitly flush the SDK cache.
- Archive verifies the exact producer, actual GPU/root/CPU refs and owner, completed episode budget, complete GPU/CPU status and actual successful CPU output referencing the checkpoint.
- Each selected version receives a private immutable backup, remote size/hash check, forced independent fresh download, replacement-manifest commit, alias preservation and mapping-persistence readback before original deletion. Active or unproven sources are retained.
- A real Kaggle preflight exposed transient DownloadDataset 404 after private metadata became visible. Credential readiness retries are bounded and only retry 404; authentication, ownership, privacy, file-count and token mismatches still fail.
- Actual local online full-state restore: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3044e6dd; full checkpoint version 3 restored.
- Separate CPU-only Kaggle HF restore smoke completed: https://www.kaggle.com/code/fasfwejlklfv/uav-storage-live-parent-cpu/output; W&B https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/dca6dc93.

Cleanup scope is fixed to 187 versions in six proven completed collections: b97d9f62, 6c1b317d, bcbd5a81, e0c7b655, 501d6d1a and ea73429e. Planned migration is 37,418,273,432 logical bytes; projected remaining artifact total is 4,701,173,324 bytes plus small new manifests. This projection is not a claim that cleanup has finished.

| Account | Versions | Logical bytes selected | CPU archive | W&B archive |
|---|---:|---:|---|---|
| account_08 | 79 | 7,997,218,350 | prepared | not submitted |
| account_09 | 79 | 8,146,780,845 | prepared | not submitted |
| account_10 | 24 | 6,101,266,030 | prepared | not submitted |
| account_11 | 3 | 5,140,595,885 | prepared | not submitted |
| account_02 | 2 | 10,032,412,322 | prepared | not submitted |

Only the selected account token, one W&B key and one HF token are attached in separate owned private datasets. No full key.txt or other accounts tokens enter a maintenance session. Maintenance metadata explicitly disables the GPU.

## Current verified Kaggle accounts

20 enabled accounts, checked 2026-10-15T07:07:54.900012+00:00. Remaining hours are a snapshot.

| Account | Username | Remaining GPU hours |
|---|---|---:|
| account_01 | duytrngoanh | 20.94 |
| account_02 | haibro1234 | 21.14 |
| account_03 | vuliu123456 | 19.83 |
| account_04 | saoling1234 | 15.28 |
| account_05 | riujick | 20.2 |
| account_06 | libiu123 | 20.91 |
| account_07 | bduy2378 | 3.51 |
| account_08 | fasfwejlklfv | 25.88 |
| account_09 | nhtbrg | 28.48 |
| account_10 | nyegerge | 26.13 |
| account_11 | ergewh54 | 29.28 |
| account_12 | trteht | 29.89 |
| account_13 | uhgiwwiurfw | 30.0 |
| account_14 | fiuhgweh | 30.0 |
| account_15 | hgyuguguhi | 30.0 |
| account_16 | trhrthrtnrth | 30.0 |
| account_17 | sfkjewfljwejf | 30.0 |
| account_18 | ouigyuftrdfyghu | 30.0 |
| account_19 | fwijevwe | 30.0 |
| account_20 | iofiewfjw | 30.0 |
