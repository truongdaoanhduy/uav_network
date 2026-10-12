# Kaggle and W&B execution evidence

Date:2026-10-12. Source repository:truongdaoanhduy/uav_network.
The old project is used only as the source of the explicitly requested local
credential copy; this project does not import its implementation.

## Local credentials and verification

Copied `/home/aduy/Documents/NCKH/uav_search_target/key.txt` in full to
`/home/aduy/Documents/NCKH/uav_network/key.txt`:1461 identical bytes, mode0600.
`git check-ignore key.txt` passes. All enabled Kaggle accounts authenticated for
quota; account_02 had30 h remaining and resolves tohaibro1234. The selected
private dataset is owned by that account and its W&B key matches the local copy.
No credential value is embedded in source, generated kernels, job manifests or
W&B config. The full local key and unrelated HF fields are not uploaded.

Local complete suite:69 tests,66 pass and3 explicit CUDA skips (local CPU host).
Independent review's tracker-initialization and process-cleanup issues were
reproduced, fixed and covered by regression tests. A real two-worker subprocess
test confirms graceful SIGTERM runs tracker/env cleanup and removes both workers.
Canonical Kaggle URLs are taken from the push response, with regression coverage
for title-derived slugs and CLI exit0 responses containing server errors.
Largest owned source/document file remains below500 lines.

Real local online run:
https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e5c4d730 .
2 episodes,8 joint transitions,1 learner update; API verification found finished
state,4 history rows,progress before episode completion and1 model artifact.
This short run verifies execution/tracking, not convergence.

## Submitted CUDA jobs

Validation source:8e3953116942d627ed3d3ef0a67f312ec90948f1.
https://www.kaggle.com/code/haibro1234/uav-network-validate-20261012-115047-be8d12 .
Metadata read back from Kaggle verifiesprivate=true,gpu=true,internet=true and
the expected credential dataset. At submission the job queued for capacity.

Training source:79b72c1a433cdb259c48796950367538f5e3a58d.
https://www.kaggle.com/code/haibro1234/uavnet-train-20261012-120134-18ad3b .
Private T4 job,12-hour limit. Default1000 episodes per algorithm,2 CPU envs per
learner,6 UAVs/scenario,10 targets,8 obstacles,800 max steps,seed44,batch512,
CUDA replay100000 and1 update/collection. A two-GPU allocation runs MASAC and
MATD3 independently on device0/device1; a one-GPU allocation runs them sequentially.

The training entry point refuses unavailable CUDA, runs all3 CUDA regressions,
then benchmarks both algorithms with the actual model/batch/env configuration
and synchronized GPU timing. Learners start only after all checks pass.
Output keeps hardware.json,cuda-tests.log,algorithm benchmark JSON/logs,
per-algorithm episode logs/config/checkpoints/summary and CPU evaluation results.

W&B run URLs:

- MASAC:https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/4189eb83 .
- MATD3:https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/65a7b4be .

This first training job transitioned running→error before learner startup:
Kaggle Python3.13 could not bootstrap venv ensurepip. No W&B runs were created.
System-package setup now uses `venv --without-pip --system-site-packages`, with
the image's installed pip. Local fresh-venv verification passed. The corrected
bootstrap is being checked remotely; no GPU-training success is claimed yet.

## Local full-config benchmark

CPU baseline using the exact Kaggle model/batch/scenario config,2 envs,512 samples
and20 measured updates after warmup,network enabled:

| Algorithm | Collector joint transitions/s | Learner updates/s |
| --- | --- | --- |
| MASAC | 129.88 | 1.61 |
| MATD3 | 124.00 | 8.26 |

Replay allocates642400000 bytes per learner. Collection and warmed learner
timings are separate; these numbers are not end-to-end training throughput or
a claim about Kaggle GPU speed. Raw reports remain in ignoredoutputs/kaggle/.
