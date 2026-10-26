# PQL sparse checkpoint storage fix — verified

Updated 2026-10-26T15:27:18.788932+07:00. User explicitly authorized fixing the previously reported checkpoint bloat. HF/W&B deletion, migration and data mutation remain prohibited. Source fix7b8a6ed plus evidence packaging457ddb3 were verified on Kaggle; no algorithm/reward/scenario/hyperparameter changes or production cohort.

## Root cause and minimal change

`MeasuredReplay.state_dict()` previously returned CUDA transition slices and CPU/CUDA `born`/`seen` slices backed by the whole replay allocation. PyTorch preserves backing storage when saving a view, so sparse checkpoints wrote unused ring capacity. In the8192-of500000-row4096-env test this added6.445GB.

The change in `uav_network/experiments/pql_replay.py:89` moves populated transition/born/seen tensors toCPU and clones only if they still reference excess storage. This avoids a second large CUDA replay allocation. Full CPU storage remains borrowed for the existing synchronous, worker-drained save. Scalar counters, keys, schema, original device_type, RNG, loader validation, learner/optimizer and world semantics are unchanged. Existing checkpoints that pass the unchanged observation/generation guards remain readable; no serialization-format conversion is needed.

## Test-first evidence

- CPU RED:2failed,2passed,8skipped. Both empty and three-row payloads grew with unused metadata capacity.
- Actual2TeslaT4RED on6e0ed9b: **6failed,6passed,0skipped**,3.30s. Every empty/sparse CPU/cuda:0/cuda:1 serialized-size case failed as expected; roundtrip cases passed. Diagnostic receipt status `expected_red`; the regression pytest itself exited1.
- Minimal fix then focusedCPU13passed/8skipped/12subtests. Independent review: no Critical/Important findings. Reviewer minor was addressed by explicit RNG bytes/all valid transition/born/seen comparisons, in addition to next-sample checks. Empty/sparse/full-wrapped states are covered with real serialization, no mocks.
- Final CPU complete suite: **778passed,89skipped,8warnings,1303subtests,exit0**,526.35s. Eight new skips require realCUDA;81existing skips remain hardware-specific.
- Actual2TeslaT4GREEN pinned`457ddb316d9111572f46486338f3d10e2cf14d9d`,Torch2.11.0+cu128/CUDA12.8: **12regression tests passed**,3.85s; **full CUDA846passed,21skipped,37warnings,2449subtests,exit0**,486.01s; **two full4096MASAC/MATD3PQLresume cases passed**,115.01s.

New tests assert serialized size is independent of unused capacity and retain exact populated data, cursor, measurement state, RNG bytes and next sampled batches. The real full-size tests restore exact environment/controller/replay state and continue4096->8192globalsteps.

## Measured checkpoint improvement

Same legacy1MB/GPU-only500000-capacity profile,8192valid transitions,full4096env warmup protocol:

| Case | Before bytes | After bytes | Reduction |
|---|---:|---:|---:|
| MASAC PQL |8,932,913,600|2,487,767,720|72.15%|
| MATD3 PQL |8,935,556,142|2,490,410,262|72.13%|

Each saves **6,445,145,880B** of unnecessary serialized storage. Existing checkpoints were not edited or deleted. Fully populated replay still stores all valid rows; these2.49GBwarmup sizes are not a promise for final trained checkpoints. Exact peak-memory or general training-throughput gains are not claimed. Primary1KiB hybrid factory/settings are unchanged; regression coverage includes its existing full-suite tests.

## Whole-suite issue caught and resolved

The first local full suite had778passing tests but one structure subfailure: the earlier checks-only pipfreeze artifact had934lines, exceeding the repo's500line text-file contract. Its exact bytes are now preserved as two467lineparts. The structure test was not weakened. An incomplete staging of the old-path removal was caught locally; the first unsubmitted launcher was stopped before any cloud push, and457ddb3contains the complete packaging fix. No running cloud job was cancelled/restarted. The final full CPU/CUDA results above are authoritative; the original failing log is retained.

## Preservation, scope and review rulings

Read-only08:20UTC service verification:W&B321runs/234finished65failed22crashed unchanged; both HF checkpoint heads,LFSobject counts and bytes unchanged. No HF/W&B upload,summary mutation or deletion was issued. Sourceactor/critic/scenario code and primary configSHA503cc6fb34133b7ffb0a43e3c9238b07a5e587958a49d68c45d3f491f711e952 stay unchanged. No20kcohort or historical-policy warm start.

Reviewer set final test outcomes,memory measurements and size measurements aside as pending evidence; root verified the actual results above and makes no unsupported peak-memory claim. Hybrid/external-publication changes were out of scope and were not introduced. Current generator/observation guards remain in force. PQL remains the project's PQL-inspired architecture; no learning-quality or bitwise asynchronous-schedule claim follows from these tests.

Reproduction: tests/test_pql_replay_serialization.py; submitted RED/GREEN kernels, launch pin, actual receipts/log/XML, size comparison, review and preservation JSON use this report's filename prefix. Fix complete. Further HF/W&B cleanup remains prohibited.
