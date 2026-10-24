# 1 KiB MASAC live state and duplicate cleanup — 2026-10-24

Fresh post-MATD3-finalization snapshot.

## Canonical MASAC lineages

### MASAC BASE

- W&B: `7f985331`
- Kaggle: `duytrngoanh/uavnet-train-20261024-073119-a53a63`
- Controller: `duytrngoanh/uav-flow-22c9acce21eef2-0000`
- W&B state: running
- Kaggle root: running
- Controller: running
- global_step: 12,181,504
- report_bytes: 1024
- num_envs: 4096
- target episodes: 20,000
- no completion gate yet

### MASAC PQL

- W&B: `e0bfbcaf`
- Kaggle: `vuliu123456/uavnet-train-20261024-074226-5b2381`
- Controller: `vuliu123456/uav-flow-7dbb76c688e6f5-0000`
- W&B state: running
- Kaggle root: running
- Controller: running
- global_step: 13,352,960
- report_bytes: 1024
- num_envs: 4096
- target episodes: 20,000
- no completion gate yet

Both current owners have only their canonical root + controller active for this cohort. Preserve them unchanged until the completion gate is reached.

## key.txt alias remap

The new Sandbox's account aliases differ from earlier handoff labels. Resolve owner from the authenticated token before any mutation. Current mappings used here:

- `duytrngoanh` -> `account_01`
- `vuliu123456` -> `account_02`
- `libiu123` -> `account_03`
- `nhtbrg` -> `account_04`

Do not assume historical `account_xx` labels still map to the same Kaggle owner.

## Redundant MATD3 PQL root removed

A second independent MATD3 PQL root was still consuming GPU:

- W&B `bbfb7e35`
- Kaggle `nhtbrg/uavnet-train-20261024-085710-ecbd3f`
- controller `nhtbrg/uav-flow-1941190da2423f-0000`

It matched the completed canonical MATD3 PQL experiment on group, seed44, env4096, 20k target, 1 KiB report, batch8192, replay1M and PQL. The only production-code difference between canonical source `0c9fbc5` and duplicate source `fbbb556` was `torch.cuda.init()` in `uav_network/cloud/execution_check.py` (cloud preflight only); no algorithm/scenario/training-path change.

The canonical MATD3 PQL `30a40bf7` had already completed 20k and passed CPU/W&B/HF finalization. The redundant controller was deleted first, then the redundant GPU root. Fresh status confirms both redundant Kaggle refs are gone while the canonical CPU result remains complete.
