# Primary 1 KiB MATD3 20k finalization — 2026-10-24

Fresh verification after the cloud E2E gate passed.

## MATD3 BASE

- Training W&B: `767cb8da`
- Training kernel: `libiu123/uavnet-train-20261024-074821-e57c46`
- CPU visualization kernel: `libiu123/uavnet-train-20261024-074821-e57c46-cpu`
- episodes_completed: 20000
- episode_budget_complete: true
- stop_reason: episode_budget
- W&B training state: finished
- CPU evaluation episodes: 30
- CPU device: cpu
- production `_cpu_complete()`: true
- W&B publication: ok
- Hugging Face publication: ok
- HF immutable revision: `24dccbd57963e7eec1c434cf3de92f6ad4751ac3`
- video/image/metrics: all have immutable URLs and sha256

## MATD3 PQL

- Training W&B: `30a40bf7`
- Training kernel: `nhtbrg/uavnet-train-20261024-080451-eda89d`
- CPU visualization kernel: `nhtbrg/uavnet-train-20261024-080451-eda89d-cpu`
- episodes_completed: 20000
- episode_budget_complete: true
- stop_reason: episode_budget
- W&B training state: finished
- CPU evaluation episodes: 30
- CPU device: cpu
- production `_cpu_complete()`: true
- W&B publication: ok
- Hugging Face publication: ok
- HF immutable revision: `fee4f14953727d0be0d1da1c3c2c86d2c840da60`
- video/image/metrics: all have immutable URLs and sha256

No manual validation/visualization submission was needed; the existing cloud controllers advanced both completed lineages automatically.
