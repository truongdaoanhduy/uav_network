# Accelerated cohort reaudit + MASAC terminal-state diagnosis

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Fresh read-only operational re-audit after the previous readiness blocker. No Kaggle job was launched, resumed, cancelled, or mutated. No W&B/Hugging Face data was modified. Existing untracked `wandb/` was preserved.

## Fresh Kaggle account gate

All 24 enabled accounts in `/home/aduy/Documents/NCKH/uav_network/key.txt` were checked independently so one invalid token could not abort the full scan. No credential value was printed.

Production readiness requires all four gates: current authentication, >=12 verified GPU-hours, zero active kernels, and read access to the account-owned canonical private `uav-cpu-wandb-credential` dataset.

Only two accounts pass now:

- `account_01`: auth OK, ~23.42 GPU-hours, zero active kernels, credential dataset accessible.
- `account_02`: auth OK, ~25.03 GPU-hours, zero active kernels, credential dataset accessible.

`account_03` through `account_24` currently fail authentication. The local file contains 24 enabled unique tokens, no disabled account and no duplicate token group, so there is no unused local credential that can satisfy the missing two production slots.

The file itself remains mode 0600. Public Kaggle documentation confirms that API access requires current credentials/tokens and that invalid/revoked/expired tokens cannot authenticate; quota metadata alone is therefore not a sufficient launch gate.

Decision: still **NOT READY** for a four-account fresh accelerated 20k cohort. Do not submit a two-account partial cohort.

## Fresh W&B lineage state

The previous preflight recorded four robust G3 MASAC lineages as running. A fresh W&B API read now reports all four as `crashed`:

| Run | Mode | Final observed global step | Episodes | W&B state |
| --- | --- | ---: | ---: | --- |
| `25eecfd1` | MASAC BASE fullGPU | 14,651,392 | 8,192 | crashed |
| `63202ad9` | MASAC BASE hybrid75 | 16,424,960 | 8,192 | crashed |
| `3b39b0c8` | MASAC PQL fullGPU | 16,232,448 | 8,196 | crashed |
| `6f0bcde3` | MASAC PQL hybrid75 | 13,889,536 | 7,045 | crashed |

The four MATD3 G3 runs remain `finished` at 20,000 episodes.

The four MASAC controller runs are also `crashed`:

- fullGPU BASE controller `45ff19d1`
- hybrid BASE controller `8e3786a8`
- fullGPU PQL controller `16f5a5f3`
- hybrid PQL controller `4f61b7cd`

## Crash timing evidence

W&B heartbeat timestamps:

- training `25eecfd1`: 2026-10-26T19:31:57Z
- training `63202ad9`: 2026-10-26T19:32:06Z
- training `3b39b0c8`: 2026-10-26T19:32:08Z
- training `6f0bcde3`: 2026-10-26T19:31:57Z
- controller `45ff19d1`: 2026-10-26T19:32:02Z
- controller `8e3786a8`: 2026-10-26T19:32:01Z
- controller `16f5a5f3`: 2026-10-26T19:32:02Z
- controller `4f61b7cd`: 2026-10-26T19:32:07Z

All eight independent W&B processes stopped heartbeating inside an 11-second window. Their summaries contain normal finite training metrics/checkpoint metadata up to the last updates and do not expose a run-specific Python exception/traceback.

This strongly argues against four independent MASAC numerical/code failures occurring coincidentally at the same instant. The evidence is consistent with a shared external termination domain such as Kaggle/platform/session/network/credential infrastructure. Because the old owner-account tokens no longer authenticate, the exact Kaggle-side termination reason cannot be retrieved safely from those private kernels now. Therefore the exact external mechanism is **unproven** and must not be mislabeled as an algorithm bug.

A valid account cannot query those private kernel statuses because access is denied to another owner's private notebook, so W&B is currently the authoritative observable for these historical lineages.

## Implications for the accelerated cohort

The measured acceleration policy remains unchanged:

- MASAC BASE: FP16 + compile ON
- MATD3 BASE: FP16 + compile ON
- MASAC PQL: FP16 + compile ON
- MATD3 PQL: FP16 + compile OFF
- fused Adam OFF

No new code/CUDA/scenario/replay defect was found in this re-audit. The blocker remains account authentication capacity. Before a fresh four-way launch, replace/refresh at least two invalid account credentials locally and rerun all four account gates. New runs must start at episode 0; do not exact-resume historical off/off checkpoints into the accelerated policy.
