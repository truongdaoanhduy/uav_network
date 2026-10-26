# Corrected four-way production launch

Observed 2026-10-26T09:02:48.190713+00:00; independently validated 2026-10-26T09:04:03.422566+00:00.

User authorized four Kaggle accounts for MASAC/MATD3 BASE/PQL. All four GPU kernels and W&B runs are RUNNING; this is launch/progress evidence, not completed training. All four have increasing global_step beyond warmup. No zero-bug guarantee is made.

| Case | Account | W&B run | Global step observations | Kaggle GPU kernel | Controller W&B |
| --- | --- | --- | --- | --- | --- |
| masac-base | account_14 | `d17a279d` | 233472 -> 520192 | `weofjwelf/uavnet-train-20261026-084830-d5be04` | `e9b9df73` |
| matd3-base | account_16 | `da65a1ba` | 471040 -> 1544192 | `mjhres/uavnet-train-20261026-084904-c37220` | `8ac3d295` |
| masac-pql | account_15 | `60fc6245` | 180224 -> 569344 | `nkjkhkhk/uavnet-train-20261026-084907-cd531f` | `db402158` |
| matd3-pql | account_17 | `6f538a33` | 163840 -> 954368 | `njbiugyry/uavnet-train-20261026-084909-08ae22` | `769adc59` |

## Protocol

- Source pinned `f39f45682b0b8a387cf8d95906c6730bf4337742`, clean detached worktree `/home/aduy/Documents/NCKH/uav_network-cohort-g2-20261026`. Main checkout's existing untracked wandb directory was preserved.
- Group `uav-g2-o3-full20k-e4096-b8192-r1kb-cpu75-20261026`.
- Primary config `configs/comparison-20k-hybrid75-1kb.yaml`, SHA256 `503cc6fb34133b7ffb0a43e3c9238b07a5e587958a49d68c45d3f491f711e952`.
- Four distinct accounts had 30 GPU hours each and no existing owned kernels before launch. Fresh W&B inventory had no running training or duplicate current cohort. HF read/write scope verified before launch.
- From episode zero, no warm-start artifacts; observation_version=3 and generation_version=2 confirmed in actual run config.
- Each actual host reports two Tesla T4, Torch2.11.0+cu128. 20,000 episodes, 4096 environments, batch8192, replay1M/CPU75%, report1KiB, horizon1800, trainseed44. AMP/compile/fused Adam off; scientific scenario/reward unchanged.
- Auto-continuation enabled. Final CPU evaluation configured for 30 episodes, seed2000000044. HF checkpoint store `duy95/uav-search-checkpoints`; W&B retains metrics and manifest pointers.

## Controller verification and limits

Four independently running Kaggle CPU controllers have matching root/source/current GPU refs and phase training. Heartbeats meet the existing <=1800s readiness predicate; normal poll period is900s. They run on Kaggle independently of the local launch process. This proves current supervisor readiness, not guaranteed future quota/platform availability or final completion.

PQL launch receipts reported unattended_ready=true. Both BASE launch receipts reported false/error_type=ValueError during the initial readiness wait; preserve those receipts. Subsequent fresh independent observations pass the same readiness conditions for all four controllers. No parent was resubmitted, cancelled, or restarted. No claim is made that the initial readiness exception root cause has been resolved; recheck if it recurs.

Adjacent `2026-10-26-corrected-four-way-status.json` contains actual hardware/config/status/controller observations and independent readiness/progress validation. Full launch receipts and preflight are in `outputs/cohort-g2-20261026/`, with each parent/controller manifest in the clean worktree outputs/kaggle directory. No secret values are included here.

## Prior verification

Latest source-equivalent CPU suite:778passed89skipped. Real2T4 full suite:846passed21skipped, plus12sparse-checkpoint regressions and two full4096PQLresume checks. See `2026-10-26-pql-sparse-checkpoint-fix.md`. Passing tests do not prove absence of every defect. Battery/sensor assumptions, reactive actors, ideal local beacons and single trainseed remain scientific limitations.

## Next action

Fresh-query these exact four lineages and controller heartbeats; inspect checkpoints/HF immutable manifests, continuation lineage and final episode-budget proof. Do not launch duplicates or warm-start historical policies. Final30-episode CPU evaluation is pending training completion. HF/W&B cleanup/deletion remains prohibited; normal training uploads are authorized. No source edits in this launch task.
