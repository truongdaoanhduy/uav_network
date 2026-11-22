# No-Oracle MATD3 `504d1b89`: Kaggle account 1 + Hugging Face + Warp root-cause audit

**Date:** 2026-11-22. **Mode:** read-only remote inspection; verified local source/test fixes performed on an isolated branch. The live Kaggle process was not stopped, edited or restarted.

## Exact provenance

- Kaggle account alias `account_1`, account owner `longlonthiu`, kernel `longlonthiu/notebooke52962646d`; Kaggle CLI returned `KernelWorkerStatus.RUNNING` during the investigation.
- W&B newer run `truongdaoanhduy-uit/uav_search_target/504d1b89`, `source_git_commit=fbba543010557deeb665b4c95081fcbfafcbfdf5`, Bayesian no-oracle per-cell reporting, E4096/U4/B8192/seed44, six UAV and 35 targets.
- W&B historical run `18d756ef`, original frozen source `/home/aduy/Downloads/M2_EXACT_REFERENCE_18d756ef_20261121/verified_source/`, MATD3 model code identical by SHA256; the reporting rules and network pre-admission rules differ.
- Public historical HF archive: `duy95/uav-search-diagnostics`, `runs/18d756ef/diagnostic-archives/` (169 `.tar.gz` parts). First 4,096 historical `episodes.jsonl` records were parsed directly and aggregated from 25 archive parts for matched-population comparison.
- New run HF: **zero matching files at `runs/504d1b89/`** when enumerated. Its actual current telemetry was recovered from W&B Files: `training-status.json`, `cohorts.jsonl`, `thread-stacks.log`, `resolved_config.yaml`, `metrics-guide.md`.
- Local diagnostic comparisons and CPU Warp microprofiles are under `/tmp/uav_cpu_profile_old_18d_1600.log`, `/tmp/uav_cpu_profile_new_504d_1600.log`, `/tmp/kaggle_account1_504d1b89_audit/` and `/tmp/uav_nooracle_bitwise_before_sparse.log` on the connected Ubuntu machine.

## Matched first-4096 completed-episodes cohort

| Metric | Historical `18d756ef` | No-Oracle `504d1b89` | Meaning |
|---|---:|---:|---|
| Found targets (%) | 26.3825 | 32.6800 | No-Oracle found more truth-matched target *cells*, not an identical task distribution |
| GCS delivery (%) | 24.9330 | 24.9386 | Essentially unchanged |
| Generated packets | 104,330 | 191,391 | +83.4% packets admitted to simulated network |
| Received packets | 104,083 | 97,236 | Fewer delivered packets |
| Packet PDR | 99.7633% | 50.8049% | Packet metric **not** target delivery |
| DATA attempts | 106,590 | 570,800 | 5.35× more attempts |
| DATA retries | 2,260 | 379,409 | 167.9× more retries |
| DATA PHY failures | 2,254 | 473,211 | 209.9× more failed transmission attempts |
| Below-required-power attempts | 615 | 468,365 | 761.6× more non-feasible attempts |
| Retry-limit failures | 247 | 94,179 | Failed links continue to consume MAC attempts |
| Peer PHY failed decodes | 1,161 | 290,919 | 61.48% of the new PHY failures |
| GCS PHY failed decodes | 1,093 | 182,292 | 38.52% of the new PHY failures |
| CCA checks (logical counter) | 42,353,387 | 995,152,051 | 23.5×; **coalesced**, not 995M individually executed kernel loops |

The new cohort's DATA `required_power_w` mean is about 56.33 W, while configured `tx_power_max_w` is 0.4 W. These values describe transmission attempts across various NLOS, distance and power conditions, not a recommended hardware power setting.

## Why the slowdown occurs

1. The 473,211 failed attempts in the new cohort split into **290,919 peer failures and 182,292 GCS failures**. This is not exclusively a peer-beacon problem: low action-selected TX power also affects attempts to GCS. In `uav_network/communication/gpu/mac.py::packet_step`, peer sends in `p.cell_reports != 0` mode skip the old omniscient remote PHY precheck: this is scientifically appropriate if there is **no peer beacon**, but it admits attempts to unreachable peers that subsequently fail and retry in the actual PHY/MAC.
2. The new simulator assigns **140 hypothesis report slots per world**, versus **35 true-target indexed slots** previously. Its session state is 5,880 entries vs 1,470 per world. WorldState E4096 is estimated at 3.88 GiB vs 2.82 GiB; report scans, event queue handling and memory traffic grow accordingly. These allocations were checked against exact run configs/source.
3. CPU Warp microprofile over 1,600 steps and 2 worlds, with 10-step warmup and the two exact run configurations, measured `packet_step`: **0.113 ms → 0.740 ms per vector step (6.55×)**, and whole-step CPU latency **1.739 ms → 1.943 ms (1.12×)**. Packet share grew from about **6.5%** to **38.1%** of CPU step wall time. CPU proportions cannot be copied onto T4: the observed GPU run was ~2.7× slower around 7M transitions, which needs actual CUDA profiling to attribute precisely.
4. The historical and new source `uav_network/algorithms/matd3.py` matched byte-for-byte. Run settings E4096/U4/B8192, CPU75%/GPU25% replay, ActorBN H256×512 and plain Critic were unchanged. Main changes are world dynamics, report identity, RF admission and observations.
5. Actual Kaggle GPU telemetry observed much higher GPU0 collector usage (~94.7% vs ~55.5%) and lower GPU1 learner utilization (~13.6% vs ~28.2%), consistent with a collector-bound pipeline. These are W&B samples, not Nsight kernel timings.

**Do not restore old omniscient per-peer checks.** This would reintroduce a hidden remote-state oracle. Any future retransmission suppression must be based on locally observed ACK/RSSI/beacons, preferably an explicit ablation, with packet/network parity and mission outcome regression tests.

## Correctness issue additionally reproduced and fixed

In the No-Oracle Actor observation, `uav_network/environment/gpu/observations.py::observe` encoded exact GCS required transmission power using `nominal_required_power()`. That calls the true, unobserved 3D obstacle/LOS simulator even though no GCS beacon or RSSI measurement is modeled. A CPU Warp test placed a new obstacle **100 m away from the UAV, outside its ~50 m sensing footprint but between the UAV and GCS**. The only changed Actor feature was **index 153, from 0 to 1**, revealing the hidden obstacle. Test `tests/test_nooracle_hidden_gcs_radio_20261122.py` FAILED before the fix and passed after exact-power feature was made unavailable in cell-report mode. The legacy mode retains its exact original feature.

This changes the Actor input semantics. **Old `504d1b89` learned weights are not scientifically interchangeable with the fixed MDP**, even though the MATD3 algorithm implementation remains byte-identical. New weights require retraining.

**Version guard:** `observation_version(config)` is bumped from **7 to 8** for the cell-report/no-oracle profile (historical version 5 unchanged). Checkpoint loading now correctly rejects v7 snapshots in a v8 environment even though the tensor dimension is still 228. TDD test `test_nooracle_observation_version_changes_when_hidden_gcs_oracle_removed` failed before the bump and passed after. Legacy `/tmp` v7 immutable model snapshots must not be silently relabeled v8; evaluation tests explicitly verify the rejection. A fresh v8 MATD3 train → Val100 smoke test is run separately to verify that *new* v8 snapshots remain evaluable.

## Root cause of absent HF logs and fixed integration

`uav_network/experiments/tracking.py` in run `504d1b89` retained `upload_compressed_journals(self)` on finalization but **did not create** either `CompressedDiagnosticArchive` or `PublicHubDiagnosticUploader`. The historic `18d756ef` tracker created both and ticked archives on periodic events. Therefore the new run did not have the configured HF forensic upload pipeline; this is not evidence of an HF outage.

The isolated audit branch restores both objects **only** when `UAV_DIAGNOSTIC_STORE=huggingface_public` and `UAV_FAST_TRAINING` is false, with a periodic 300-second/8-MiB archive cadence. New notebook cell 2 includes `ENABLE_HF_FORENSICS=True` (toggle off for W&B-only), and sets the public diagnostics repo explicitly to `duy95/uav-search-diagnostics`. Archive filenames are immutable and publisher validates public status and downloaded content hash. TDD fake-HF tests verified actual archive creation, upload through mock receiver, and final manifest; no secret values were uploaded in tests.

**It does NOT retroactively recover diagnostics that were never archived for the current run.** The running Kaggle kernel cannot be patched in place merely by updating GitHub or a downloaded notebook. Enabling HF archive may add bounded network/compression overhead; do not claim it makes training faster. Check via actual GPU run throughput before making a scientific comparison.

## Other verified caveats

- Last Kaggle status was **RUNNING**. W&B had monotonically increasing global steps and optimizer updates. No confirmed NaN, fatal learner error, stalled step progress or reported CUDA low-headroom alert from the observed data; no direct access to a live Nsight trace.
- `recovery/full_checkpoint_available=0` in live W&B, so a manually canceled run is not proven to have a resumable *full optimizer/replay/collector* checkpoint.
- No 3D obstacle reconstruction or GCS DS-map fusion is added; this is Bayesian target search → radio report → GCS only.
- Physical packet PDR and true target-delivery rate are different measures. Training returns and Validation100/Final Test4000 must not be confused.
- The strict repository style suite retains seven inherited >500-line violations; no production simulator failure is inferred from them.
- The correct next performance validation is isolated T4×2 kernel timings for `packet_step`, `sense_reports`, `confirm_observed_cells`, `observe`, plus end-to-end transitions/s and memory use, **before** changing MAC semantics.

## Test evidence

- Red/Green TDD: `tests/test_nooracle_hidden_gcs_radio_20261122.py` reproduces hidden obstacle leak, then passes with corrected Actor input.
- Red/Green TDD: `tests/test_nooracle_forensic_uploader_20261122.py` reproduced missing uploader, then passed with restored opt-in archiver and fake-HF publishing.
- Original 18d and new 504d CPU Warp kernel profiles are saved locally and used only as CPU evidence.
- `tests/test_matd3_reference_style_kaggle_ui.py` now rejects a notebook without user-visible forensic toggle / HF repo configuration; the rebuilt notebook must remain below Kaggle's 1,000,000-byte source/file limits and preserve the three original cells.
- Fresh v8 `scripts/audit_nooracle_train_validation_20261122.py` completed **3/3** independent runs with actual MATD3 optimizer: **132 optimizer updates and 100 validation cases per snapshot**, confirmed seed44 identical Actor weights SHA256 `0e498134305e433db09f81e1b44c69bd1bb1c1bac1a05a8d758f5837e62e722c`; seed45 differs. Log `/tmp/uav_v8_cpu_train_validation_smoke_20261122.log`. This CPU smoke is not a substitute for production T4×2 training.
- A 1,600-step native Warp CPU parity trace after fixes was bytewise identical to the same random-seeded trace before the fixes (SHA256 `625eab05b581d5a5281917834dd4ef2e960b056532b41d1324de14b88172c9f7`, two worlds, found38, delivered27); all packet/session states and model inputs are included in the digest. Corrected GCS LOS leakage is contingent on where an unseen obstruction appears; this randomized trace did not exercise that difference.
- The new reference-style 3-cell Kaggle notebook is labeled **No-Oracle V8**. The first user-visible cell exposes the public HF forensic toggle; unlike v7 the compiled source includes the real archiver/uploader. Because uploading large forensic archives can affect throughput, benchmarks must state whether this toggle is enabled.
- Full repository functional test command and **observed result**:

```bash
python -m pytest -q tests --ignore=tests/test_project_structure.py --disable-warnings
# 1610 passed, 105 skipped, 8 warnings, 1111 subtests passed in 404.57s; exit 0
```

## Reproduction and remaining boundaries

The full functional suite ran as a fresh pytest process with all fixes and the new observation version-8 semantics applied. A previously obsolete pytest process was terminated only on the Ubuntu workstation to avoid simultaneous test contention; the Kaggle kernel was not terminated or altered. Script syntax, pinned algorithm SHA256 digests, and `git diff --check` were revalidated. The independent strict style suite still reports **7 inherited over-length-file failures** (not runtime errors).

**No production T4×2 run and no T4 CUDA Nsight timeline were executed**, so this audit does not claim that a specific MAC optimization improves real GPU throughput or that a 20,000-episode V8 policy converges. The production MAC scheduler and unknown-peer admission were deliberately left unchanged to prevent reintroducing an omniscient remote-link oracle.
