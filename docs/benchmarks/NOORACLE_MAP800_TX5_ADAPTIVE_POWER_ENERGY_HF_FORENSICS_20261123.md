# MAP800 MATD3 NoOracle — adaptive TX up to 5W, flight/communication energy, deep HF forensics

**Date:** 2026-11-23. Parent source commit `8ed4fcfad9a723397de2d9f6a132f1f80e0f613f`. New isolated branch `experiment/map800-tx5-learned-hf-deep-forensics-20261123`.

## Intent and scientific model

Keep unchanged the audited MAP800×800m, belief40×40 at20m, 6UAV/35 targets/25 obstacles, 0–150m 3D altitude and adaptive Bayesian confirmation (`yes`, posterior>=0.99, positive LR>=4), actor/critic H256×256 orthogonal/Inductor, E4096, U4/B8192, full20,000 episodes specified in YAML but **not launched**.

Policy action remains **3D motion + a real continuous RF TX action** in [-1,1] mapped logarithmically to P in [0.001,5.0]W, **plus discrete** silence, each of five peer UAVs, or direct GCS. Policy is **not** forced to 5W. Different power choices influence actual PHY reception SINR, path loss, interference with concurrent transmissions, retries, ACK/custody and airtime, communication energy subtracted from the *same battery* that powers motion. Selecting peer, GCS or silence is a separate decision. The simulated source still enforces nominal peer/GCS range 500m and 10s packet lifetime, 300s report TTL, 2Mbps datarate. TX power does **not** remove the hard range gate or guarantee successful delivery.

Ground truth (hidden target XY) is restricted to stochastic sensor generation and private scoring/forensic log, not Actor actions or radio report admission. Actual on-air REPORT is a 2D observed-cell index. Real-world sensor/packet/loss calibration and transmit-hardware power limits are not provided by this simulator.

### Why Pmax=5W as one controlled experiment, not a universal optimum

Existing power max0.4W: nominal LOS at500m ~0.00442W, NLOS100m ~0.2803W; NLOS250m ~4.3797W, NLOS500m ~35.04W under source code's urban2.4GHz/22MHz/SINR6dB PHY model **without concurrent interference**. Therefore TX5W newly permits some shorter NLOS direct hops but keeps distant NLOS communication dependent on relay/moving. Doubling to0.8W would only widen the ~100–140m obstructed regime and not support250m NLOS. However 5W can increase interference, hidden-node/CCA contention and retries: it may be *worse* than a carefully selected low-power peer relay or delayed send, so learning has a meaningful choice.

### Physical battery and reward caution

Current PHY subtracts `E_TX = ((1024*8+464)/2e6) * (0.1 + P_TX/0.35)` joules per DATA attempt. Typical one-hop values: P=.4W=>~.00538J; P=.8W=>~.01033J; P=5W=>~.0623J, plus ACK, RX and retries. Flight draws about140J/s when hovering: increasing TX alone will *not* create strong physical flight-endurance pressure, even though actual RF Joules are correctly deducted from the shared battery. Never claim realistic flight-time improvements if not measured.

The pre-existing reward also has a normalized RF efficiency penalty `c8=-communication_efficiency * (new_Rf_J) / report_reference_tx_energy_j`. **Unfixed Pmax-relative normalization would DIVIDE this penalty by ~11.6 when raising Pmax to5W**, making identical energy artificially appear cheaper to the optimizer. This experiment adds an opt-in `reward.communication_reference_tx_power_w=0.4`: reference report energy is pinned at the 0.4W configuration even when Actor Pmax=5W. At 5W one attempt incurs roughly -11.6 shaping reward units versus -1 at0.4W, before other terms; +50 for true GCS delivery remains. It changes scientific reward and requires fresh training/ablation (not simply hardware change). The legacy/default reference value0 preserves old behavior for historical runs.

The scalar `reward.communication_energy=0.01` physical-Joule penalty is unchanged, while `reward.communication_efficiency=1` scales the above fixed-reference dimensionless RF penalty. Flight energy penalty uses the original Joule→kJ formula and is independent of this reference.

## Source/code changes

- New science config: `configs/matd3-nooracle-lr4-map800-h256x256-orthogonal-compile-tx5-learned-hf-20261123.yaml`. Baseline config from commit8ed4fcf remains unchanged (TX max0.4W). Optional earlier0.8W ablation is also preserved.
- `uav_network/configuration.py`: new reward `communication_reference_tx_power_w` default0; validates >=0.
- `uav_network/communication/link_budget.py`: optional stable reference report power; same physical TX model. `uav_network/environment/gpu/state.py` and `uav_network/environment/env.py` consume this opt-in reference. Old configs unaffected.
- `uav_network/communication/gpu/diagnostics.py` high-power histogram: `[<.01, .01–.1, .1–.5, .5–1, 1–3, >=3]W`. For low-power historical configs retains old `[<.1, .1–.2, .2–.3, .3–.4, .4–.5, >=.5]W`. `uav_network/communication/diagnostics.py` emits corresponding bin edge metadata and `uav_network/communication/gpu/metrics.py` passes actual Pmax.
- `uav_network/experiments/forensic_transitions.py`: selected-world pre/post adds exact `tx_energy`, `rx_energy`, and full `pkt_power`, `tx_power`, packet cell, receiver, attempted destination, attempt records, before/after belief; already existing action continuous XYZ/RF, route index, Actor logits/pretanh and exact obs/critic state.
- Safe DRY-RUN production entrypoint: `scripts/launch_nooracle_map800_tx5_training_with_hf_forensics_20261123.py` with optional explicit `--execute`. Rejects CPU fallback, missing second GPU, and output overwrite. Does **not** itself run full GPU Validation/Test; the full `one_session.py` handoff remains separate.

## HF/W&B deep evidence collection

**Intended public dataset:** `https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs/<wandb_run_id>` (only after the user runs the full tracked GPU job and successfully supplies valid W&B/HF credentials). **No actual remote upload was performed for this new TX5 experiment during local verification.**

The launch profile sets:

- `UAV_DIAGNOSTIC_STORE=huggingface_public` for immutable incremental archives every ~300s, with byte-range SHA256 manifest + readback verification; upload errors never fabricate success.
- `UAV_ROOT_CAUSE_AUDIT=true` so first16 learner updates, Actor gradient/tanh collapse, actual replay minibatches, Q discrepancies and sampled transitions are captured. **Without this flag the previous integration smoke revealed `forensic-minibatches.jsonl` could remain empty.**
- `UAV_FORENSIC_ENABLED=true`, `UAV_FORENSIC_WORLD_IDS=0,1`, stride16 vector steps, full belief true, raw cap512MiB per selected trace. The first16 vector steps around warmup are captured densely. This records sampled worlds and summary counters, NOT every transition from all4096 environments. Dense logs have a storage/performance cost.
- `UAV_WANDB_PLOTS_ONLY=false` plus `UAV_WANDB_COMPACT_LOGS=true`, `UAV_WANDB_EPISODE_LOG_EVERY=32`: RF power histograms and key names visible as live W&B scalars at decimated episode cadence, whilst raw detail remains public HF. Original plot-only would filter these power keys; a real end-to-end unit test reproduced and fixed it.
- Additional per-episode radio diagnostics include `data_tx_power_w` min/mean/max, peer/GCS split, SNR/SINR/RSSI, TX margin to required power, weak signal vs interference/collision, retransmissions, ACK, lost/expired reports, queue, relay-hops, per-UAV flight/RF battery, and reward components including `communication_efficiency`.
- Journals archived under `<wandb-id>/diagnostic-archives/` include `episodes.jsonl`, `training-step-reward.jsonl`, `training-step-optimizer.jsonl`, `training-step-progress.jsonl`, `forensic-transitions.jsonl`, `forensic-minibatches.jsonl`, `run-provenance.jsonl`, `root-cause-timeline.jsonl` and failure/diagnostic events. Only actual completed lines and immutable archive members are uploaded. Do not expose API credentials through notebook/source bundle.
- Test executes **real CPU Warp+compiled MATD3** optimizer with **FAKE W&B and HF network transports only**, verifies W&B TX histogram fields, real selected-world `tx_energy`/`pkt_power` snapshots, non-empty minibatches, reconstructed archive manifest and all expected sources. This proves local source code's logging workflow, not public Hugging Face availability in the future.

## Scientific run matrix and measurements

- Control A: MAP800 Pmax0.4W baseline (existing 8ed4fcf config).
- Control B: MAP800 Pmax0.8W ablation (existing 8ed4fcf config; old Pmax-based reward reference means *not* strictly matched shaped reward to5W variant unless using equivalent fixed ref).
- Candidate C: MAP800 Pmax5W learned TX and RF reward fixed at0.4W reference (new experiment, fresh MATD3 train). For a strictly causal Pmax ablation, also add fixed0.4W reference to A and B via explicit config overrides (new experiment IDs), rather than comparing historical training scores as if normalized reward were matched.
- Train from scratch for each, multiple random seeds (44 plus additional independent seeds), equal environment transitions, fixed Val100/checkpoint, independent held-out Test1000, identical architecture/science except manipulated Pmax, report PHY PDR/NLOS vs LOS, direct vs relayed GCS deliveries, Tx power distribution, per-UAV RF energy & flight energy, battery/dead rate, false cell confirmations, RL reward curves and AUC.

## Tests and remaining limitations

TDD red tests caught un-fixed TX reference, collapsed >=0.5W histogram, missing HF-root-cause-flag and W&B plot filter silently stripping power telemetry. Confirmed green tests include actual Warp histogram emission .005/.02/.2/.75/1.5/4.5W; real compiled MATD3 CPU smoke with HF/W&B fake network transports; deep scientific archive integrity and independent inherited energy, network and training regressions.

**Additional real Warp CPU full-horizon stress:** 16 seeded worlds ×1500 steps were exercised with this TX5 config, max altitude150m, max allocated report cells51/140, no internal Warp errors, 502/560 targets observed and399/560 GCS delivered. These actions are hand-crafted, not a trained MATD3 policy; outcomes matched the TX0.4 stress case and do **not** establish any improvement from power5W. Per-world JSON is retained in local Downloads `MAP800_TX5_LEARNED_16WORLD_FULL_HORIZON_STRESS_20261123.json`.

**Targeted regression:** `tests/test_map800_tx5_energy_tradeoff_hf_20261123.py` (6 passes), `tests/test_map800_tx5_hf_launch_profile_20261123.py` (2 passes), `tests/test_map800_tx5_real_wandb_hf_archive_20261123.py` (1 passed, real compiled CPU MATD3 and mocked network W&B/HF), 98 related passes +14 subtests, and 90 broader Train/Eval/Resume/NoOracle passes: **188 passing tests +14 subtests, no failures**, with seven warnings in the related batch. These are selected tests, not full repository coverage.

**NOT DONE:** user Kaggle T4×2 E4096 training or real HF push, 20k episodes, full100-episode-per-checkpoint GPU validation, best-Actor selection, held-out Final Test1000, simulation wall-clock/VRAM throughput or learned policy proof. No physical hardware/radio regulatory verification. TX5W simulation capability does not authorize transmitting at5W on actual 2.4GHz hardware.

## Usage

On project worktree:

```bash
# Prints plan, no network write or training:
python -m scripts.launch_nooracle_map800_tx5_training_with_hf_forensics_20261123

# Run only on an intentionally configured TWO-CUDA-GPU Kaggle host:
# Must configure valid WANDB_API_KEY and HF_TOKEN/credentials beforehand.
python -m scripts.launch_nooracle_map800_tx5_training_with_hf_forensics_20261123 --execute
```

Do not resume old MAP1000/MAP800 actor checkpoint because reward/power distribution changed. A full Train→Eval→Test production notebook still requires wiring the separate 2-GPU one_session supervisor, and should not silently claim completion from a successful TRAIN-only launcher.
