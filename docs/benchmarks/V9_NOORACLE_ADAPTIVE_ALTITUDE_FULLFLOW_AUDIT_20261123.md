# NoOracle V9 Adaptive Altitude — scientific full-flow and performance audit

Date: 2026-11-23. Branch: `experiment/nooracle-altitude-aware-perf-audit-20261123`.

## 1. Request and scientific contract

UAVs choose their own altitude **between 0 and 150 meters**. FoV footprint increases with altitude (`r=z tan(90°/2)`), detection probability decreases and false-positive probability increases across altitude anchor array `[0,50,100,150]`:

| altitude | Pd | Pf | FoV radius |
|--:|--:|--:|--:|
| 0m | .99 | .01 | 0m (camera disabled at z<=0) |
| 50m | .90 | .10 | 50m |
| 100m | .80 | .20 | 100m |
| 150m | .70 | .30 | 150m |

Camera yields a binary `yes/no` for a visible grid cell; a Bayesian posterior is updated from sensor likelihoods with actual fractional circular-cell overlap `q`. NoOracle Actor sees only own local observations, local belief, own altitude and vertical speed; reports contain cell IDs rather than ground-truth target IDs. Ground truth is permitted for camera outcome sampling, simulated scoring and host-only evaluator, never policy next-hop admission.

## 2. New, explicitly versioned confirmation rule

**New MDP ONLY:** `sensor.confirmation_mode=likelihood_ratio`, `confirmation_min_likelihood_ratio=4.0` and `confirmation_threshold=.99`. There is **no fixed 50m confirmation ceiling**. A newly observed positive measurement is eligible if

```
P(Y=1|empty cell) = ef = 1 - (1-Pf(z))^q
P(Y=1|occupied cell) = ed = q Pd(z) + (1-q) ef
eligible = (Y=1) AND (posterior >= .99) AND (ed >= 4.0 * ef)
```

The altitude cutoff is **emergent from Pd, Pf, q, and posterior**, not an explicit flight constraint. For well-covered cells this permits 75m and generally excludes 150m with current Pd/Pf; partially covered cells may need to descend farther. Actor may explore high and descend selectively toward promising local belief. `investigation_shaping` uses the quality-rule full-cell evidence horizon (~100m) rather than fixed fine-altitude50m. The ratio4 is a *proposed experiment hyperparameter*, not theoretically or empirically optimized.

Historical mode `fine_altitude` remains the default and strictly preserves 18d/e079-style confirmation at `z<=50`. `observation_version=9` is assigned to adaptive NoOracle (historical NoOracle stays version8): **do not load/resume historical weights or compare science hashes as equal**.

Experiment configs:
- `configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml` — preserves historical H256x512, E4096/U4/B8192, replay1M/75%CPU, seed44, z<=150. Intended prospective training configuration only; **20k not launched**.
- `configs/matd3-nooracle-adaptive-altitude-lr4-h256x256-20261123.yaml` — isolated smaller-network ablation, otherwise E4096/U4/B8192.

## 3. Full flow audit

### Pre-sensor and flight
- Validate config: 3D, NoOracle reports, increasing altitude anchors, nonincreasing Pd, nondecreasing Pf; configured random seed, deterministic Warp runner, altitude max150m.
- Actor produces 3D motion and RF TX-power control plus categorical silence/peer/GCS.
- Warp rejects unsafe 3D movements, handles battery, obstacles and boundaries; observation includes own altitude/speed, local belief patch and aggregated local belief, not hidden target coordinates.

### Camera/Bayesian
- Apply FoV geometry and 3D occlusion, sample `yes/no` by Pd/Pf using per-world deterministic RNG, calculate Bayesian posterior, then apply quality gate and `.99` threshold.
- Allocates candidate buffer based on max permitted FoV, not legacy 50m candidate capacity (avoids explicit error116).
- Aggregate local cell hypothesis and emit NoOracle `cell_id`; false hypotheses remain possible and are not suppressed with ground truth.

### Radio/score
- REPORT enters per-UAV buffer, can relay via PHY/MAC CSMA/CA/ACK with true wireless link budget and energy consumption, then GCS receipt. Ground truth enters *only* truth scoring for discovery/delivery reward and performance metrics.
- **Fixed actual evaluator bug**: `WarpMissionEnv` previously sent 6x140 cell-hypothesis slots to `MissionMetrics`, which expects 6x35 target-indexed flags, raising `ValueError: Target observations must have consistent shapes and monotonic discovery/delivery`. Host-only `truth_aligned_report_status` maps `cell_id` to true target cell for scoring and rendering, discards false cells for true-target counts, preserves raw slot arrays as `reports.slot_known`/`reports.slot_delivered`. The Actor and MAC code do not import this adapter.

### Actual training/eval/checkpoint
- Passed short *real* CPU MATD3 train: 4 completed 12-step episodes with at least 9 Adam learner updates -> checkpoint with observation version9 -> load -> 2 held-out CPU evaluation episodes and plot/metrics. No target-oracle errors.
- Loaded actual HF best.pt checkpoints from `duy95/uav-search-checkpoints` for legacy runs `18d756ef` and `e079eace`, and completed one entire held-out CPU episode for each. Historical 18d: 1,382 steps, 40.00% discovery, 37.14% delivery; NoOracle e079: 1,500 steps, 31.43% discovery, 20.00% delivery. These are **single-episode smoke tests, NOT official validation means**. Re-run future historical evaluations using their pinned historic source for publication-grade reproduction.
- 2 seeded full-horizon NoOracle Warp CPU worlds at 1,500 steps completed without simulator capacity errors using hand-crafted exploratory altitude actions: seed44 discovered29/35, delivered16, false confirmed cells5; seed45 discovered23/35, delivered15, false confirmed cells3. These outcomes are **not results for a trained Actor**; they validate camera+radio+scoring flows and reaching 0m/150m altitude range.

## 4. Performance changes with A/B integrity

### Accepted as source improvements but not proven faster on T4
- `cell_coverage` zero-overlap fast path for a grid cell wholly outside circle using nearest-corner distance with conservative margin; returned exactly the same **all-field** Warp trajectory hashes over E4x28 and E128x42 CPU steps.
- `investigation_potential` Warp loops exactly patch cells `range(-radius,radius+1)` twice, for 5x5 belief patch reducing potential loop iterations 169 to25 per UAV when invoked; passes same all-field CPU trajectory hashes.
- A *different* fully covered-cell fast path was **rejected** because belief, belief_coarse, belief_potential, before, reward_components hashes changed from floating-point rounding; source was reverted to original integration.

CPU timing of accepted changes: E4 +1.1% on one noisy measurement; E128 -0.02% (essentially unchanged). No evidence yet of production GPU training speedup; no unsafe scientific “performance” assertions. Full GPU E4096/20k or power usage has NOT been measured for this new science.

### Previous GPU speed A/B reporting bug fixed
Historical Kaggle `372720755` baseline/candidate four runs: `baseline cold=2140.3227`, `candidate warm=6029.2511`, `candidate warm=6062.4428`, `baseline warm=6054.5347` transitions/s. Old unpaired median marked **+47.552% beneficial**, but this is cold Warp/CUDA compilation contamination. Corrected warm-only baseline6054.535, candidate median6045.847, apparent **-0.14349%**, with only one warm baseline sample: **no verified benefit**. All four had zero complete 1500-step episodes, so equal episode SHA was vacuous. Corrected analyzer `scripts/analyze_matd3_ab_cold_start_20261123.py` and original Kaggle notebook reporter now refuse a false “verified beneficial” claim.

## 5. Test evidence and limits

- TDD RED tests reproduced both the old hard50 limitation and NoOracle CPU evaluator shape mismatch, then passed after fixes.
- Focused end-to-end CPU proof: `tests/test_nooracle_adaptive_altitude_20261123.py`, `tests/test_nooracle_evaluation_truth_projection_20261123.py`.
- Original pinned/source-bundle Kaggle GPU pilot now emits `MATD3-NOORACLE-ADAPTIVE-ALTITUDE-LR4-H256x512-E4096-U4-B8192-T4X2-BOUNDED-PILOT-20261123.ipynb`. Actual new-notebook source extraction followed by **real CPU training from the extracted files** passed. GPU pilot requires **Kaggle T4×2**; hard stops after64 vector steps, with 131,072-transition warmup and ~132 learner updates. Does NOT complete1500-step episodes or 20k training, so cannot certify learned altitude behavior/mission success.
- Final related regression suite: **`96 passed`, 7 warnings in 40.10 seconds**, including notebook/worker tests, historical checkpoint one-episode smoke. Full collection contains1791 tests; this is NOT a claim that all 1791 passed.

### Additional cross-run isolation bug found and fixed

The adaptive pilot initially set process-global `UAV_TRAIN_ENV_STEP_LIMIT`, `UAV_TRAINING_ONLY`, W&B and forensic environment flags without restoring them, contaminating unrelated train/eval regression tests (17 failures in combined order, while isolated tests passed). A new TDD regression reproduced this. The worker now wraps `train(c)` in scoped environment overrides using `patch.dict(os.environ,controls)` and reliably restores original process environment even on failure. The previously failing combined regression returned to **96 passing tests**, zero failures.

### Remaining risks / blockers
1. **CUDA/E4096** source build, JIT compile, VRAM, training numerical parity and throughput need the user's Kaggle T4×2 run; connected local Ubuntu does not expose CUDA.
2. **Long-run belief capacity** and 140 report-slot ceiling need multi-seed mission endurance stress. The two 1500-step worlds succeeded (34/26 hypotheses), but that is not a proof across all random seeds/policies; if 140 unique false cells are confirmed, the simulator fails closed.
3. The exact ratio4/confirmation0.99 reward balance must be trained anew from scratch and validated across 100 seeds/checkpoint and held-out 1000 Final Test, including mission time and false reports; cannot infer success from synthetic high-belief tests.
4. Causal hardware timing may be dominated by event-driven PHY/MAC `packet_step`, Bayesian sensing and replay. GPU pilot plus separate kernel profiling are needed before any further speed claim.
5. Posttraining GPU deferred checkpoint selection, W&B/HF artifact publishing and multi-seed final test have **not** yet been executed with the newly trained MDP. The original two historical fullflows remain available and unchanged.

## 6. References

- Training provenance W&B `18d756ef`: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef
- Training provenance W&B `e079eace`: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e079eace
- Hugging Face historical best metadata: https://huggingface.co/duy95/uav-search-checkpoints/tree/main/deferred-gpu-eval
- HF archived diagnostics: https://huggingface.co/datasets/duy95/uav-search-diagnostics/tree/main/runs
- Kaggle previous A/B probe: https://www.kaggle.com/code/therwed/notebook43d7d397be/notebook?scriptVersionId=372720755
