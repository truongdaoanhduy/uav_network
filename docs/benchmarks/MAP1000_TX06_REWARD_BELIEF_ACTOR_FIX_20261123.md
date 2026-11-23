# MAP1000/TX0.6 NoOracle — calibrated Bayesian prior, RF reward, Actor stabilization

**Date**: 2026-11-23. **Status:** NEW research ablation, CPU full-flow and multiworld mission verified, **CUDA T4×2 full 20k policy quality NOT verified**. Historical run `59042708`, its W&B/HF logs and checkpoints untouched. Branch `experiment/map1000-prior-calibrated-reward-aligned-actor-normalized-20261123`.

## Evidence-based reason for this change

The 2026-11-23 W&B/HF run `59042708` became more *rewarded* while finding and delivering fewer targets. HF archive 000034 vs 000050:

| Mean episode statistic | Archive 34 | Archive 50 |
|---|---:|---:|
| Return | -880.260 | -223.187 |
| True target found | 31.679 | 23.672 |
| True target delivered to GCS | 22.658 | 15.929 |
| Map coverage (%) | 86.402 | 75.847 |
| Peer transmissions selected | 954.435 | 63.941 |
| RF efficiency reward component | -1050.969 | -136.615 |
| Unsafe attempts | 7503.861 | 8424.277 |
| Commanded stalls | 5381.481 | 7008.231 |

The original RF cost is `-reward.communication_efficiency * (delta communication_energy / report_reference_tx_energy)` in `uav_network/communication/gpu/reports.py`, with `reward.communication_efficiency=1` and `report_reference_tx_energy≈0.005379 J`, causing approximately 185.9 RL penalty units per RF Joule. Avoiding transmissions can gain ~914 points when true delivery/discovery rewards lose ~363 points. This is objective/reward misalignment, **not an arithmetic summation error**. Posterior-only with `belief_prior=0.5` is also badly calibrated for 35 true target cells / 1600 belief cells.

## Scientific changes (four knobs only)

**Original config:** `configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml` (untouched).

**New config:** `configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml` (new checkpoint/run required).

| Knob | Previous | New | Scientific motivation |
|---|---:|---:|---|
| `sensor.belief_prior` | 0.5 | **0.021875 (35/1600)** | Initial target-cell occupancy better reflects sparseness. This is still only a uniform *approximation*: actual placement excludes certain map regions |
| `reward.communication_efficiency` | 1.0 | **0.2** | Offline sensitivity check reverses the observed bad reward ordering without deleting genuine TX/ACK energy costs |
| `matd3.output_normalization` | false | **true** | Normalizes the **motion head only** before tanh; keeps RF power head independent; avoids motion-head extreme tanh output for large common pre-tanh magnitudes |
| `matd3.actor_output_regularization` | 0.0 | **0.001** | Uses the pre-existing raw-Actor-output L2 term, which can generate gradients even if tanh is saturated; mitigates but does NOT prove elimination of boundary/stall behavior |

No physical simulator, reward arithmetic, observation/critic target fields, UavNetSim CSMA/CA/PHY, report TTL, report retransmission or false-confirm penalties changed. The `motion_safety=reject` mechanism remains; **no CBF** or retired XYZ DWA. TX maximum stays0.6W, map size1000m, 40×40 occupancy, FoV/Pd/Pf by altitude, 6 UAVs, 35 targets, 25 obstacles, 1500 steps/episode, Posterior-only positive camera+posterior≥0.99, NO independent LR gate, NO 50m ceiling. The old checkpoint cannot be resumed under this new science.

## Truth-scoring quality metrics (new)

New scorer module: `uav_network/experiments/report_quality_metrics.py`. Its inputs are **terminal world summary counters only**; it is never invoked from Actor observation, sensor report generation or packet admission. No ground-truth filtering of false reports was added.

- `training/confirmation_precision_percent = 100 * true_targets_found / (true_targets_found + false_confirmed_cells)`.
- `training/gcs_report_precision_percent = 100 * true_targets_delivered / (true_targets_delivered + false_reports_delivered)`.
- `training/false_confirmed_cells` and `training/false_reports_delivered`.
- The GPU held-out evaluator aggregates pooled *counts*, not a mean of small/single-report percentages; periodic W&B data includes these precision metrics.
- When a world makes **zero** local confirmations (or zero GCS report deliveries), precision is undefined and omitted, **not 100%**. Legacy target-ID runs missing false-count fields remain compatible; no false=0 is imputed.

This avoids mistaking an episode with 100% Discovery for perfect report precision or full mission completion. The target truth used to score precision is inaccessible to the decentralized Actor.

## Real 16-world×1500 Warp CPU mission A/B

Same scene seeds `400..415` and same scripted XYZ action generator; the report-recipient mask can differ after Bayesian evidence changes, so it is a **paired environment scenario comparison, not bit-identical full actions or trained-policy ablation**. Both runs retained UavNetSim full MAC and complete simulator, no Warp errors.

| Summed 16-world statistic | Original prior0.5, RF1.0 | New prior0.021875, RF0.2 |
|---|---:|---:|
| Targets truth-found | 535 | **531** |
| True GCS-delivered targets | 442 | **437** |
| False-confirmed unique cells | **393** | **7** |
| False report deliveries at GCS | **184** | **3** |
| Precision unique cell confirmation | 57.65% | **98.70%** |
| Precision GCS reports | 70.61% | **99.32%** |
| Internal simulator errors | 0 | 0 |

New evidence `/home/aduy/Downloads/MAP1000_TX06_PRIOR0021875_REWARD02_16WORLD_FULL_HORIZON_20261123.json`, generated by `scripts/audit_map1000_new_reward_belief_full_horizon_20261123.py`. Old reference evidence `/home/aduy/Downloads/MAP1000_TX06_AUTOTUNE_16WORLD_FULL_HORIZON_20261123.json`.

**This does not prove the new model's trained-policy Discovery/Delivery will improve.** The new policy must learn under the new reward and belief, and needs controlled independent validation.

## Fresh CPU MATD3 Train→7 Actors→Val/Actor→Best→Test

Executed `python -m scripts.audit_map1000_new_reward_belief_cpu_fullflow_20261123`, which uses this NEW config and no prior checkpoint. It completed 6 short MATD3 episodes (2 CPU worlds, horizon12 for smoke), created seven immutable Actor snapshots, validated **each** on two independent seeds, selected the best by success/delivery/discovery/return, and ran two held-out tests. Outputs `/home/aduy/Downloads/MAP1000_NEW_REWARD_BELIEF_CPU_FULLFLOW_20261123/fullflow_report.json` and log `/home/aduy/Downloads/MAP1000_NEW_REWARD_BELIEF_FULLFLOW_CPU_20261123.log`. Discovery/Delivery 0 over a 12-step smoke is expected and **not policy quality evidence**.

A separate automated regression **extracts the exact sealed source archive from the generated 3-cell Kaggle notebook** and runs real CPU MATD3 Train2→saved checkpoint→independent Test2 from extracted files, verifying source reproducibility and new parameters.

## Test coverage

- New focused TDD tests: new config versus old, exactly four science changes, no CBF/DWA, no old checkpoint, full new source SHA and mandatory Kaggle preflight files, RF head distinct from normalized XYZ head, actor pre-tanh regularization gradient, quality metric precision/undefined case, GPU held-out pooled quality, reward rank sensitivity, and exact sealed source CPU training.
- Broad regression through MAP1000 Kaggle fullflow, GPU metric formatting, latest `fe799af211` bootstrap, NoOracle privacy, Posterior-only Bayes, hybrid replay, historical evaluation compatibility. **102 passed** in `/home/aduy/Downloads/MAP1000_PRIOR0021875_REWARD02_REGRESSIONS_20261123.log`; the newly added sealed-source test separately **10 passed**.
- The connected Ubuntu host has no NVIDIA CUDA driver. T4×2 20,000 episodes, every-Actor Val100 and final Test1000 **cannot be claimed to have completed**. The notebook retains this exact protocol for a new Kaggle execution.

## Kaggle full-flow notebook

Builder: `scripts/build_map1000_reward_belief_actor_fix_ui_20261123.py`.

Notebook: `notebooks/MATD3-MAP1000-TX06-PRIOR0021875-REWARD02-ACTORNORM-T4X2-VAL100-TEST1000-20261123.ipynb`.

This is a new experiment with unique MATD3 run suffix `-PRIOR0021875-REWARD02-ACTORNORM`, not a silent hot patch to `59042708`; source/config are SHA-256 sealed. Includes W&B/HF forensic logs, hardware MAC/replay autotune guarded by state checks, E4096/U4/B8192, 20k episodes, independent Validation100 for every periodic Actor, deterministic Best selection, Test1000 on held-out seeds. Kaggle needs T4×2, Internet, authorized `WANDB_API_KEY` and `HF_TOKEN` Secrets.

## Acceptance criteria on *real* GPU

1. Preflight CUDA suites pass; science config and source hash match this commit and report metadata.
2. Check that `gpu_eval_periodic/{discovery_rate_percent,gcs_delivery_rate_percent,confirmation_precision_percent,gcs_report_precision_percent,map_coverage_percent,mean_return}` use independent seeds and the **real checkpoint_global_step**.
3. Report false confirmations, true targets found/delivered, packet collisions, PDR, raw unsafe action%, actor pre-tanh saturation and actor gradient norm *across training*. Do not select a checkpoint using training reward alone.
4. Compare on matched MAP1000/TX0.6 seed/traffic reference; prioritize held-out true GCS deliveries and nontrivial precision, target recall, coverage, success, energy/target, and duration. If precision improved but recall/delivery/coverage collapsed, **reject** the config.
5. Actual 20k training and Val100/Test1000 are required before claiming convergence, eliminating boundary stalls, or showing GPU training speedup. This fix is a scientific candidate, **not a proven production policy improvement**.
