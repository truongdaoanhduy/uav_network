# Historical MATD3 `18d756ef` vs new MAP800 TX0.8W

## Verified historical source — DO NOT rewrite the old run

Original W&B run: `MATD3-EVAL100-ACTORBN-PLAINCRITIC-E4096-U4-B8192-H256x512-S44-18d756ef`.
Pinned git commit: `0804dd75af39351c5628b4f2abda930dcbfedac7`.
Source: `uav_network/environment/gpu/bayesian.py`, line171.

The **historical confirmation** is `if yes and pos[2] <= p.fine_altitude and posterior >= p.confirmation_threshold`: posterior>=0.99, altitude<=50m. The UAV flight envelope is 0-150m. **There is NO `confirmation_min_likelihood_ratio=4` or independent `LR>=4` confirmation criterion in this historical source.** Bayes calculations use likelihoods `a,b` to update posterior; this is not an LR-threshold rule. The code subsequently uses ground-truth target XY for target-ID report creation; do not call this the fully NoOracle scenario.

Read-only source link: https://github.com/truongdaoanhduy/uav_network/blob/0804dd75af39351c5628b4f2abda930dcbfedac7/uav_network/environment/gpu/bayesian.py#L145-L178

## Selected new science branch (fresh run only)

- MAP 800m x 800m, Bayesian belief grid40x40, 6 UAVs, 35 targets, 25 obstacles.
- Adaptive-altitude **confirmation requires positive observation AND posterior>=0.99 AND positive LR>=4**, permitting confirmation from above50m when measurements are individually informative (not forced to descend to50m). FOV and Pd/Pf vary with altitude. Altitude envelope remains0-150m.
- MAP800 MATD3 architecture H256x256 (not historical H256x512), Actor BN, Critic plain, orthogonal initial weights, torch.compile, train config E4096/U4/B8192/seed44. Source uses NoOracle cell reports and no ground-truth target ID for policy report generation.
- **TX max0.8W**, min0.001W, Actor-learned logarithmic continuous power. The Actor also independently selects silence, a peer relay, or GCS (subject to PHY conditions/500m hard range). This does **not** set fixed TX=0.8W.
- Radio TX/RX and flight subtract physical Joules from same battery. RF normalized-reward reference pinned to 0.4W (`reward.communication_reference_tx_power_w=0.4`) so changing Pmax does not make identical joules artificially cheaper in the RL objective. Reward coefficient `communication_efficiency=1`. Historical reward/MDP is not identical; train from scratch.
- New config: `configs/matd3-nooracle-lr4-map800-h256x256-orthogonal-compile-tx08-learned-hf-20261123.yaml`. Existing historical, MAP800 0.4W, earlier 0.8W ablation, MAP800 5W branch configs remain unchanged.
- Deep HF/W&B training launcher (DRY RUN unless `--execute`): `scripts/launch_nooracle_map800_tx08_training_with_hf_forensics_20261123.py`. On actual GPU run, logs two selected worlds every16 vector steps (plus early dense window), full belief, action head/continuous RF, route, actual pre/post TX/RX energy and packet power, replay minibatches and root-cause-gradient timelines; per-episode RF power histograms and diagnostics are logged to W&B every32 episodes; incremental public HF archives are readback-SHA256 checked. **Not all E4096 transitions are individually captured, only sample worlds/aggregate statistics.** Actual HF upload requires Kaggle GPU+credentials; none was run in this local audit.
- A separate production Train->GPU Val100->Best->Test handoff is required for a complete experiment; launcher only starts train. No 20,000-episode policy or end-to-end Kaggle E4096 speed measurement is claimed.

## Verification of new TX0.8

- 93 selected regression tests PASS + 14 subtests, 7 warnings. Includes MAP800 Bayesian LR4, privacy, TX hardware and reward, HF/W&B network fake integration with **real compiled CPU MATD3+Warp**, prior0.4W regressions.
- MAP800 full-horizon manually generated actions, 16 CPU Warp worlds x1500 steps: no internal runtime errors, z_max150m, max51 allocated observed-cell report hypotheses, 502/560 true targets found, 399/560 delivered. These are **not trained MATD3 results** and are not proof learned TX0.8 outperforms0.4/5W.
- TX0.8 enables some NLOS links that fail at0.4W, but 0.8W remains well below required TX for long obstructed 200m+ links. Need controlled multi-seed experiments and RF-vs-propulsion energy comparisons to establish actual benefit.
