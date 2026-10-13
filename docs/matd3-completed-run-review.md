# MATD3 completed-run review — 2026-10-13

Run: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/afdad4a9
Finished after2446 seconds (~40m46s);1000 cumulative episodes,776026 transitions,
384014 learner updates. This segment restored episode940 and collected60 new
episodes. Adam/replay were recreated. Sourcea855862, not the newer diagnostics code.
Final artifact: truongdaoanhduy-uit/uav_search_target/uav-network-matd3-afdad4a9:v6.
All final model tensors are finite.

## Actual episode history

The final checkpoint contains1000 original episode records; aggregates match W&B.
These combine the prior940 episodes with the60-episode continuation.

| Window | Return | Search % | Delivery % | Success % |
|---|---:|---:|---:|---:|
| All1000 | -140.709 | 73.28 | 60.14 | 5.6 |
| First100 | -503.113 | 39.2 | 27.7 | 1 |
| Last100 | -164.429 | 71.7 | 57.2 | 2 |
| Episodes901–940 | -89.789 | 77.5 | 61 | 2.5 |
| Continuation941–1000 | -214.189 | 67.833 | 54.667 | 1.667 |

Different training episodes use different maps/seeds and exploration; these windows
describe trends rather than proving a causal effect of restarting.

944 episodes ended at the time limit and56 by mission success.
None had an exhausted UAV. Mean final battery51.35%; minimum over every UAV/
episode17.30%. Battery capacity is not the binding failure condition in this run.

Mean blocked-movement rate50.86%, boundary-clip rate49.75%.
Mean expired reports1.471, dropped reports0.088 per episode.
Mean total propulsion energy809151.44J versus UAV RF energy0.06379J per episode.
With the present payload/radio accounting, RF-energy tradeoffs are extremely small.

Last episode1000:800 steps,4/10 found and delivered, return-523.535,
3740 blocked UAV-step actions,2942 boundary clips, all6 UAVs active.
Average battery54.70%. Blocking penalty-748 dominates the+40 discovery/+200
delivery gains; step cost-8 and flight penalty about-7.535 explain the return.
No expiry/drop in that last episode. Packet PDR100% only refers to packets sent;
it does not imply the mission delivered all10 targets.

Across all1000 episodes:23299 generated packets,14425 received, weighted PDR61.913%.
Report hop count weighted by delivered targets1.931, demonstrating relay use.
Native metrics also record1490 collisions and44792 PHY failures.
Per-episode average PDR and weighted packet PDR are different summaries.

## Deterministic evaluation

Two checkpoints evaluated on the same20 maps/seeds10044–10063 using the saved
mission configuration and actual CPU UavNetSim. Exploration disabled.

| Checkpoint | Mean return | Search % | Delivery % | Success % | Blocked % |
|---|---:|---:|---:|---:|---:|
| Episode940 | -160.253 | 69 | 55.5 | 0 | 49.75 |
| Episode1000 | -326.901 | 57.5 | 44.5 | 0 | 59.94 |

Final policy is weaker on this20-map set, and neither completes a mission.
This is one training seed and20 evaluation scenarios; it is not a general
statistical claim that MATD3 is inferior.
Final policy has85.48% of movement components at magnitude>=0.99.
No evaluation UAV exhausted its battery; mean final battery52.20%.

## Confirmed routing-admission issue from execution

Seed10046 was rerun with a side-effect-free observer before network admission.
Return matched the uninstrumented run within1e-8:
-628 selected transmission actions, of which50 passed report eligibility.
-577 selected a UAV already knowing every report held by the sender.
-1 failed range/other admission checks.
-Actual packet metrics:18 generated,11 received,35 PHY failures,4/10 targets delivered.
-Mean requested TX power among eligible choices0.36463W.
These are action decisions, not628 physical transmissions.

destination_mask currently advertises range/liveness/buffer-at-source;
_backend._eligible additionally checks ReportStore.can_receive().
Consequently a policy-visible destination can reject every held report.
The actor does not observe each peer's per-target knowledge.
This confirms an admission/mask mismatch causing ineffective routing decisions,
not a proof that it is the sole reason for low training quality.
A fix needs explicit report-admission/handshake semantics or observable peer
metadata; avoid silently giving the actor privileged information.

## Logs and limits

The submitted old tracker used console=off. W&B console API returns0 captured
lines. Kaggle output API presently exposes no MATD3 log files while the joint
job still runs MASAC. We read the actual1000 structured episode records and
performed fresh execution/evaluation; full stderr has not been inspected.
Finished status and finite final tensors show no observed numerical crash, but
do not prove absence of all logical errors.
The new diagnostics source captures console/failure context for future launches.

The60-episode continuation declines and resets optimizer/replay, but that
association is not enough to establish why the policy degraded. Detailed Q/TD/
gradient traces are unavailable for this old-source run.

## Next priorities

1. Correct report eligibility versus available-destination semantics and log
   rejected selected actions without inventing packet sends.
2. Investigate saturated movement and prolonged obstacle/boundary blocking.
3. Select checkpoints by held-out evaluation; do not assume final is best.
4. Revisit RF/pin modeling only with meaningful, documented energy assumptions.
5. Compare before/after on the same maps, preserving algorithm/scenario baselines.

Evidence is retained under ignored outputs/matd3-completed-review:
checkpoint, episode records, analysis.json, routing-probe.json, both20-seed
evaluation results, logs and previews. No checkpoint or credential is committed.
