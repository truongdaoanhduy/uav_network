# Current task — V121 DS-only scenario validation

Active candidate YAML: research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml.
Pure-GCS RF: research/zhang_xue/configs/scenario_v111_ds_rf_stress_no_peer_1km.yaml.
V105 is the separate high-noise PEER+MULE factorial arm.

- Read docs/V121_SCENARIO_AUDIT_VI.md and outputs/v121_scenario_audit/SCENARIO_FREEZE_DECISION.json.
- Scenario:1000m square,6UAVs,15static targets,20x20 grid,50m altitude,900steps,dt1s.
- DS-only on CPU/Warp: vacuous(0,0,1), raw provenance counters, SUM at GCS, eta0.
- Raw actor observation549; actor learns normalized continuousXY. CPU env.step input is m/s.
- Fixed TX0.1W, pure-GCS application evidence; HELLO/DV multi-hop forwarding remains active.
- Network is UavNetSim-inspired CSMA/CA surrogate, not upstream SimPy-equivalent.
- DS masses(.8,.8) are borrowed design strengths, not fitted Pd/Pf; correlated sensing is a stress limitation.
- Semantic pooled map channels are derived features, not a three-mass simplex.
- Algorithms30/30 and YAML bytes stay unchanged; Phase2 baseline work and training are deferred.
- The old CURRENT_TASK incorrectly named a Bayes profile; V121 explicitly fixes that metadata.
- Use fresh RED/GREEN, non-reset CUDA4096x900, shield differential, paired mission and full regression evidence.
- No GitHub push. Local audit branch preserves rollback history.

Freeze comes from the machine-readable evidence decision, not a heading or a
test count. Preserve counterexamples and failed logs. Scoped validation does
not imply full stochastic-world/SimPy parity or physical sensor calibration.
