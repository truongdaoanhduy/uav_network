# Current task — V93 scenario audit, not frozen

Working branch: experiment/v93-scenario-audit-20261127.
Verified implementation commit: a571e9ee6972b33d992fbde83ef81cc1bd52b169.
Parent verified GitHub/Sandbox commit: f0433d51b5de1cf90b5b265221c5e9ad606f47b9 (V91).
V92 attached package was compared: 331 files identical to V91; three added probes/tests and one changed manifest.

- Active candidate YAML: research/zhang_xue/configs/scenario_v91_bayes_1km.yaml.
- Historical DS comparison: scenario_v89_static_1km.yaml. These are not reward-matched.
- Source changes are scenario MAC, validation, fail-closed safety and audit tooling only.
- MATD3/MASAC learner, loss, optimizer, actor and critic source are unchanged.
- Main report: docs/V93_SCENARIO_AUDIT_VI.md.
- Reproduction: scripts/gpu/v93_mission_audit.py, v93_network_microbench.py, v92_network_and_peer_probe.py.
- Current machine: Vast RTX3060 12GiB, one logical cuda:0; Warp1.18+cu12 and torch2.7.1+cu128.
- State remains CANDIDATE_NOT_FROZEN / NO_GO_BASELINE / NO_TUNING.
- Do not equate Warp CPU/CUDA event parity with UavNetSim macro equivalence or field RF calibration.
- Fail-closed rejections are reported separately from a committed collision. A necessary pair viability guard is not a complete motion planner.
- Preserve all failed probe logs and first-terminal snapshots in the delivered evidence package.

Next: resolve the remaining scope-specific validation and coordinated safety limitations using the evidence matrix before tagging SCENARIO_FROZEN.

Verification receipt: CPU502 PASS /23 skipped /103 subtests; CUDA519 PASS /6 skipped.
E4096x900 environment8/8 gates PASS,78.3916s; sampled device peak2572MiB.
See docs/V93_GATE_STATUS.json. Passing regressions does not close the scientific OPEN gates.
