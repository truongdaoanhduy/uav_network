# Current task — seven-radio environment only

Current verified branch: experiment/zhang-xue-v61-full-gates-20261125.
Baseline: experiment/zhang-xue-v60-newscenario-only-matd3-masac-20261125.
Parent reference commit: 82512e827b5a4c9f5c6fda1283a628967d0bbf3c.

- Canonical scenario config is research/zhang_xue/configs/scenario_7radio.yaml.
- Old root configs/environment/trainers/async pipelines are retired from HEAD.
- MATD3/MASAC learner code and generic offline evaluation/telemetry are retained.
- No full training loop connects the algorithm to the new observed belief maps and route-action mask.
- No new online W&B/HF connector or live GPU/T4 benchmark is claimed.
- Preserve tests and enforce current source rules from AGENTS.md.

Only operate on this isolated branch. Verify remote SHA after push.

## v6.1 result (source/CPU only)

- Full suite: 112 PASS, 50 additional unittest subtests PASS.
- Warp 1.18.0 CPU 2-hop DATA/E2E ACK and reset isolation PASS.
- CPU MATD3/MASAC: each consumed 4 real scenario transitions and executed
  one optimizer update using the corrected actor 522d / joint state 3132d.
- Source governance and setup validation PASS.
- GPU CUDA/T4, end-to-end per-observation AoI and UavNetSim exact parity
  remain unverified or unimplemented; consult docs/V61_STEP_BY_STEP_QA_REPORT.md.
