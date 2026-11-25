# Current task — seven-radio environment only

Branch: experiment/zhang-xue-v60-newscenario-only-matd3-masac-20261125.
Parent reference commit: 82512e827b5a4c9f5c6fda1283a628967d0bbf3c.

- Canonical scenario config is research/zhang_xue/configs/scenario_7radio.yaml.
- Old root configs/environment/trainers/async pipelines are retired from HEAD.
- MATD3/MASAC learner code and generic offline evaluation/telemetry are retained.
- No full training loop connects the algorithm to the new observed belief maps and route-action mask.
- No new online W&B/HF connector or live GPU/T4 benchmark is claimed.
- Preserve tests and enforce current source rules from AGENTS.md.

Only operate on this isolated branch. Verify remote SHA after push.
