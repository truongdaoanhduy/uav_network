# Paired relay and power audit

Run on CPU using the configuration saved in the checkpoint:

```bash
PYTHONPATH=.:tests .venv/bin/python scripts/audit_relay.py /path/to/checkpoint.pt \
  --output-dir outputs/relay-audit --seeds 2000000044 \
  --modes learned direct_only fixed_power --trace
```

Omitting `--seeds` selects 30 consecutive seeds starting at the saved
`runtime.eval_seed`. An explicit subset must contain 1–30 unique unsigned
32-bit seeds. All modes use identical policy bytes, saved configuration, and
seed bank. The audit records the actual completed training episodes and global
steps; a partial checkpoint is not described as fully trained.

`learned` copies the policy action. `direct_only` preserves silence and existing
GCS selections; it redirects a peer selection to GCS only when the original
pre-action GCS admission mask permits it, otherwise selecting silence.
`fixed_power` preserves movement and destination and maps physical watts back
through the existing power mapping. The default is the saved maximum TX power;
`--fixed-power-w` must lie within saved physical bounds. These are communication
ablations of a fixed policy, not retrained direct-only baselines. Each action's
movement components are preserved; later physical trajectories can diverge
because battery and delivery feedback affect later observations.

The audit writes `protocol.json` before episodes, atomic complete case files in
`cases/`, `progress.json` every ten steps and at episode boundaries, and
`summary.json` after each completed episode. The protocol fingerprints checkpoint
SHA256, full configuration/scenario, seed bank, modes, physical fixed power, trace
setting, and controls version. Restart the identical command to reuse complete
matching cases. Incomplete/malformed cases are rerun; changed protocols require
another directory. `summary.complete` refers to the entire requested matrix;
per-mode completeness is also explicit. Use one writer per output directory.

Summaries retain MissionMetrics and OutcomePool packet PDR, power/underpower
diagnostics, search, delivery, success, battery, and successful direct/relayed
GCS report counts. `total_mission_uav_energy_j` includes UAV propulsion plus
UAV TX/RX radio consumption across all completed missions, including failures.
`mission_energy_j_per_delivered_report` uses that numerator.
`total_mission_uav_radio_energy_j` and
`uav_radio_energy_j_per_delivered_report` expose radio consumption separately.
Both ratios are null when zero reports were delivered; both exclude GCS energy.
`--trace` saves actual successful whole-report custody transfers independently
of animation. Power is the last fragment's TX power, not a per-packet log.

## Privileged feasibility

`run_feasibility(config, seeds, margin_db=3.0, on_progress=None)` uses the selected
backend and preserves the supplied horizon, targets, obstacles, reports, and
buffer sizes. For `warp_packet`, it creates one CPU Warp world and adapts the
existing privileged search/motion/power controller through a read-only view and
the backend's observation admission masks. No scientific kernels change.
The result records backend, seed, initial layout hash, scenario, raw metrics,
and `not_a_policy_baseline: true`. Equal native/Warp numeric seeds do not
establish equal geometry. An oracle failure describes this controller and does
not prove mission impossibility. The optional callback receives episode start,
each completed step, and episode-complete progress; interrupted calls do not
return a complete result. Existing `run_explicit_feasibility` stays native.

Future Kaggle runs use
`configs/comparison-20k-steps1800-scenario-v3.yaml`:
4096 environments, 20000 completed episodes, 1800 steps, seed 44, 35 targets,
25 obstacles, 1MB reports, and 3MB buffers. All other scientific and algorithm
parameters match the prior 2000-step profile. Existing checkpoints retain their
saved 2000-step evaluation horizon. Exact full-state continuation across the
horizon change is incompatible.
