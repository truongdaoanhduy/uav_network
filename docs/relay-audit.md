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
per-mode completeness is also explicit. `run_relay_audit` remains the single
writer that establishes the protocol. After `protocol.json` exists,
`run_relay_audit_cases` may evaluate disjoint case subsets in parallel; each case
is fingerprint-checked and atomically replaced, and already-complete matching
cases are skipped.

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

## Live paired evaluation snapshot (2026-10-20 03:25 UTC)

The existing four historical-checkpoint CPU audits were inspected in place and
left running; none was restarted. At this snapshot they had completed 183/360
atomic cases across 30 paired evaluation seeds and three communication modes per
checkpoint. These checkpoints retain their saved 2000-step horizon. This is a
legacy checkpoint evaluation, not a new Kaggle algorithm/training run; the
1800-step requirement remains binding for new training runs.

| Checkpoint | Cases | learned - direct-only | learned - fixed-power |
| --- | ---: | ---: | ---: |
| MASAC BASE | 41/90 | +2.65 pp (14 paired seeds) | -1.98 pp (13 paired seeds) |
| MATD3 BASE | 46/90 | +0.38 pp (15 paired seeds) | -1.33 pp (15 paired seeds) |
| MASAC PQL | 44/90 | +8.19 pp (15 paired seeds) | +3.27 pp (14 paired seeds) |
| MATD3 PQL | 52/90 | +0.34 pp (17 paired seeds) | -2.02 pp (17 paired seeds) |

All four audit processes were alive when this snapshot was recorded. The values
above are preliminary paired mean delivery-rate differences only. The analysis
intentionally emits no 95% bootstrap confidence interval until all 30 paired
seeds exist for a contrast, so these numbers are not final algorithm or relay/
power conclusions. The checkpoint training budgets are also partial and unequal,
which further prevents a final algorithm ranking from this snapshot.

### Paired-audit health check (2026-10-20 03:29 UTC)

A later non-invasive health check showed 189/360 completed atomic cases, up from
183/360 at the earlier snapshot. All four audit processes were alive and actively
using CPU, and all four `progress.json` files were being updated. Validation of
every completed case found zero malformed/incomplete cases and zero fingerprint
mismatches. No `.tmp` or `.partial` result files were present. Approximately 68 GB
of filesystem space remained. The only stderr message was Warp's expected warning
that no CUDA driver is available; these audits are intentionally running on CPU.

Current completed cases by checkpoint were 42/90 MASAC BASE, 48/90 MATD3 BASE,
46/90 MASAC PQL, and 53/90 MATD3 PQL. The matrix remains incomplete, so final
confidence intervals and communication-ablation conclusions are still withheld.

### Integration and baseline-preservation gate

The archived full CPU integration run for this branch completed 526 tests with
`OK (skipped=68)` and `UAV_TASK3_UNITTEST_EXIT=0`. A fresh source-tree diff from
commit `c547595042ab336140f62c101b865592132a179b` through the current audit branch
shows no changes under `uav_network/algorithms`, `uav_network/experiments/training.py`,
`uav_network/experiments/pql_training.py`, or `configs`. The post-audit changes in
that range are documentation and exact-resume verification tests only. Thus the
MASAC/MATD3 learner/training baseline source remains unchanged by the current
relay/power audit work.

This gate does not close the paired experiment itself. Final completion still
requires all 360 atomic cases (4 checkpoints x 30 seeds x 3 communication modes),
followed by a fresh final verification and the 30-seed paired confidence intervals.


## Final paired evaluation (2026-10-20)

The legacy-checkpoint communication matrix is complete: **360/360 atomic cases**
(4 checkpoints x 30 seeds x 3 communication modes). Every case was reread after
completion; all 360 were complete, parseable, matched the expected checkpoint
fingerprint/mode/seed, and contained a record. No `.tmp` or `.partial` result
files remained. Each paired contrast therefore has all 30 seeds and a 10,000-
replicate paired bootstrap 95% confidence interval.

| Checkpoint | learned - direct-only delivery | 95% CI | learned - fixed-power delivery | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| MASAC BASE (`1b6cacd3`) | +1.33 pp | [-2.38, +5.05] | -4.95 pp | [-10.29, +0.29] |
| MATD3 BASE (`3e97c782`) | -0.19 pp | [-0.86, +0.48] | -1.24 pp | [-2.95, +0.38] |
| MASAC PQL (`11869744`) | **+6.19 pp** | **[+2.95, +9.52]** | +2.76 pp | [-0.57, +6.10] |
| MATD3 PQL (`89f59c30`) | +0.38 pp | [-0.10, +0.95] | **-2.86 pp** | **[-4.48, -1.33]** |

Within these fixed historical checkpoints, the strongest supported relay result
is MASAC PQL: preserving its learned peer/GCS destination decisions improves
mean delivery by 6.19 percentage points relative to forcing direct-only
communication, and the paired 95% interval excludes zero. MASAC BASE, MATD3
BASE, and MATD3 PQL do not show a statistically resolved learned-vs-direct-only
change under this 30-seed audit.

The power ablation gives a different signal. MATD3 PQL's learned power is 2.86
percentage points worse than fixed maximum power and its 95% interval is wholly
below zero. For MASAC PQL, MASAC BASE, and MATD3 BASE, the learned-vs-fixed-power
intervals cross zero, so this audit does not establish a power advantage for the
learned control in those checkpoints.

Mode-level diagnostics are consistent with those contrasts. MASAC PQL delivered
526 reports with learned communication versus 461 direct-only and 497 fixed-
power; its learned mode completed 203 relayed GCS reports. MATD3 PQL delivered
184 reports learned versus 180 direct-only and 214 fixed-power. Fixed maximum
power also produced pooled packet PDR near 98-99% for all four checkpoints,
whereas learned MATD3/MATD3-PQL power had substantially lower pooled PDR
(73.68% and 70.93%, respectively).

These are **communication ablations of immutable partial checkpoints**, not an
algorithm ranking and not evidence of training convergence. All four checkpoint
records report `episodes_completed=0`, `episode_budget_complete=false`, unequal
global-step exposure, and a saved 2000-step evaluation horizon. Across every
checkpoint/mode, all 30 audited episodes ended with all UAV batteries empty.
Therefore this matrix can support statements about relay/power behavior at each
fixed checkpoint, but it cannot establish that MASAC, MATD3, or PQL is globally
better or energy-optimal. New Kaggle training remains governed by the 1800-step
profile.

Final verification after matrix completion ran the relay/feasibility/resume test
group and reported **34 passed, 11 skipped, 721 subtests passed**. A fresh diff
from audit base `c547595042ab336140f62c101b865592132a179b` still shows no changes
under `uav_network/algorithms`, `uav_network/experiments/training.py`,
`uav_network/experiments/pql_training.py`, or `configs`; the MASAC/MATD3 training
baseline source is unchanged.

A supplementary Kaggle attempt was not used in the scientific matrix. The
kernel failed before evaluation because the launcher assumed a dataset mount
path that did not contain `protocol.json`; no remote case from that failed run
was imported. The complete 360-case matrix above was produced and validated from
the local fingerprinted case store.
