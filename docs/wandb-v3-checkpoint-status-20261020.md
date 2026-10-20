# W&B V3 checkpoint status — 2026-10-20

Fresh read-only W&B API snapshot from project
`truongdaoanhduy-uit/uav_search_target` at approximately 03:28 UTC.
No credentials are recorded here.

## Four checkpoints used by the paired relay/power audit

| Run ID | Variant | State | Global step | Completed episodes | Search progress | Delivery progress | Mean battery |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1b6cacd3` | MASAC BASE | finished | 1,863,680 | 0 | 89.27% | 43.03% | 68.89% |
| `3e97c782` | MATD3 BASE | finished | 2,846,720 | 0 | 84.86% | 24.08% | 56.30% |
| `11869744` | MASAC PQL | finished | 1,974,272 | 0 | 84.55% | 45.08% | 67.53% |
| `89f59c30` | MATD3 PQL | finished | 2,568,192 | 0 | 76.70% | 19.49% | 60.17% |

All four runs report:

- configured training target: 20,000 episodes;
- 4096 environments, seed 44;
- historical saved horizon: 2000 steps;
- `episode_budget_complete=false`;
- `train/episodes_completed=0` and `cohort/episodes=0`;
- zero completed-episode network/report cohort samples.

The `progress/*` search/delivery/battery values above are therefore in-flight
collector progress from an unfinished first cohort, **not final episode metrics**
and not evidence of convergence. They may be useful for debugging the checkpoint
state but must not be ranked as final algorithm performance.

A project-wide W&B query returned no runs in state `running` at this snapshot.
The four 30-seed relay/power evaluations currently executing locally are offline
fixed-checkpoint audits, not W&B training runs.

## Interpretation

The online evidence agrees with checkpoint metadata used by the audit: these are
partial historical checkpoints. Their unequal global-step exposure also prevents
a fair BASE/PQL or MASAC/MATD3 algorithm ranking. Communication ablations remain
valid as fixed-checkpoint paired analyses when interpreted per checkpoint, but
paper-quality algorithm conclusions require completed/equally-budgeted training
and the planned paired evaluation protocol.

New Kaggle algorithm/training runs remain subject to the 1800-step profile; this
document records historical 2000-step checkpoint runs only.
