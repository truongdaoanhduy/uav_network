# Engineering contract — standalone seven-radio UAV study

This is a new environment with 6 learning UAVs and one independent GCS. The previous environment, root configs, runner collection and other learning schedules are NOT part of the current experimental specification.

## Architecture and reproducibility

1. Source code belongs to named modules/folders and each owned code/text file must contain no more than 500 lines.
2. Validated YAML/config/CLI is the only normal place to change experimental parameters.
3. Separate the scenario environment (research/zhang_xue) from algorithm hyperparameters (uav_network/algorithms).
4. Decentralized Actor has no true-target access. The centralized critic may receive permitted global training state.
5. Avoid physical discontinuity, missed swept collisions, false evidence duplication and fake timestamp renewal.
6. Preserve packet custody, actual MAC ACK and E2E ACK, queue TTL, routing and RF energy semantics.
7. Synthetic CPU learner tests prove only algorithm operation. Validate actual CPU–Warp event parity and GPU T4 separately.
8. Preserve run seed, scientific config, code SHA and checkpoint contract for every valid experiment.
9. Never track generated checkpoints, model snapshots, output logs, datasets or private credentials.

## Scope of retained utilities

The retained algorithms are MATD3, MASAC and their Vanilla counterparts, with supporting network/replay code. The retained experiment files are reusable offline metric statistics, aggregation, plotting and transition reward collection. They do not by themselves provide an online W&B or Hugging Face pipeline for this scenario.

Old experiments are recoverable from the immutable v5.9 Git commit but intentionally absent from this branch. Do not silently reuse their training settings or validation results.
