# UAV search target — independent Zhang-Xue seven-radio scenario

This branch contains ONE current research scenario: six UAVs, one separate GCS, 20x20 Dempster–Shafer belief map and packet-level CSMA/CA. It does not reuse the earlier UAV training scenario, its configs or its asynchronous learner.

## Where the active code lives

- research/zhang_xue/src/zhang_xue/: scenario, belief, mobility, packet network and Warp kernels.
- research/zhang_xue/configs/scenario_7radio.yaml: canonical research configuration (1000m, 15 targets).
- research/zhang_xue/configs/cpu_smoke.yaml: small CPU smoke configuration.
- uav_network/algorithms/: retained MATD3, MASAC and Vanilla baseline mathematics; learner-specific hyperparameters.
- uav_network/experiments/: generic evaluation statistics and metrics aggregation ONLY.

The historical scenario and run configs were removed from this branch, but their history is preserved in the v5.9 parent commit.

## Run CPU regression

    bash setup.sh --dry-run
    PYTHONPATH=.:research/zhang_xue/src python -m pytest -q tests/ research/zhang_xue/tests/
    PYTHONPATH=.:research/zhang_xue/src python research/zhang_xue/scripts/run_scenario.py --config research/zhang_xue/configs/cpu_smoke.yaml --steps 4

No current MATD3/MASAC training driver is connected to the new 514d actor observation and separate destination mask. The algorithm sources can be loaded and smoke-tested, but this is not a full training pipeline.

Online W&B and Hugging Face persistence from the old scenario are intentionally NOT carried over. They require a new integration only when a matching trainer/evaluator exists. No GPU T4 throughput or CPU–Warp parity is claimed.

Generated maps are seeded and created at reset; no external dataset is required. All outputs, checkpoints and logs belong under Git-ignored outputs/.
