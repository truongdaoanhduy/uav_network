# Seven-radio target search environment

The scientific scenario has 6 UAV radios and one distinct GCS receiver, a 20x20 belief grid, building-aware sensing, a CPU packet-level CSMA/CA reference and experimental NVIDIA Warp kernels.

Canonical YAML: configs/scenario_7radio.yaml.
Small CI smoke YAML: configs/cpu_smoke.yaml.

Run from this folder:

    PYTHONPATH=src python -m pytest -q tests/
    PYTHONPATH=src python scripts/run_scenario.py --config configs/scenario_7radio.yaml --steps 4 --seed 44

The CPU script is not an MATD3/MASAC trainer. The learners' continuous head has two motion values plus one RF-power value. The current environment uses fixed RF power and a separate destination mask, so a formally tested observation/action adapter is needed before training.

The top-level version-suffixed files in src are compatibility imports for the current modular scenario implementation, NOT the old root scenario.

Online tracking or checkpoint archival for this new scenario has not been built. Avoid mixing old experiment results with these metrics.
