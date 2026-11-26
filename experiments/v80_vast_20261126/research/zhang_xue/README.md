# Zhang–Xue research scenario (v6.4 current)

Read the repository-root [combined academic/code audit](../../docs/V64_COMBINED_INDEPENDENT_AUDIT_VI.md) and `configs/scenario_7radio.yaml` first. The fixed-TX main profile is **2D velocity + 3-way communication control**, TX=0.1W, CCA=-82dBm; old text below records the earlier learned RF experiments and is NOT the default pipeline. CPU scalability does not imply GPU kernel parity or zero-shot checkpoint transfer.

```bash
cd ../..
bash setup.sh --cpu-gate
bash setup.sh --test
```

---

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

## v6.3 namespace migration

Previously flat versioned source modules were grouped by responsibility:
`src/zhang_xue/gpu/kernels/` (PHY/MAC kernels), `src/zhang_xue/learners/`
(state-derived replay and legacy CPU trainer), `src/zhang_xue/compat/`
(thin compatibility adapters). Flat imports `from v56_... import` no longer
resolve; prefer `from zhang_xue.gpu.kernels.events import ...` etc.
The generated procedural data source can be prepared via root `setup.sh`;
real training logs and snapshots remain outside source Git in `outputs/`.
