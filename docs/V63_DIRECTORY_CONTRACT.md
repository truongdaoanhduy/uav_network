# Project-wide directory, config and size contract

- `uav_network/algorithms/`: MATD3 and MASAC in separate `matd3/` and
  `masac/` packages; shared math/replay in `core/`, historical control
  experiments in `baselines/`, typed YAML configuration in `configuration/`.
- `uav_network/experiments/`: reusable offline evaluation and plotting.
- `research/zhang_xue/src/zhang_xue/`: only current scenario implementation,
  organized by `environment/`, `belief/`, `mobility/`, `communication/`,
  `gpu/kernels/`, `learners/`, `configuration/`, `compat/`.
- `research/zhang_xue/configs/`: strict scenario YAML. Only current
  1000 m/20x20 packet wire format is supported; no silent 3000m preset.
- `configs/learners/`: learner YAML (MATD3/MASAC hyperparameters), not
  scenario. `configs/data/`: declared reproducible sources and checksums.
- `scripts/data/`: source preparation and optional pinned external downloads.
- `scripts/maintenance/`: full-tree policy and SHA-256 manifest verification.
- `research/zhang_xue/scripts/`: CPU experiment, network simulation,
  and optional GPU preflight, with older reproduction under `legacy/`.
- `tests/` and `research/zhang_xue/tests/`: structural and scientific gates.
- `docs/`: engineering rationale. `.github/workflows/`: policy checks.
- **ALL generated output**: ignored `outputs/` (maps, datasets, weights,
  activation traces, replay, CSV, logs, checkpoints). External heavy data:
  source manifest with HTTPS + SHA256, not committed binary.
- Hard-fail on unknown YAML key, wrong types, broken import, source >500 lines,
  tracked secrets, invalid numerical parameters; untracked files are excluded
  from the release ZIP by `git archive`. GPU parity still needs separate tests.
- Compatibility packages `uav_network.algorithms.checkpoints` and
  `uav_network.algorithms.vanilla_baselines` preserve old API imports without
  duplicate algorithm source files.

## Config and CLI

```bash
# Check all packages and local source policy
bash setup.sh --check-only
# Synthetic canonical map dataset in ignored outputs/data/
bash setup.sh --prepare-data --seed 44 --episodes 4
# Test two real MARL updates with TX learned and AoI on CPU
bash setup.sh --cpu-gate
# Override only declared algorithm and scenario fields, no code edit
PYTHONPATH=.:research/zhang_xue/src python \
  research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py \
  --algorithm matd3 --transitions 4 \
  --learner-config configs/learners/matd3_cpu_gate.yaml \
  --learner-set matd3.gamma=0.98 \
  --scenario-config research/zhang_xue/configs/scenario_v63_learned_tx_aoi.yaml \
  --scenario-set pd=0.90
```

**Runtime limitation:** CPU MAC timing settings are dynamic, while Warp
kernels still use compile-time constants; 40×40 grid, altitude controls,
full GPU env and compressed runtime are not configurable without code.

## Verify source checksums before starting a run

```bash
python scripts/maintenance/verify_source_manifests.py --check
```

The manifests record **tracked source only**, never generated data in
`outputs/`. To update them after modifying/staging code:

```bash
git add -A
python scripts/maintenance/verify_source_manifests.py --write
git add SOURCE_SHA256_MANIFEST.json research/zhang_xue/SHA256_MANIFEST.json
```
