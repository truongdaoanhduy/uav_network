# Zhang-Xue v5.8 source publication (2026-11-25)

- Branch: `experiment/zhang-xue-v58-audit-20261125`.
- Location: `research/zhang_xue_v58_20261125/`; standalone experimental package isolated from the existing `uav_network` MARL implementation.
- Source and tests are reconstructed from the uploaded v5.7 archive plus byte-verified v5.8 source changes. The v5.8 source, scripts, new regression tests and audit report are included.
- The README inherited from the original package references `stage_v58_on_v57.sh` used to construct a standalone copy; this published Git branch is **already staged**, so that script is intentionally not included. It is not necessary for running these tests.
- The performance/research status is NOT production-ready: CPU-Warp parity, CUDA T4 profile, fully coupled 4096-env training and safety guarantees have NOT been verified.
- This commit does not change the default branch or existing learning algorithms.

## Verify

```bash
cd research/zhang_xue_v58_20261125
PYTHONPATH=src python -m pytest -q tests/
PYTHONPATH=src python scripts/v58_scenario_audit.py --strict
```

The scenario audit intentionally refuses production training while research mismatches remain.
