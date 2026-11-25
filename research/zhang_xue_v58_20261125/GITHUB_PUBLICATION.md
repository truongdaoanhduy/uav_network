# v5.9 architecture restoration and notebook retirement (2026-11-25)

- Branch: `experiment/zhang-xue-v59-architecture-gates-20261125`.
- Restores 12 October source rules through an explicit governance policy/CI gate;
  `src/zhang_xue/` groups environment, movement, belief, 7-radio CPU, Warp and
  configuration source into modules each **<=500 lines**.
- Legacy `uav_v54_*`/`v56_warp_*` public import paths remain compatible.
- CPU reference 80-step zero/full communications replay is identical to v5.8;
  Warp 1.18 CPU two-hop relay GCS DATA/MAC ACK/E2E ACK completed.
- Removes all 25 historical `.ipynb` files from current HEAD by user request
  (~19.51 MiB). Historical files remain at immutable v5.8 commit
  `7d7ab1ccea3d31a121f9708e428bbf96c3c529a6`; SHA256 manifest under
  `docs/benchmarks/NOTEBOOK_ARCHIVE_V59_20261125.md`. No Git history rewrite.
- Root `setup.sh` provides pinned upstream dependencies; the new research
  `setup.sh` delegates to root and tests configuration imports.
- New production-grade CUDA T4 4096-env training is **NOT** verified or
  integrated. Whole repository still has pre-existing overlong source files.

## v5.9 verification

```bash
PYTHONPATH=research/zhang_xue_v58_20261125/src \
  python -m pytest -q research/zhang_xue_v58_20261125/tests/
python scripts/check_project_policy.py --scope changed \
  --base 7d7ab1ccea3d31a121f9708e428bbf96c3c529a6
```

---

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
