# v6.6 Independent Scientific Audit — Execution Plan

> For agentic workers: use superpowers:executing-plans sequentially; each item ends in a RED→GREEN regression and never bypasses physics or packet parity.

**Goal:** Verify and correct narrow v6.5 defects while preserving original source behavior and a portable 500-lines-per-file ZIP.

**1. Input inspection & reproduction.** Read AGENTS.md, README.md, configs, all v6.5 tests; record baseline pytest/policy/manifest results before edits. Preserve the original ZIP unchanged.

**2. Map normalization.** Write tests for 2km obstacle radii, UAV/GCS spawns, and reproducible direct GCS grid sweep; observe RED, patch `communication/common.py` and `environment/core.py`; rerun GREEN.

**3. Compact3 action parity.** Test nearest dirty receiver instead of nearest merely alive peer; test blackout/unavailable transmitter; preserve topology-only fallback from old regression; patch `gpu/runtime_v65.py`, run GREEN + v6.5 old regression.

**4. Real data-mule test.** Packet experiment two distinct disconnected-contact phases, prove evidence and E2E ACK arrive at GCS only with multi-origin forwarding enabled; warn that changing topology between phases is not a flown trajectory.

**5. GPU claims.** Test compressed buffer class lacks packet step, 4D ACK omitted and memory estimate undercounts data-mule; patch preflight to BLOCK until fully integrated Warp kernels + CPU parity.

**6. Ablation profiles.** Test peer on/off only toggles `enable_peer_dissemination`, 2km targets scale with area. Add explicit `--cpu-gate-datamule` and run both algorithms on four 1/2km profiles.

**7. Sources, publication integrity, packaging.** Validate upstream `utils/config.py` and scholarly texts, write caveats, refresh SHA-256 manifests, full project pytest, build ZIP, unzip in Git-free directory and re-run checks. No Kaggle credentials or GitHub push.
