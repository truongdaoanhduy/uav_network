# v5.9 modular research refactor — source contract and configuration

The v5.8 CPU packet model is now split into **named modules at most 500 lines**:
`src/zhang_xue/{communication,environment,belief,mobility,gpu,configuration}/`.
The v54/v56 top-level filenames are compatibility import aliases, not a second
implementation. The original v5.8 code still exists at its immutable Git commit.
The **500-line source rule is enforceable on changes**, but the historical
whole-repository rule still fails on inherited source and notebooks.

The canonical 1 km, 15-target research configuration is in
`configs/benchmark_1km_v58.yaml`; `configs/cpu_smoke.yaml` is for fast tests.
The 3 km, 35-target study is **not** silently supported by this 20×20 simulator.

```bash
# dry run shows pinned UavNetSim checkout, does not download anything
bash setup.sh --dry-run
# full setup invokes the root venv + pinned sparse UavNetSim checkout
bash setup.sh --system-site-packages
# environment validation without installation (choose Python with dependencies)
PYTHON=/path/to/python bash setup.sh --check-only
PYTHONPATH=src python -m pytest -q tests/
PYTHONPATH=src python scripts/run_scenario.py --config configs/cpu_smoke.yaml --steps 4 \
  --set num_targets=4 --output outputs/cpu-smoke.json
```

**Not a GPU trainer:** no E4096 T4 or CPU–CUDA event parity is claimed.
Do not run the obsolete `stage_in_repo.sh` on this already-published branch.
Project-wide rules: `../../docs/PROJECT_ENGINEERING_CONTRACT_20261125.md` at repo root.

---

# v5.8 independent audit & targeted source corrections (2026-11-25)

**Status: research candidate, NOT a completed Warp-GPU 4096-env training system.**
This package preserves the uploaded v5.7 source and audit reports, adds new
v5.8 patches/tests, and documents *unresolved* parity/scenario/reward issues.

- Baseline v5.7 tests in artifact builder: **49 passed, 2 skipped**.
- After v5.8 targeted patches: **56 passed, 2 skipped**, 0 failed.
- Warp 1.18 CPU exists on the user's sandbox machine, but no CUDA device there;
  the v5.8 patched event kernel has **not been compiled/executed with Warp**.
- Source code changes: detect terminal unsafe shield fallback, make target placement
  bounded and optionally constrained, scan Warp MAC queue for E2E ACK/routable DATA,
  compute real SoA byte budget from allocated arrays. The E2E-AoI EWMA pipeline
  and complete GPU trainer remain unimplemented.
- See **`07_V58_INDEPENDENT_AUDIT_AND_FIXES_VI.md`** for scientific audit,
  source-by-source verdicts, gate matrix and papers.
- For an exact v5.7 repository checkout, use **`stage_v58_on_v57.sh`** to create
  a *new* branch after verifying preimage SHA-256. The legacy `stage_in_repo.sh`
  stages v5.6 only and does **not** integrate all v5.7/v5.8 files.
- Read-only scenario audit: `python scripts/v58_scenario_audit.py --strict`
  exits 2 intentionally because research settings and training gates differ.
- GPU diagnostics only (requires CUDA):
  `python scripts/v58_t4_scale_gate.py --allow-experimental --devices cuda:0 cuda:1`.
  It does not run training or authenticate to Kaggle.

---

# UAV Search Target — v5.6 experimental 6-UAV + GCS packet-network research package

**2026-11-25 status:** CPU packet environment + Actor private belief observation + provenance-safe DS/AoI + reward tests PASS; NVIDIA Warp 1.18.0 **CPU-backend** 1-hop and 2-hop MAC/route/E2E-ACK/reset proof PASS. **NOT production-ready:** true Tesla T4 CUDA validation, CPU–Warp event-level parity, complete physics/battery/safety, parallel CUDA rollout + MARL optimizer, and long training are still blocked. Never describe this ZIP as a completed 4096-environment trainer.

## What changed

- `src/uav_v54_7node_env.py`: 6 decentralized Actors receive **514 features each** (14 private telemetry + 5 x 10x10 private belief/freshness channels); no truth/global GCS map in Actor; actual GCS delivery/detection reward, per-radio energy, packet-aware private ledger.
- `src/uav_v54_ds_and_battery.py`: numerically stable canonical DS, Zhang `zhang_max` baseline vs origin-tagged `sum`, optional age discount for a **received source snapshot**, not byte-identical reproduction of Xue per-measurement discounting.
- `src/uav_v54_7radio_csmaca_sim.py`: auditable CPU reference event scheduler for seven distinct radio identities, MAC ACK, DATA, TTL, E2E ACK, queue and topology.
- `src/v55_warp_actor_maps.py`: Warp spatial map projector; `src/v55_state_derived_replay.py`: no duplicate centralized state in replay; `src/train_v55_packet_cpu.py`: bounded CPU-only optimization smoke test.
- `src/v56_warp_7radio_mac.py`, `src/v56_warp_reset_kernels.py`, `src/v56_warp_injection_kernels.py`, `src/v56_warp_event_kernels.py`: **real `@wp.kernel` state transitions** for 7 radios (6 UAV + GCS): reset, sensor evidence, packet injection, CSMA/CA / CCA / DIFS / backoff, PHY SINR, MAC ACK, distributed HELLO routes, multi-hop relay, dedup, GCS application E2E ACK. `WarpSevenRadioNetwork.run` is **fail-closed** unless `experimental=True` for regression research, because production parity is not established.

## Verified commands and status

Run locally with NumPy + PyTorch (the Warp execution tests skip if Warp is absent):

```bash
python3 -m pytest -q tests/test_v54*.py tests/test_v55*.py tests/test_v56_warp_gate.py
python3 reproduce_80step_reference.py
python3 src/train_v55_packet_cpu.py --help
```

Run independently on a machine with `warp-lang==1.18.0`, with a working CPU backend (event-kernel first compilation may take >1 min):

```bash
python3 tests/v56_network_topology_cases.py
# expect PASS two-hop relay, E2E ACK, no premature fusion, reset isolation
```

**Do not launch production training** until the following is green on CUDA Tesla T4:

```bash
python3 scripts/v56_cuda_preflight.py --device cuda:0 --envs 32
```

That preflight is deliberately a readiness diagnostic, **not** a trainer. CPU–Warp parity requires replaying the *same* topology, packets, packet loss and clock/RNG schedule through two implementations and comparing first event divergence. The current Warp HELLO advertises neighbor's current DV rather than timestamped packet snapshots (source-level mismatch with CPU), so no full parity claim is possible.

## Installation/branch integration

`bash stage_in_repo.sh /path/to/clean/v53/git/checkout` creates a separate `experiment/zhang-xue-v56-sevenradio-gate3-20261125` integration branch from v53; it refuses to overwrite existing files or alter a dirty checkout. It stages **but does not commit/push automatically**; review the staged files and run tests first. The original v53 branch and prior training snapshots remain unmodified.

## Critical scientific limitations

- **MAP:** 20x20 simulated target cells compressed to 10x10 in Actor; no proof this is optimal vs CNN/20x20.
- **DS/AoI:** `eta` is a declared scenario parameter; sample-level discounting and statistical calibration to actual Pd/Pf have not been demonstrated.
- **REWARD:** numerical weights and safety veto dynamics require ablation. AoI for never-delivered cells is *elapsed-time censored* by convention, not an unbiased statistical estimate.
- **PHYSICS:** acceleration-constrained action proposal exists, but unsafe-proposal rejection resets velocity instantaneously (not guaranteed physically feasible). Needs a validated acceleration-feasible shield before flight/safety claims.
- **MAC:** simplified packet PHY; distinct from UavNetSim full SimPy implementation, not an RF waveform model or IEEE 802.11 certification.
- **CUDA:** no Tesla T4 execution evidence; two 16GiB T4 memories cannot be pooled. Source memory is only a lower bound, excluding replay/optimizer/activations.

See `01_*.md`, `02_*.md`, `03_*.md`, `04_*.md` for precise data, caveats and paper source mapping. References include Zhang et al. IEEE T-IV (2024), DOI 10.1109/TIV.2024.3352581; Xue et al. Electronics (2026), DOI 10.3390/electronics15091875; the UavNetSim and NVIDIA Warp repositories.
