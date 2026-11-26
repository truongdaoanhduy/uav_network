# ACTIVE V88 SCIENTIFIC SCOPE — READ FIRST

**Active and reproducible scenario:** 1,000 × 1,000 m, 6 UAVs, 15 static targets, 20 × 20 belief grid, fixed 50 m altitude and 900-step horizon. The active source of truth is `research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml` together with `scripts/gpu/run_v81_real_rollout.py`; see [`docs/ACTIVE_SCENARIO_V88.md`](docs/ACTIVE_SCENARIO_V88.md).

All experiment descriptions below are historical reference material only; none overrides this active scenario, and no historical GPU notebook defines the current run. New GPU work is on Vast.ai only.

---

# V73 — Direct 4096-env / batch-8192 capacity diagnostics

The current target is `configs/learners/e4096_b8192.yaml`. There is no required
environment-count ramp. Small functional tests report their effective dimensions
separately. The new runner writes flushed phase logs, host RAM, device-wide VRAM
snapshots and Torch allocation peaks before/after risky work.

```bash
python scripts/gpu/run_capacity_probe.py --mode functional \
  --network-device cpu --learner-device cpu --output-dir outputs/functional_matd3
python scripts/gpu/run_capacity_probe.py --mode capacity --algorithm matd3 \
  --network-device cuda:0 --learner-device cuda:1 --output-dir outputs/capacity_matd3
```

Capacity mode runs the Warp **radio subsystem** at the exact configured size,
fills the existing replay using four disclosed real CPU transitions repeated for
memory commitment, and performs two real learner updates. It cannot establish
4096 full GPU environments, learning convergence or simulator event parity.
See [Vietnamese V73 status and reproducible commands](docs/V73_CAPACITY_DIAGNOSTICS_VI.md).

Historical V72 release follows. Its account numbering/token observation predates
the corrected active credential file; current account2 is active after renumbering.

---

# V72 — Reproducible scenario/packet corrections (Warp CPU verified)

Current audit: [Vietnamese findings and fixes](docs/V72_AUDIT_AND_FIXES_VI.md),
[primary-source comparison and GPU roadmap](docs/V72_RESEARCH_AND_GPU_PLAN_VI.md),
[issue register](docs/V72_BUG_STATUS.json).

The uploaded V71 source was independently rerun before editing: 242 pytest tests
and 64 subtests passed. V72 adds real environment, packet and Warp CPU execution
regressions. Final results and raw RED/GREEN logs are in `outputs/qa_evidence_v72/`.

```bash
bash setup.sh --install
source .venv/bin/activate
# Optional Warp kernels, needed for the execution gates:
python -m pip install warp-lang==1.18.0
python -m pytest -q tests/ research/zhang_xue/tests/
python scripts/gpu/warp_packet_cuda_gate.py --device cpu --envs 4
python scripts/gpu/packet_contract_gate.py --device cpu
python scripts/maintenance/run_qa_bundle.py --output-dir outputs/new_qa --seed 44
```

CUDA/T4 execution, complete CPU–Warp event parity and full GPU RL training remain
unverified. The matched CPU/Warp fixture proves original evidence reaches GCS,
not simulator timing fidelity. Generation-only Warp watermarks support EWMA mode
with strictly increasing source/cell timestamps; violations fail closed. Main
compact3 peer selection now uses local, unexpired HELLO routes by default.

`run_kaggle_v72_gates.py` runs bounded per-device kernel/packet tests on an already
configured Kaggle runtime; it performs no login, public upload or training.
The authorized account2 token was inactive when checked; no GPU job was submitted.

The older release descriptions below are historical, not the current status.

---

# v7.0 — Independent packet-radio audit, RED→GREEN fixes, CUDA fail-closed gates

**Current release:** [full Vietnamese issue audit](docs/V70_INDEPENDENT_AUDIT_VI.md),
[issue register](docs/V70_BUG_STATUS.json),
[CUDA/Kaggle validation guide](docs/V70_CUDA_DEPLOYMENT_VI.md).

```bash
bash setup.sh --check-only
bash setup.sh --qa --seed 44
python scripts/maintenance/paired_scenario_gate.py
python scripts/maintenance/network_memory_report.py --output outputs/qa_evidence/memory_estimates.json
bash setup.sh --gpu-packet-gate   # 3=BLOCKED on CPU; attempts *network-only* CUDA smoke
```

The v7.0 package retains all source, old research revisions and copied original v6.9
QA artifacts while adding a fresh reproducible `outputs/qa_evidence/` gate log.
Warp v6.7 radio network has not been integrated into parallel full RL rollout,
reward, replay, optimizer and evaluation. **No 2×T4 / 4096-env training success
or CPU-vs-Warp packet-event parity is claimed.**

---

# v6.8 — Full-source + reproducible academic QA logs

**This release includes source code AND raw test artifacts** under `outputs/qa_evidence/` (JUnit XML, command logs, individual exit codes, RED→GREEN evidence, CPU learner smoke, map sweep, environment metadata and GPU blocker report). The former v6.7 ZIP had only source, tests and prose audit; it had no raw QA logs.

```bash
bash setup.sh --check-only
bash setup.sh --qa --seed 44
cat outputs/qa_evidence/SUMMARY.md
```

**What actually passed:** CPU tests and 4-transition MATD3/MASAC smoke. **What did not pass:** complete UavNetSim packet-event parity, Warp data-mule packet step, T4 4096-env throughput/VRAM, and convergence evaluation. `--gpu-gate` intentionally exits 3 (BLOCKED) until real kernels and parity tests exist.

See [v6.8 scientific & code audit](docs/V68_AUDIT_VI.md) and [machine-readable issue register](docs/V68_BUG_STATUS.json). All source is under named folders and the QA artifacts live in `outputs/` to keep Git history clean.

---

# UAV target-search — v6.7 source-code release (CPU validated, GPU BLOCKED)

**Latest scientific corrections:** [v6.7 audit and source-level evidence](docs/V67_PHY_SAFETY_TERMINATION_AUDIT_VI.md).
**Use:** `bash setup.sh --test`, `bash setup.sh --cpu-gate-v67 --seed 44`.
**Matched ablations:** `scenario_v67_{datamule,no_peer}_{1km,2km}.yaml` in `research/zhang_xue/configs/`; `battery_termination_mode: all_depleted` distinguishes v6.7 mission semantics from the previous default. CCA governs *MAC sensing* rather than physical packet decode; CPU test cases include power below CCA but SINR above threshold. Do **not** claim UavNetSim event parity, Warp full packet support, CUDA training, or converged-policy gains.

## Previous v6.6 audit retained below for traceability

# UAV target-search research — v6.6 independently audited release (CPU only)

**Current scientific audit:** [docs/V66_INDEPENDENT_SCIENTIFIC_AUDIT_VI.md](docs/V66_INDEPENDENT_SCIENTIFIC_AUDIT_VI.md).
**Source provenance:** `SOURCE_SHA256_MANIFEST.json` and `research/zhang_xue/SHA256_MANIFEST.json`.

This archive keeps the modularized project-wide MATD3/MASAC core and the v6.5
**CPU-only** seven-radio / data-mule packet experiment. It preserves the historical
source and does not claim NVIDIA Warp packet-event parity or full training.
`research/zhang_xue/configs/scenario_7radio.yaml` is a **historical 6.4 smoke profile**
(`w_p2p=0.10`, data-mule off), **not** the main v6.6 research baseline.

## Reproducible checks

```bash
bash setup.sh --dry-run
bash setup.sh --check-only
bash setup.sh --prepare-data --seed 44 --episodes 4
bash setup.sh --test
bash setup.sh --cpu-gate           # old 6.4 algorithm CPU regression
bash setup.sh --cpu-gate-datamule  # v6.6 1km data-mule, MATD3 + MASAC 4-transition gates
bash setup.sh --gpu-gate           # BLOCKED deliberately (exit=3), NOT a T4 result
python scripts/maintenance/academic_scenario_audit.py --sizes-m 1000 2000 3000
python scripts/maintenance/verify_source_manifests.py --check
```

## Pre-registered communication ablation profiles

- `scenario_v66_datamule_1km.yaml` vs `scenario_v66_no_peer_1km.yaml`
- `scenario_v66_datamule_2km.yaml` vs `scenario_v66_no_peer_2km.yaml`

Each pair differs **only** in `enable_peer_dissemination`, retains
`enable_data_mule_forwarding=True` and `w_p2p=0.0`. 2km uses 40x40 cells
and 60 targets to keep nominal target density fixed (15 per km²). This
is an ablation specification, **not** a claimed converged/trained comparison.
Read the audit for other non-invariant geometry/radio confounders.

`P_tx=0.1 W`, `CCA=-82 dBm`, `noise=4.389577e-13 W`, 22MHz and 2Mbps
match selected **UavNetSim configuration parameters**, not its complete
PHY/MAC, SimPy, Sionna RT or routing implementation.

`WarpCompressedRadioNetwork` only allocates GPU buffers; it has **no**
packet-step method. Its 379,848-byte/env estimate omits the CPU data-mule
4D acknowledgment watermarks (806,400 additional bytes/env at 6/20x20),
packet kernels, and training temporaries. Full Warp training and zero-shot
N-UAV generalization remain unverified.

---

# UAV search target — independent Zhang-Xue seven-radio scenario

## v6.3 — project-wide folder structure & reproducible CPU gates (2026-11-25)

**Start here:** [V63 independent audit (Vietnamese)](docs/V63_INDEPENDENT_AUDIT_VI.md)
& [directory/config contract](docs/V63_DIRECTORY_CONTRACT.md).

This version reorganizes MATD3, MASAC, baselines and shared replay/gradient
code into separate nested packages. It also moves all historical flat research
implementations under named `zhang_xue/{gpu/kernels,learners,compat}` folders,
adds strict learner YAML/CLI, one `setup.sh` for ALL packages/data/tests, and a
working 4-step CPU Actor→RF/XY/route→packet environment→optimizer gate.

```bash
bash setup.sh --dry-run
bash setup.sh --check-only
bash setup.sh --prepare-data --seed 44 --episodes 4
bash setup.sh --test
bash setup.sh --cpu-gate
```

**Scientific restrictions:** learned TX, EWMA and half-step PHY are tested on
CPU only. The compressed v6.2 Warp memory layout has **not** been activated;
actual GPU allocation remains the v6.1 SoA. No full T4 training, production
W&B/Hugging Face logging or UavNetSim event parity was measured in this ZIP.


This branch contains ONE current research scenario: six UAVs, one separate GCS, 20x20 Dempster–Shafer belief map and packet-level CSMA/CA. It does not reuse the earlier UAV training scenario, its configs or its asynchronous learner.

## v6.1 — source-correctness gates (2026-11-25)

- [Step-by-step v6.1 report](docs/V61_STEP_BY_STEP_QA_REPORT.md)
- Fixed scientific config validation, removed old hard-veto safety mode and
  retired obsolete CLI scenario presets.
- Integrated a source-faithful actor adapter: 514 local observation values +
  1 preserved distance + 7 route-mask bits = **522 policy inputs** per UAV.
- CPU gate proves four real environment transitions and one optimizer update
  each for MATD3 and MASAC, with no global target truth in Actor observations.
- **No CUDA T4 validation or full 4096-env training is claimed.** Per-reading
  AoI evidence fusion through the radio stack is not yet complete.

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

The CPU gate now connects a 522d Actor and separate route+RF controls to real network actions and a single optimizer update. This is not a full training pipeline or convergence run.

Online W&B and Hugging Face persistence from the old scenario are intentionally NOT carried over. They require a new integration only when a matching trainer/evaluator exists. No GPU T4 throughput or CPU–Warp parity is claimed.

Generated maps are seeded and created at reset; no external dataset is required. All outputs, checkpoints and logs belong under Git-ignored outputs/.
