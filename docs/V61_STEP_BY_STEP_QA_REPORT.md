# v6.1 research code: step-by-step independent verification

Date: 2026-11-25. Source branch:
`experiment/zhang-xue-v61-full-gates-20261125`.
Immutable baseline v6.0 SHA:
`aa05bab4ed7f4f966004a7b5ef37354dd13f8d72`.

The separately uploaded v5.7 ZIP is **historical comparison material**,
not merged back into this independent new scenario.

**Recorded local gates:** 112 pytest PASS plus 50 unittest subtests;
Warp 1.18.0 CPU 2-hop relay/E2E ACK PASS; source governance full-tree
PASS; 4-step learner CPU integration PASS for both algorithms.

## Gate 0 — Immutable source and old code removal

- Keep v6.0, v5.9, main unchanged; work in a dedicated Git branch.
- Root old `configs/`, legacy world environment, old asynchronous learner,
  historical notebook and vendor source remain absent.
- Active configuration: `research/zhang_xue/configs/scenario_7radio.yaml`.
- Only MATD3, MASAC, their Vanilla implementations, generic offline
  statistics/evaluation/plots and 7-radio research environment retained.

## Gate 1 — Scenario and configuration validation (fixed)

**RED observed:** before the patch, at least 16 invalid configuration
variants were silently accepted, including impossible battery/velocity
parameters, nonfinite rewards, and mistyped safety/observation modes.
Also, the obsolete `hard_veto` option and old `v54` preset were selectable.

**GREEN:** all scientifically critical scalar values are checked as finite,
strictly positive where appropriate, and mode strings must match known
implementations. The only supported motion mode is
`kinematic_shield`; obsolete immediate-stop veto branch removed.
The command-line preset is `current`; old `v54/v57` profiles are rejected.
Configuration remains 1 km, 15 static targets, 20×20 belief cells, six
UAVs, one independent GCS and five fixed cylindrical obstacles.

This is a *different research benchmark* from earlier 3 km/35-target/25-
obstacle settings; numbers cannot be mixed in published comparisons.

## Gate 2 — Target map, sensing, belief, reward and physical feasibility

- Finite bounded target-cell placement and seed-based reset have regression
  checks; infeasible placements fail rather than loop indefinitely.
- Sensor Pd/Pf and 3D building occlusion are represented, with an optional
  source-count DS calculation and support for the chosen belief features.
- Movement enforces acceleration constraints and checks swept trajectories.
  An infeasible initial relative-velocity configuration **cannot be
  guaranteed collision-free by any admissible action**; the engine explicitly
  flags a safety failure/penalty rather than falsely certifying safety.
- Reward and energy tests are regressions, *not* paper-matched ablations.
  Fixed transmitter power remains 0.5 W in the current benchmark.

**Unresolved scientific conflict:** network packets carry cumulative
per-source cell counts and one freshest generation timestamp. Discounting
that cumulative snapshot cannot be claimed to implement **per-observation**
exponential AoI weighting in Xue et al. (2026), Eq. 9. The pure
`ewma_effective_counts_update` mathematical helper does not by itself
establish end-to-end MAC/ledger application. Correct realization needs
versioned per-source age-weighted ledger or individually timestamped
observations, plus packet encoding/ACK watermark changes on both CPU/Warp.

The DS reliability cap 0.8 is a declared empirical heuristic; it is not
a proven Bayesian calibration.

## Gate 3 — Seven-radio MAC and network semantics

- Seven separately represented radios (6 UAV + GCS), real CPU discrete-event
  CSMA/CA, per-radio queue and route status, packet custody, ACK+E2E ACK.
- Actual Warp CPU two-hop topology: UAV0 -> UAV1 -> GCS, GCS receives DATA
  and returns E2E ACK, with separate reset isolation. This tests **CPU
  backend**, not GPU physics or real hardware.
- The custom discrete-event radio simulator is **not a bit-for-bit port** of
  UavNetSim, and changing its PHY or timing is an explicit scientific
  experiment. Full packet-event parity and time-varying mobility during the
  same one-second radio interval remain unverified.

## Gate 4 — Actor / Critic and MATD3 / MASAC (fixed)

**RED observed:** v6.0 environment returned `obs(6,514)` and
`mask(6,7)`, but the retained learner expects the destination mask at the
*end* of its observation, and requires an alive flag at input feature #9.
Previously, raw feature #9 was GCS distance, so learner interpretations
were wrong despite isolated learner unit tests passing.

**GREEN adapter:**
- 514 original local features + 1 preserved GCS distance + 7 destination
  bits = **522 actor features per UAV**.
- Critic state uses 6 × 522 = **3132 joint local-policy features**,
  without leaking true target occupancy to the Actor.
- The learner's continuous head remains `[XY, RF]`; XY controls are
  constrained to a unit disk and scaled to m/s. The RF head is ignored
  because this scenario currently uses fixed TX power.
- Illegal route choice fails closed.
- Both algorithms executed **four real CPU environment transitions**
  with seed 44, built minibatches from those transitions, ran real
  backward/optimizer steps, and reported finite metrics.
- MATD3 actor remains untouched on first critic update by design
  (`policy_delay=2`); MASAC actor updates on its first eligible step.

This is a tiny CPU wiring proof, **not convergence**, not full training,
and not online W&B or Hugging Face logging.

## Gate 5 — Source governance, packaging and reproducibility

- Whole-tree and changed-source 500-line rules and syntax checks.
- `setup.sh` validates imports for independently installable learner and
  scenario packages; generated maps require no external dataset download.
- SHA-256 source manifest, immutable Git SHA, tests, CPU scripts and
  this report are packaged together.
- All old notebooks, credentials, model weights, caches, HF/W&B data and
  generated binary outputs excluded from the new code ZIP.

## Gate 6 — CUDA memory and simulation throughput: NOT VERIFIED

The code currently allocates **635,931 bytes/environment** for the Warp
radio SoA (not the earlier proposed 315,710-B compressed layout).
At 2,048 environments per GPU and replay 300,000 / obs 522, the
**planning-only** total is about 7.799 GiB/GPU including an *assumed*
3 GiB PyTorch/CUDA reserve. No T4 peak VRAM, real kernel wall time,
GPU dual-hop event parity or 4096-env MATD3/MASAC training was measured.
The sandbox has Warp CPU but no NVIDIA driver.

Do not call any paper reproduction, Q1/A* result, CUDA performance
or scientific convergence verified until the separate gates have passed.

## Scientific reference alignment

- Zhang et al., *IEEE Transactions on Intelligent Vehicles* 9(2), 2024,
  DOI 10.1109/TIV.2024.3352581: evidential target / empty / unknown maps,
  cooperative reconnaissance and DCDDPG. Our MATD3/MASAC models are
  NOT identical to Zhang's DCDDPG (central + local critic architecture).
- Xue et al., *Electronics* 15(9), 1875, 2026,
  DOI 10.3390/electronics15091875: AoI from timestamped buffered
  observations, with freshness discount before DS fusion. We have
  not proved the same end-to-end mechanism.
- UavNetSim reference implementation:
  https://github.com/Zihao-Felix-Zhou/UavNetSim
- Warp: https://nvidia.github.io/warp/

## Reproduce the verified local gates

```bash
bash setup.sh --dry-run
PYTHON=/path/to/installed/python bash setup.sh --check-only
PYTHONPATH=.:research/zhang_xue/src python -m pytest -q \
  tests/ research/zhang_xue/tests/
PYTHONPATH=.:research/zhang_xue/src python \
  research/zhang_xue/scripts/v61_cpu_actor_learning_gate.py --seed 44 --transitions 4
cd research/zhang_xue
PYTHONPATH=src python tests/v56_network_topology_cases.py --device cpu
```

## Acceptance and remaining blockers

All passed checks are local CPU/source gates. CUDA, per-observation AoI
through the radio stack, real UavNetSim parity and paper-scale MARL
training remain explicit **FAIL/BLOCKED or NOT IMPLEMENTED**, not silently
passed tests.
