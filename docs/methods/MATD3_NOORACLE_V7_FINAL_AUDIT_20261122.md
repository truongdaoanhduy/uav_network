# No-oracle MATD3 v7 — last-pass integration audit, 2026-11-22

This note concerns **Bayesian ground-target search → cell REPORT → GCS packet
receipt** for 6 UAVs, 35 targets, 25 obstacles, horizon 1500 s and 1000m map.
The requested 3D-obstacle reconstruction and DS-based GCS belief fusion are
**not implemented** here. The historical `18d756ef` model weights are **not**
reused: the v7 observation/action semantics require new training.

## Source integrity

- Frozen `uav_network/algorithms/matd3.py` SHA256:
  `e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa`
- Frozen `uav_network/algorithms/masac.py` SHA256:
  `7e5fd9cb00c664bbea88321dcfdd3d0b75527c992b041a1a42434162a92796ce`
- Both match the reference-source pin test without patching the original
  model source. Extra integrity fixes are confined to checkpoint, training
  observer interoperability, stand-alone probe and corresponding tests.

## Verified no-oracle flow

1. `uav_network/environment/gpu/bayesian.py` implements Warp camera evidence
   and posterior updates. Positive/negative sensor outcomes have physically
   different likelihoods at different altitudes. No truth-target ID enters
   decentralized Actor observation or radio-report admission.
2. `uav_network/environment/gpu/vector.py::step_async` launches native Warp
   motion, `sense_bayesian`, cell confirmation, `sense_reports`,
   `packet_step`, `end_step`, and new observations/state.
3. `uav_network/communication/gpu/mac.py::packet_step` is a true `@wp.kernel`
   implementation of CSMA/CA packet events. Upstream UavNetSim uses Python
   and **SimPy** on CPU (`third_party/UavNetSim/simulator/simulator.py`,
   `third_party/UavNetSim/mac/csma_ca.py`); it is not itself a Warp library.
   The Warp network is a reimplementation compared to the SimPy reference by
   native parity tests, not a claim of byte-identical network internals.
4. Truth-rated training/evaluation scores in `nooracle_telemetry.py` and
   `gpu_mission_metrics.py` are observational metrics only: policy outputs do
   not get to access the true target coordinates through those scorers.
5. MATD3 E4096/U4/B8192/Replay1M/75%-CPU-25%-GPU/BN256×512/seed44 and
   immutable Actor snapshots with fixed 100 validation seeds follow the
   original frozen `18d756ef` **training mechanics**, not old MDP model weights.
6. A held-out final test requires the **actual 20,000-episode budget** and
   complete deterministic validation/selection. Nothing here claims remote
   execution success merely because local CPU smoke tests passed.

## Bugs corrected in this audit pass

- Best-checkpoint restore incorrectly rejected valid provenance because the
  verifier omitted canonical fingerprint encodings used by the writer. Its
  expected hashes now include `config_fingerprint(config)`.
- Training failed when a minimal checkpoint-only tracker lacked the optional
  `step_reward` API. Dense step journals are enabled only when supported;
  standard W&B `OnlineTracker` still receives full step data.
- The external MASAC AMP memory probe emitted **scaled** gradient telemetry
  while the production gradient probe emitted unscaled values. A context-local
  read-only scaler corrects the diagnostic values without changing the pinned
  MASAC source, actual gradients or optimizer steps.
- Test expectations for archive storage, vectorized critic hook counts, sparse
  MASAC `valid_routes` target Q, and diagnostic per-chunk statistical samples
  were outdated. Updated tests still strictly compare learner weights,
  optimizer state, deterministic RNG, non-diagnostic metrics and physical chunk
  bounds; they no longer mistake hook count or stochastic sampled telemetry
  for actor/critic numerical mismatches.

## Repeated checks, with exact scope

- 3× `test_gpu_packets.py`, `test_propagation.py`, the no-oracle cell-report and
  evaluator integrity test files: **68 pass + 10 subtests per repeat**, CPU Warp.
- 3× real native CPU Warp mission horizon, 2 worlds/repeat, 1500 max steps:
  all six world terminal results passed finite reward/state, energy, packet,
  delivery and completion invariants. The search policy was randomized.
- 3× local MATD3 actual training optimizer (4 episodes, 2 envs, horizon 20,
  U4, H256×512), followed by 100-seed grouped validation on initial/final
  Actor snapshots: **132 optimizer updates and 100 validation worlds per
  policy per repeat**, smoke-level only.
- Two training repeats with seed44 returned **bitwise-identical Actor tensor
  hashes**; seed45 returned a different hash.
- Source signatures compile (`python -m compileall`), `git diff --check` pass.

Final clean, end-to-end functional pytest command and observed completion:

```bash
python -m pytest -q tests --ignore=tests/test_project_structure.py --disable-warnings
# 1602 passed, 105 skipped, 8 warnings, 1111 subtests passed
# completed in 419.85s; process exit 0
```

This was a new pytest process started after functional fixes were complete,
not the earlier mixed-source exploratory run. No failing functional tests were
observed. The explicit **separate** project-structure suite remains red on
7 inherited file-length violations; its frozen-algorithm checksum subtest is
passing. CPU test success must not be represented as T4 CUDA certification.

## Evidence and known limits

- Ubuntu access in this run has **no CUDA driver** available to Warp or
  PyTorch. All native Warp stress tests ran on **CPU**, not T4 CUDA kernels.
- The historical `tests/test_project_structure.py` rule demands <=500 lines
  for each owned source/document except a checksum-pinned MATD3 model.
    Seven pre-existing files exceed this limit: configuration, training,
  deferred GPU evaluation, tracking, cause analysis, `test_tracking.py`,
  and `CURRENT_TASK.md`. **This style gate is not green**;
  its failures should not be misrepresented as passing functional tests.
- Estimated static replay storage for full production E4096 is 12.39 GiB CPU
  + 4.13 GiB GPU. E4096 native Warp WorldState alone is approximately
  3.88 GiB on the collector GPU; actual peaks include transient buffers,
  PyTorch optimizers, and model activations, which require real T4×2 profiling.
- A successful short smoke test does **not** demonstrate trained policy
  convergence, search accuracy, GCS delivery success or Final Test4000 scores.
- The Kaggle API quota verification/remote launch was blocked by a local SDK
  `kaggle`/`kagglesdk` version mismatch; use the generated self-contained
  Kaggle UI notebook to launch when GPU access and Secrets are available.
