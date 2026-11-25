# Repository engineering contract — source of truth (2026-11-25)

This policy **reinstates** the 2026-10-12 owner constraints from
[implementation-plan.md](implementation-plan.md), [design.md](design.md),
[README](../README.md), and `tests/test_project_structure.py`.
It is not a replacement for scientific scenario specifications.

## 1. Source structure and naming

- Keep each owned Python, YAML, shell, Markdown, TOML and text source at **500 lines or fewer**.
- Package source by responsibility. New Zhang-Xue code goes under
  `research/zhang_xue_v58_20261125/src/zhang_xue/{communication,environment,
  mobility,belief,gpu,configuration}/`.
- Compatibility filenames `uav_v54_*.py` and `v56_warp_*.py` remain as tiny
  **import aliases** so existing scripts/checkpoints can still load code.
- Add new components to the named package, never create additional flat
  `src/new_version.py` files.
- Avoid "magic" scientifically adjustable constants: expose them via checked
  `EnvConfigV54`, YAML, CLI overrides or explicitly label a GPU compile-time
  constant with the reason it is not dynamically supported.
- Do not claim study-3000/35T results from the 20x20, 1-km 15T benchmark.
  Unsupported map geometry must fail closed rather than silently rescale.

## 2. Research and physical correctness

- Treat truth as environment-private; Actor receives only local belief/neighbor
  information permitted by decentralized execution; global state is for critic.
- Preserve packet custody on loss, TTL/drop semantics, radio energy, independent
  GCS budget, proper end-to-end ACK and source-provenance deduplication.
- Bound physical movement and check swept collisions. Flag impossible initial
  states and **never call an acceleration-constrained collision detector a
  certified collision-prevention system**.
- Preserve independent RNG streams/seeds, resolved config, pinned code and
  evaluation evidence for paper comparisons.
- Verify CPU vs NVIDIA Warp events and T4 peak memory/throughput before
  production GPU training. Network smoke tests are not an end-to-end GPU trainer.

## 3. Setup, outputs and Git hygiene

- The repository root `setup.sh` creates the virtual environment and uses
  pinned sparse-checkout of UavNetSim. Zhang-Xue `setup.sh` delegates to that.
- Generate seed-dependent maps at reset. No target dataset download is required
  for this benchmark; do not invent a dataset URL.
- Keep `outputs/`, `.venv/`, `wandb/`, downloaded `third_party/`, private
  credentials and checkpoints out of Git. Place large experiment artifacts on
  external storage with immutable SHA256/URL manifests.
- On the v5.9 branch, the owner explicitly authorized retiring 25 historical
  tracked notebooks (~19.5 MiB) from HEAD. The full set remains recoverable
  at the immutable v5.8 Git commit. See
  `docs/benchmarks/NOTEBOOK_ARCHIVE_V59_20261125.md`. This HEAD-only removal
  does not erase those bytes from previous Git objects or main.

## 4. Red/green gates

1. **Red:** first reproduce with focused unit test, retain known numerical
   baseline and minimal input.
2. **Green:** implement minimal source-compatible correction and pass full
   modified-component regression.
3. **Architecture:** `python scripts/check_project_policy.py --scope changed
   --base <base_commit>`; filename, directory, source-length, syntax, token
   heuristic and new large-asset checks must pass.
4. **Install/config:** validate `bash -n setup.sh`, YAML validation,
   `--set` overrides, impossibility rejection and outputs containment.
5. **Runtime:** CPU packet step, Warp device smoke (when installed), event-level
   parity against same seed and actual T4 gate (when CUDA available).
6. **Delivery:** commit in isolated branch, push, read remote SHA, update
   `docs/superpowers/CURRENT_TASK.md` index and preserve history.

## 5. Previously existing debt — not hidden

The repository source restored on 2026-11-24 already contains large files
outside Zhang-Xue (including pinned training algorithm source, sizeable
experiment scripts and project notes). The strict full-tree policy
`--scope full` currently **fails**. The default CI enforces all new/modified
sources while maintaining a documented unresolved baseline; it does **not**
advertise full-tree compliance. Split legacy modules in separate validated
changes, except immutable author/run snapshots whose hashes must be preserved.

## 6. Keep a meaningful review signal

`tests/test_project_structure.py` is an older full-tree guard, not replaced
by the new diff-scoped guard. Its known failure is part of the backlog and
must not be deleted or marked xfail merely to make CI green. A clean research
module suite and green diff gate do **not** certify legacy code.
