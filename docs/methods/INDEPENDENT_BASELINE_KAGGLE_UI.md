# Independent baseline Kaggle UI notebooks

These two notebooks package the independent profile from the
[main-pair audit](MASAC_MATD3_MAIN_PAIR_AUDIT_2026-11-07.md).
They use independent actors and twin critics, plain MLPs, BASE and NOCHUNK.
This step changes export/bootstrap handling only; algorithm equations,
initialization, scenario, reward, battery and CBF remain as declared in
`configs/comparison-independent-baseline-xyz-sensing.yaml`.

## Build

From a clean committed checkout, after the pinned UavNetSim setup:

```bash
python -m pip install "nbformat>=5,<6"
python scripts/build_independent_baseline_ui_notebooks.py --seed 44
```

Defaults write to Downloads and outputs/kaggle. Repeat `--output-dir` for
other destinations. The builder needs nbformat, makes files only, and never
submits Kaggle jobs. Both notebooks embed one identical local source archive
and verify its SHA-256. No git clone runs inside Kaggle.

## Manual use

1. Create a separate Private Kaggle notebook for each file and import it.
2. Enable Internet and GPU T4 x2.
3. Grant WANDB_API_KEY and HF_TOKEN via Kaggle Secrets.
4. Set NOTEBOOK_REF to the notebook's real owner/slug.
5. Keep TRAIN_SEED=44 for this pair. For later training seeds, edit that
   variable in both corresponding notebooks.
6. Run preflight, then Save & Run All.

Names/tags reflect the actual algorithm, independent profile and edited seed.
Every fresh Kaggle execution gets a fresh run ID and lineage. MASAC declares ALPHA and
MASKAWARE; MATD3 declares DEFAULT initialization. Old SH/initialization
notebooks are unchanged. There is no warm-start, automatic continuation,
CPU visualization or GitHub Actions submission.

## Comparison protocol and limits

The profile trains 20,000 episodes. Compare periodic checkpoints at equal
environment transitions, because episode lengths can differ. Both use the
same five recurring validation layouts every 3.6864M transitions. These are
selection layouts; final paper comparisons need new paired layouts and
multiple training seeds.

The profile has the same 1500-step cap and existing all-UAV-depleted early
termination. Return remains the environment objective; mission_score,
unsafe proposals, CBF intervention and post-filter violations remain separate.
No export change is claimed to improve MASAC's ranking.

Schema/CPU tests do not verify T4 VRAM fit, CUDA throughput or convergence.
No full GPU training was submitted by this notebook preparation.

## Checkpoint compatibility correction

Whole-suite verification caught a legacy MASAC best-state identity that
explicitly retained the default log_alpha field while current canonical
identity omits that default. Restore now accepts both equivalent encodings
while still requiring the saved config, exact checkpoint digest and remaining
provenance to match. Switching to temperature_loss=alpha or changing actor
normalization remains rejected. The existing historical-state matrix now
checks that changing the temperature objective also fails. This affects
checkpoint compatibility, not policy optimization or scientific results.

Use a fresh Kaggle session for every subsequent training launch. Re-running
only the launch cell in an already-used session is unsupported: the inherited
runner refuses an existing source directory and also has fixed training output
locations. This prevents accidentally mixing a new run with old outputs.

## Verification record

The broad suite produced1356 passes,103 skips and1644 passed subtests;
4 failures were confined to the two source-structure guard test functions
(three file-length subtests plus stale algorithm hashes). No behavior tests
failed. Those guards were refreshed after an independent provenance review.
Blank-line-only source shortening preserves exact ASTs; history archiving
preserves exact concatenated bytes. The follow-up project-structure and
notebook suite passed13 tests and578 subtests. The original broad log and
exit status are preserved rather than relabelled as a new full-suite pass.

No GPU training or learning-rank improvement was verified by these checks.
