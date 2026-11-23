# Kaggle notebookfe799af211 — scriptVersionId 372917422: investigation and scoped fix

**Date:** 2026-11-23. **User's exact notebook version:** https://www.kaggle.com/code/longlonthiu/notebookfe799af211/notebook?scriptVersionId=372917422 . Code source at prior release commit `7143240ecccb47168acf0cf8ced747f994315427`, **not verified that Kaggle actually uploaded that exact local file** (cannot pull the version). The user reported a Kaggle execution failure.

## Actual remote evidence & access limitations

1. Direct public page at `/notebook?scriptVersionId=372917422` and `/log?scriptVersionId=372917422` was fetched using JavaScript-capable web extraction. It returned a **Kaggle frontend failure**: `Something went wrong and this page crashed! Loading CSS chunk 1019 failed (https://www.kaggle.com/static/assets/KernelViewer.21f94db0016b9074.css)`. That error is **website rendering**, not a Python traceback from the Kaggle kernel. The page did not expose the session's kernel status, stdout, stderr or attached output files.
2. The Ubuntu Kaggle CLI binary exists, but `kaggle kernels status longlonthiu/notebookfe799af211` failed because its expected `~/.kaggle/kaggle.json` configuration was unavailable. No authenticated Kaggle kernel was pulled or inspected, and no user tokens were disclosed or modified.
3. Public Hugging Face `duy95/uav-search-diagnostics` `runs/` directory did not contain a folder named for the exact Kaggle notebook ref or `autotune`; this absence is not evidence that a random-run-id upload did not happen.
4. Therefore: **the actual root cause of execution failure in scriptVersionId 372917422 is still unknown**. Do not assert that the stale reference below explains the notebook's crash. A useful next evidence source is its version-specific Kaggle **Logs** tab, or a downloaded `.ipynb` with cell outputs, or an authenticated `kaggle kernels logs/pull` output where the version identity is verified. As of current Kaggle CLI versions, several commands may ignore explicit historical version tags; do not attribute the latest log to this scriptVersionId without checking metadata.

## Locally verified source defect, fixed without changing training science

In the shipped old full-flow `.ipynb` **cell2** there was:

```python
NOTEBOOK_REF = os.environ.get('KAGGLE_KERNEL_REF','longlonthiu/notebookbbd0aec150')
```

The new user notebook is **`longlonthiu/notebookfe799af211`**, so this default is stale. It flows into `SPEC['kernel_ref']`, `SPEC['account']`, `UAV_KAGGLE_KERNEL`, and the tracking/continuation provenance. Unless explicitly overridden by an environment variable, the notebook would identify itself as the older Kaggle notebook. This can corrupt run attribution or later workflows, though it is not proven to be the cause of the reported crash.

A new, separate version-only builder `scripts/build_map1000_tx06_autotune_fullflow_fe799_fix_20261123.py` now sets the correct `KAGGLE_KERNEL_REF` fallback to `longlonthiu/notebookfe799af211` and emits an informative `KAGGLE_TARGET_KERNEL_REF` line. Existing Kaggle notebooks and branches are not overwritten.

**Science and training exactly unchanged:** The entire Python cell3/source-archive payload is byte-for-byte identical to the prior full-flow builder, including its SHA256-pinned source archive. Map1000×1000m; 40×40 Bayes belief map, 25m cells; 6 UAV, 35 targets, 25 obstacles; TX power max0.6W, Posterior-only positive sensor+posterior≥0.99; UavNetSim PHY/MAC, all packet/false-confirm effects; MATD3 ActorBN H256x256, E4096/U4/B8192, seed44, 20,000 episodes; periodic Actors, Validation100 each, best from validation, heldout Test1000. Keep W&B/HF forensic and T4 autotune/guarded fallback.

## Tests & scope

- TDD introduced new tests for the exact Kaggle ref, complete 3-cell notebook structure, science SHA and previously required preflight settings.
- **Actual 3-cell Python startup executed locally**: generated source unpacked with real checksum; `.venv` and external cloud transports mocked; all scientific imports/config validations ran as actual CPU subprocesses; Kaggle T4 and Secrets provided only as synthetic fixtures; MAC/replay GPU benchmarks simulated with checked T4 timing and SHA values; actual `job.json` assembled and the train/one_session runner commands validated; completion report mocked. No Kaggle account accessed; no upload.
- Tests also verify arbitrary runtime `KAGGLE_KERNEL_REF` override is preserved. This fixes the stale fallback without hardcoding a different user account in new scenarios.
- The broad regression suite additionally includes genuine bounded CPU MATD3 train → Actor snapshot → Validation → best → heldout Test and pinned archive extraction tests. **A passing CPU/mocked test is not proof of real Kaggle T4×2 training success.**

## What to do on Kaggle

Import the new `MATD3-MAP1000-TX06-T4X2-AUTOTUNE-FIXFE799-V372917422-20261123.ipynb` into `longlonthiu/notebookfe799af211`, select two T4 GPUs, enable Internet, and grant its existing `WANDB_API_KEY`/`HF_TOKEN` Secrets. Save as a **new version** and run. No code auto-push to the user's Kaggle notebook has occurred from this analysis. This is a scoped reference/provenance fix, not a guaranteed repair for the inaccessible session error. If the new version also fails, obtain its **actual Logs last traceback** rather than replacing arbitrary code without evidence.
