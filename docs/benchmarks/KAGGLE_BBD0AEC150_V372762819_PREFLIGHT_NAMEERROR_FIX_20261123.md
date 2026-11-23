# Kaggle `longlonthiu/notebookbbd0aec150`: runtime NameError in source preflight

**User-reported URL:** https://www.kaggle.com/code/longlonthiu/notebookbbd0aec150/log?scriptVersionId=372762819

## Evidence

Read-only downloaded Kaggle notebook log to `/home/aduy/Downloads/kaggle_bug_bbd0aec150_latest/notebookbbd0aec150.log` using `kaggle kernels output longlonthiu/notebookbbd0aec150` (Kaggle CLI latest-session log). It reports, at about 49.885s:

```
papermill.exceptions.PapermillExecutionError
Exception encountered at "In [2]"
NameError: name 'CONFIG' is not defined
  108 subprocess.run([str(python),'-c',
  109   'from uav_network.configuration import load_config; ...'
> 110   'c=load_config('+repr(CONFIG)+'); '
```

**Root cause:** `CONFIG` exists only as a local constant in the *notebook builder* module. The resulting three-cell notebook has `NOTEBOOK_REF` and training parameters in cell2 and `SPEC['config']` in cell3; it does **not** create a variable named `CONFIG`. When the embedded bootstrap entered its science config preflight, Python evaluated `repr(CONFIG)` and raised immediately. The source tar extracted and the Kaggle pip install phase ran first; pip's dependency-conflict warnings are *not* the fatal traceback for this notebook execution.

## Fix

Use the actual Kaggle runtime job object as the single source of truth:

```python
'c=load_config('+repr(SPEC["config"])+'); '
```

instead of the old code:

```python
'c=load_config('+repr(CONFIG)+'); '
```

The fix is in `scripts/build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123.py`. It changes **only bootstrap source string generation**, not Matd3/Bayesian/env/PHY, the retained source tar or the science YAML. User's requested settings remain Posterior>=.99 with positive camera YES, no LR gate or fixed 50m confirmation restriction, TX learned max0.8W, MAP800, E4096/U4/B8192 H256×256 orthogonal `torch.compile`, 20k episode design, Val100 for each snapshot, two-GPU test1000 and HF/W&B deep forensic logs.

## Regression proof

1. **TDD RED:** `tests/test_kaggle_bbd0aec150_v372762819_preflight_fix_20261123.py` executed real generated Kaggle cell3 preflight with only runtime variables. The test failed with the **same `NameError: name 'CONFIG' is not defined`** as the Kaggle log.
2. **TDD GREEN:** after switching to `SPEC["config"]`, isolated bootstrap preflight invokes a real subprocess to import the actual new science config and assert posterior-only/version11/TX0.8; succeeds.
3. **End-to-end three-cell Kaggle bootstrap emulation:** executes cell2 and cell3 (real tar extraction and checksum, real science preflight subprocess) with test doubles for GPU availability/Secrets, pip and cloud training/Eval transports; reaches the intended `uav_network.cloud.remote` and `uav_network.experiments.one_session` calls, checking Eval8192 and Test500/GPU arguments. No genuine API tokens used; no GPU training performed.
4. **Real compiled CPU MATD3:** extract the EXACT new source from the embedded tar, real Warp+Inductor optimizer training, checkpoint, heldout CPU Eval. No code download required by notebook.
5. **Deep logging test:** real CPU compiled MATD3+Warp train with fake W&B/HF network transports verifies TX/RX telemetry, root-cause minibatches and recoverable diagnostic archive. The source SHA-256 remains identical to the previous working science bundle.
6. **46 selected tests passed**, no test failures, in the re-run following the fix.
7. New `.ipynb` file length **738,764 bytes**, below strict 1,000,000-byte Kaggle kernel source limit. All 3 cells retained. Note: actual Kaggle GPU/save-run was not executed in this local validation; the user must import and run the fixed notebook to confirm hardware and end-to-end hosted execution.

## File

`notebooks/MATD3-POSTERIORONLY-MAP800-TX08-T4X2-FULLFLOW-UNDER1MB-20261123.ipynb`

Also copied to Ubuntu Downloads under same name and a distinct `...-FIX-NAMEERROR-372762819.ipynb` filename, to avoid accidentally selecting the failing prior file.
