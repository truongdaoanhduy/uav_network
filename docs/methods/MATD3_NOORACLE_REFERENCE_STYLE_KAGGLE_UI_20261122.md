# Kaggle UI notebook matching the reference run `18d756ef`

**Reference**: `/home/aduy/Downloads/M2_EXACT_REFERENCE_18d756ef_20261121/kaggle_notebook/notebook5051433b29.ipynb`.

The corrected notebook uses the **same three visible cell types, in the same order**, as the original: introductory experiment markdown; editable `ALGORITHM`, `NOTEBOOK_REF`, `UPDATES_PER_VECTOR_STEP`, `TRAIN_SEED`, validation/GPU batch limits, T4×2 preflight and Kaggle Secrets checks; then the checksum-verified embedded source/bootstrap with full MATD3 Training → Validation100 → best checkpoint → independent Test4000 and optional W&B dual-axis view.

Changes versus the historical run are explicit: the task environment now uses the no-oracle Bayesian cell report ID rather than reading true target IDs for report generation; the Actor is observation-v7 and cannot reuse the old weights. GCS DS-belief fusion and 3D obstacle reconstruction have **not** been added. The fixed-seed protocol remains 20,000 training episodes, Actor BatchNorm/Plain Critic [256,512], seed44, E4096/U4/B8192, snapshot every 100 vector steps, validation100 and final Test4000. `CHECKPOINTS_PER_GPU_BATCH` is conservatively 2 with 256 max evaluation worlds/GPU rather than the original 81/8192; these are **evaluation throughput settings only**, selected to lower the chance of T4 memory exhaustion. This has **not yet been certified on T4×2**.

The Kaggle run identity is generated at runtime, as in the original notebook. `NOTEBOOK_REF` is obtained from `KAGGLE_KERNEL_REF` when Kaggle supplies it; if the first preflight cell raises an error, replace `YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG` with your notebook URL suffix. In Kaggle select **T4×2**, Internet On, provide `WANDB_API_KEY` and `HF_TOKEN` as notebook Secrets, then **Save & Run All**.

The reference's custom source patch and older oracle-assisted simulation **are not transplanted**: using them would change the new scientific scenario. The code in the bootstrap is an independently hashed bundle from the corrected repository. No credentials are embedded in notebook cells. No new GPU run, quota check, or training convergence is claimed by merely constructing this notebook.

Repeatable validation:

```bash
python -m pytest -q tests/test_matd3_reference_style_kaggle_ui.py tests/test_18d756ef_fullflow_integrity_20261122.py
# 7 passed on the connected Ubuntu sandbox
```
