# Kaggle one-megabyte notebook source limit — resolved 2026-11-23

User-reported Kaggle error: **"An error occurred while committing kernel: The kernel source must be less than 1 megabytes in size."**

- Old notebook `MATD3-NOORACLE-POSTERIORONLY-MAP800-TX08-E4096-U4-B8192-H256x256-ORTHO-COMPILE-VAL100-TEST1000-T4X2-FULLFLOW-20261123.ipynb`: **1,214,275 bytes**, rejected by Kaggle source size validator.
- New notebook `MATD3-POSTERIORONLY-MAP800-TX08-T4X2-FULLFLOW-UNDER1MB-20261123.ipynb`: **738,754 bytes**, <1,000,000 bytes, leaving ~261 KB of safety margin. New source code cell also <825 KB.
- Fix is in `scripts/build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123.py: minimal_runtime_source_archive`. Begin with the already vetted pinned UavNetSim/source tar; omit `tests/**` and unrelated `configs/**` entries only. **Retain** all `uav_network/**`, four cloud runtime scripts, pinned `third_party/UavNetSim/**` and the selected posterior-only/TX0.8 YAML. No module code or data paths were altered, and no Github/code network checkout is needed at Kaggle runtime.
- Compact tar uses gzip compression level9 and deterministic `mtime=0`. The exact embedded bytes are SHA-256 pinned in the notebook and verified before extraction on Kaggle. The notebook still has the exact 3-cell format of the user's reference `therwed/notebook5051433b29`.
- User's prior notebook reference defaults to `longlonthiu/notebookbbd0aec150` only if `KAGGLE_KERNEL_REF` is not set, avoiding using a hardcoded kernel URL as an environment variable name.
- **34 selected tests passed**, including real CPU compiled MATD3 from precisely the reduced embedded tar -> checkpoint -> held-out Eval; separate real CPU compiled MATD3 + Warp/HF-W&B fake transport verifying forensic journal output. The previous posterior-only selection, NoOracle privacy, double-GPU Eval layout, all-checkpoint review, and per-checkpoint100/finalTest1000 specs are unchanged.
- Kaggle **actual kernel commit with user's credentials and CUDA T4×2 execution not performed** in this local fix. File-size validation on the exact `.ipynb` is confirmed. Users must Import a **NEW** file: importing the old oversized notebook reproduces this error. Kaggle Save Version (Save & Run All) after selecting GPU T4×2, enabling Internet and granting Secrets WANDB_API_KEY and HF_TOKEN.

Never present this test as evidence of successful E4096 CUDA fit, GPU Eval8192/T4, or complete20k training; those are actual Kaggle runtime goals subject to platform quota and memory.
