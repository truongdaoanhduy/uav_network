# 10 Kaggle UI notebooks — MATD3 BASE ×5 and MASAC BASE ×5

Requested 2026-11-08. These are **manual notebook files**, not submitted Kaggle jobs. Do not use PQL. The original reward code and all physics/environment/sensing/CBF definitions are left intact. MASAC is mask-aware (`mask_aware_discrete_entropy_target=true`), consistent with the active 35T25O profile; MATD3 uses its **default** initialization, not Kaiming/orthogonal.

## Experiment matrix (each algorithm has exactly these 5 cases)

| Case | Shared parameters across agents | Hidden Actor/Critic | Updates per vector collection | Actor LayerNorm | Critic LayerNorm |
| --- | --- | --- | ---: | --- | --- |
| 01 | Both Actor and Critic shared | 512 → 512 | 8 | Yes | No |
| 02 | Both Actor and Critic shared | 512 → 512 | 16 | Yes | No |
| 03 | Both Actor and Critic independent | 512 → 512 | 16 | Yes | No |
| 04 | Both Actor and Critic independent | 512 → 512 | 16 | No | No |
| 05 | Both Actor and Critic shared | 256 → 256 | 8 | Yes | No |

**Important**: shared actors/critics means parameters **shared between UAVs**. Actor and Critic are still separate networks, with decentralized Actor observations and centralized Critic state. LayerNorm is on Actor layers only, because the existing baselines do not implement Critic LayerNorm. Changing this would change the architecture instead of reproducing the requested five-case matrix.

For MATD3, `policy_delay=2` means Actor updates = half of Critic update calls (U8 → 4 Actor updates; U16 → 8). For MASAC the Actor updates every learner update (U8 → 8, U16 → 16), including entropy-temperature training. Each collector iteration takes **4,096 joint transitions**; U8/U16 is **not** 8/16 updates per individual new transition.

## Shared protocol and setup

Both algorithms: 6 UAVs; 1km map; 35 targets; 25 obstacles; up to 1,500 steps per episode; 20,000 completed episodes; seed 44; `deterministic=true`; `num_envs=4096`; replay capacity 1,000,000; batch 8,192; 75% CPU + 25% GPU replay; learning starts after 131,072 transitions; `critic_chunk_size=0` (no chunk); `vectorize_critics=false`; `output_normalization=false` for MATD3; AMP/compile/fused optimizers disabled; no PQL, no warm start. All reward, CBF, sensing, learned TX power and network parameters from `configs/comparison-independent-35t25o-h512-xyz-sensing.yaml` remain unchanged.

Save periodic **policy** checkpoints every 256 completed episodes (retain two local episode files; full-state recovery is scheduled separately), validation fixed seed bank of 5 episodes every 1,843,200 transitions, report metrics online to W&B project `truongdaoanhduy-uit/uav_search_target`, save chosen best policy to HF `duy95/uav-search-checkpoints`. Final on clean exit. **Training-only**; no automatic 30-episode final test, no CPU visualization. Kaggle runner may stop before 20,000 episodes due to a time limit; the baseline launch allowance is `timeout_seconds=39600` minus cleanup reserve, which overrides the profile's `max_train_seconds=41400` in actual Kaggle. No claim about T4 x2 memory sufficiency or convergence is made.

## Files and workflow

The generated **10 self-contained `.ipynb` files** and one `.zip` bundle are copied to `~/Downloads` (and `outputs/kaggle/`) by:

```bash
python scripts/build_matd3_masac_10_ui_notebooks.py --seed 44
```

No credentials are read into the notebook during build; the embedded archive is checked and source commit pinned. The notebook contains its own source bundle (no runtime `git clone`). **This makes source code readable to anyone with access to the notebook:** use Private notebooks on Kaggle, and do not upload these `.ipynb` files to a public GitHub repository.

For each notebook, open Kaggle and import the `.ipynb` into a new **Private** notebook. Set **Internet = On**, accelerator **GPU T4 x2**, grant Kaggle Secrets `WANDB_API_KEY` and `HF_TOKEN`. Edit the first code cell `NOTEBOOK_REF` to its real `username/notebook-slug` (do not reuse across notebooks). Keep seed 44, run **Save & Run All**. Each run uses a fresh random run ID and distinct output dir, logs W&B, uploads best policy to HF when validation improves. Do not reuse a finished kernel session for another run; do not assume an active Kaggle job is launched merely because a file was built.

## Reproduction and safety

- Source generator: `uav_network/cloud/matd3_ui_notebooks.py` (existing five MATD3 cases), `uav_network/cloud/masac_ui_notebooks.py` (five matching MASAC cases), `scripts/build_matd3_masac_10_ui_notebooks.py` (zip/exports).
- Tests: `tests/test_matd3_ui_ablation.py`, `tests/test_masac_ui_ablation.py`, `tests/test_active_checkpoint_cadence.py`, `tests/test_scaled_ui_notebooks.py`. Tests assert resolved `SPEC` config/seed/reward untouched, actor/critic sharing and architecture, independent target entropy, and selected CPU learner update smoke.
- Notebook pin: emitted `metadata.uav_ablation.source_commit` and `archive_sha256`; inspect these before launching if local project source changes.
- The MASAC independent/shared case architecture is identical in shape to MATD3; the algorithms themselves are **not** numerically equivalent, so an identical update count does not mean the same actor-update cadence.
- **Do not merge the embedded notebook payload into public git**. Only the lightweight builders and tests are part of the repository.
