# Matched architecture matrix — seed 44 (2026-11-04)

Source baseline: `cf09ccc` (3D XYZ, Bayesian, CBF). The branch changes only the experiment generator/test, not the MASAC/MATD3 implementation.

Generate **10** matched YAML configs without launching cloud work:

```bash
PYTHONPATH=. python scripts/prepare_matched_architecture_study.py
```

Manifest: `outputs/matched-architecture-seed44-20261104/manifest.json`.
The 10 configs differ only by algorithm (MASAC/MATD3), training regime (BASE/PQL), sharing (independent/shared), vectorization flag for MASAC BASE shared control, and output path. `seed=44`, `deterministic=true`, fixed-order BASE collection (`collector_overlap=false`), FP32 eager, chunk size 0, output normalization off, CUDA runtime, 4096 envs, 20k episodes, 1500 max steps, 20 targets/15 obstacles, 30 eval episodes seed 2000000044 are common. Their PQL training uses `mode=async`, but deterministic scheduling prevents concurrent worker threads.

## Verified locally (CPU only)

- 29 focused tests and 46 subtests passed.
- All 10 short Warp CPU smoke trainings finished: **4 episodes, 7 critic updates** each, with small (2-env, 4-step) test overrides; this is not a training-performance comparison.
- MASAC PQL Shared Run 04/10: identical model/episode/settings hashes; a second check in **two separate Python processes** also reproduced all three hashes.
- MASAC BASE Shared Run 02/09: identical model/episode hashes when the vectorization flag is toggled. The shared critic uses a single network and does not call independent-critic `torch.vmap` in MASAC.
- Supplementary **Independent critic** MASAC BASE `vectorize_critics=true` smoke (not one of 10): model hash changes, episode hash does not; maximum absolute final-parameter difference `7.450580596923828e-09` across 7 updates, consistent with FP32 operation-order rounding. This is not evidence of equal final learning performance after 20k episodes.

JSON evidence: `docs/superpowers/evidence/2026-11-04-matched-architecture-seed44.json`. Raw local logs/configs/checkpoints are in `outputs/matched-architecture-seed44-20261104/` (ignored by git).

**No Kaggle jobs were submitted.** Kaggle Terms of Use do not permit a single person operating multiple active accounts to increase GPU quota. Study files can be used with an authorized GPU platform. No conclusion on speed or algorithmic superiority without matched long GPU training and fixed-seed evaluation; for a real vmap speed comparison use **Independent critic** with the flag off/on at equal workload.
