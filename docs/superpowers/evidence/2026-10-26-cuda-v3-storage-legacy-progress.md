# CUDA v3, bounded storage cleanup, and legacy provenance proof

Updated: 2026-10-26T02:11:34.313187+00:00

## CUDA v3 — latest-HEAD verification

Kaggle kernel `duytrngoanh/uav-cuda-scenario-algorithm-audit-20261025` completed successfully pinned to source commit `4c5b8862b2c71e6dc7ead6b9b99e1402ca2fd137`.

Hardware receipt:

- 2 x Tesla T4;
- Torch 2.11.0+cu128;
- CUDA 12.8;
- Warp 1.18.0;
- Warp default device `cuda:0`;
- Python 3.13.15.

Stages:

1. scenario tests: exit 0, **280 passed, 14 warnings, 115 subtests**;
2. production-shaped primary scenario probes: exit 0, status pass; diagnostic/oracle traffic delivered all 35 targets and exercised multi-hop/network energy;
3. CUDA learner probes: exit 0 with `FOUR_CUDA_LEARNERS_PASS` for MASAC BASE/PQL-inspired and MATD3 BASE/PQL-inspired;
4. full repository CUDA suite: exit 0, **834 passed, 21 skipped, 37 warnings, 2432 subtests, 0 failed**.

The v2 14-failure cluster is resolved as a test-environment classification defect. On v3 the affected archive/visualization tests are explicit CPU-only skips; production CPU-only guards remain unchanged.

Primary receipt directory: `outputs/cuda-scenario-audit-20261026/kaggle-v3-latest/`.

## Periodic-policy cleanup — bounded batch 01

After the earlier 3-version canary, a fresh read-only closure selected exactly 10 public-repo periodic policy versions from producer `01c9edab`, superseded by retained full-state successor `uav-network-masac-01c9edab:v32`.

Fresh proof verified:

- 10 selected versions;
- 1,653,111,278 B;
- no functional ref/SHA/path pins;
- current HF heads unchanged since proof;
- 1,213 current immutable tuples before the batch;
- legacy archive paths and synthetic-test pollution paths excluded.

The exact W&B versions were retired with `delete_aliases=False`, the exact HF paths were removed under parent-head guard with a tombstone, and the 10 matching LFS objects were physically purged with `rewrite_history=False`. The apply script then hit a local receipt-formatting bug (`LFSFileInfo` indexed as a dict) after the purge receipts had already been written; destructive work was **not** repeated. A verification-only recovery proved the final state.

Final batch-01 receipt:

- `ok=true`;
- 10 W&B versions retired;
- 10 LFS objects / **1,653,111,278 B** removed;
- 1,203 immutable tuples verified after removal;
- 864 retained immutable tuples preserved;
- 72 retained full-state manifests preserved;
- 193 protected legacy archive paths preserved;
- six synthetic-test paths preserved;
- all original Git revisions retained;
- only the expected public main ref advanced;
- three retained checkpoints were force-downloaded and SHA256 verified;
- **339** periodic-policy candidates remain and are not authorized by this receipt.

Receipt: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-01/completion.json`.

## Legacy archive provenance — read-only proof

The six missing W&B artifact families are:

- `uav-network-masac-b97d9f62`;
- `uav-network-masac-bcbd5a81`;
- `uav-network-masac-ea73429e`;
- `uav-network-matd3-501d6d1a`;
- `uav-network-matd3-6c1b317d`;
- `uav-network-matd3-e0c7b655`.

Fresh W&B probes confirm the original collections and the currently referenced replacement versions are absent. The protected HF archive at immutable revision `446dced7362791e68a027a64b8631545263f47b3` still contains exactly **193 files / 37,418,273,432 B** with API metadata matching immutable Git-LFS pointers.

The read-only repair proof establishes:

- **187** archived original W&B versions across the six families;
- those 187 versions account exactly once for all 193 HF files;
- each family's final original version contains both `checkpoint.pt` and `resume.pt`, explaining the six extra files;
- all 187 reconstructed `checkpoint.json` descriptors pass `validate_manifest`;
- all six source runs remain finished and their scientific config/summary hashes are unchanged;
- one real HF checkpoint per family was force-downloaded and SHA256 verified;
- no remote repair mutation has been performed yet.

The safe write design is to recreate manifest-only artifacts in the **same original collection names**, starting with a one-version family canary. Do not create `-archive-repair` collections because the current resolver requires the mapped artifact ref to use the original collection prefix. Any source-summary mapping update occurs only after the new manifest artifact and immutable HF payload are freshly verified.

Proof: `outputs/cuda-scenario-audit-20261026/legacy-provenance-repair/repair-plan-readonly.json`.

## Current next order

1. run a one-version legacy provenance repair canary with strict version/ref verification;
2. if green, extend the repair in bounded families while preserving old broken refs in durable receipts;
3. continue periodic-policy cleanup only in fresh, bounded batches from the remaining 339 candidates;
4. consolidate final storage totals and update the durable task record;
5. no corrected 20k production cohort is launched by this work.
