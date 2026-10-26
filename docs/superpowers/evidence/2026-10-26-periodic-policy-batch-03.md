# Periodic policy cleanup batch 03 — VERIFIED

Updated: 2026-10-26T04:16:45.490372+00:00

Batch03 completed the remaining periodic-policy retirement for producer `01c9edab`. A fresh proof selected all **12** still-live candidates from that producer, totaling **2,051,703,428 B**; no candidate from another producer was authorized.

## Fresh proof

- W&B live model versions before mutation: **1313**;
- immutable tuples: **1386/1386** verified;
- retained immutable tuples: **1057**;
- retained full-state manifests: **78**;
- repaired legacy W&B versions protected: **187**;
- immutable legacy HF paths protected: **193**;
- selected refs: **12**, all producer `01c9edab`;
- selected bytes: **2,051,703,428 B**;
- no functional ref/SHA/path pin, alias, consumer, legacy collision, or synthetic-test collision;
- public HF head matched batch02 verified head `d6fc553f587508bf4f760e28449338eacbbb4eb4`.

Proof: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-03-proof/proof.json`.

## Mutation and receipt anomaly

The exact 12 W&B versions were retired with `delete_aliases=False`. The public HF head advanced under parent guard to `971b770ba903e06dec90c7baab30af12c2e7ce59`, deleting only those 12 checkpoint paths and adding `retirement/20261026-periodic-policy-batch-03.json`. The exact 12 matching LFS objects were purged with `rewrite_history=False`.

During the primary background job an `error.json` was also written at 04:11:58 UTC with the guard message `Existing receipts: reconcile manually; never repeat batch`. It is older than the primary job's terminal completion at 04:13:03 UTC. The primary job stdout shows it continued through `hf_paths_retired`, `lfs_purged`, and `complete ok=true`; its stderr is empty, and the script contains only one `main()` call. The origin of the overlapping/stale guard invocation was not established, so the error receipt is intentionally preserved rather than deleted or explained beyond the evidence. No destructive apply was rerun.

## Fresh independent verification

A separate verifier completed with exit code 0 and `fresh-verification.json: ok=true`:

- **12** W&B versions and **12** LFS objects removed;
- exact byte decrease **2,051,703,428 B**;
- live model versions now **1301**;
- **1374/1374** surviving immutable tuples verified;
- **1057** retained tuples and **78** full-state manifests preserved;
- all **187** repaired legacy W&B versions / **193** legacy payload paths preserved;
- source scientific record unchanged;
- main HF repo unchanged; only the expected public main ref advanced; all original Git revisions retained;
- synthetic-test paths preserved;
- three retained checkpoints force-downloaded and SHA256 verified;
- `uav-network-masac-01c9edab` now has exactly one live model version: retained aliased full-state `v32`.

Cumulative periodic-policy retirement is now **35 versions / 5,572,364,993 B**. **317** candidates from other producers remain untouched and require fresh proof before any mutation.
