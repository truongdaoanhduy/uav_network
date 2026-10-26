# Periodic policy cleanup batch 02 — VERIFIED

Updated: 2026-10-26T03:53:54.341839+00:00

A fresh post-provenance-repair closure selected exactly 10 additional unaliased, unused periodic policy versions from completed producer `01c9edab`. No other candidate was authorized or removed.

## Fresh pre-write proof

- live model versions before batch: **1323**;
- live immutable tuples: **1396/1396 verified**;
- repaired legacy W&B versions preserved: **187**;
- protected immutable legacy HF paths: **193**;
- retained immutable tuples: **1057**;
- retained full-state manifests: **78**;
- selected versions: **10**, totaling **1,790,786,734 B**;
- no functional ref, SHA, checkpoint-path, consumer, alias, legacy-path, or synthetic-test collision was present;
- HF heads matched the post-batch-01 reviewed heads before mutation.

Proof: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-02-proof/proof.json`.

## Applied mutation

The exact 10 W&B versions were retired with `delete_aliases=False`. The exact 10 public-HF checkpoint paths were removed under parent commit guard and tombstone `retirement/20261026-periodic-policy-batch-02.json` was added. The matching 10 LFS objects were physically purged with `rewrite_history=False`.

Public HF head advanced only from `1859682381403d7734fd031eb966a90528f9f9f7` to `d6fc553f587508bf4f760e28449338eacbbb4eb4`. Main checkpoint repo did not change.

## Fresh independent verification

`fresh-verification.json` is `ok=true` and proves:

- exactly **10** W&B versions removed;
- exactly **10** LFS objects / **1,790,786,734 B** removed;
- live model versions now **1313**;
- **1386/1386** surviving immutable tuples verify;
- **1057** retained immutable tuples preserved;
- **78** retained full-state manifests preserved;
- all **187** repaired legacy versions and **193** protected legacy HF paths preserved;
- all original Git revisions retained;
- only expected public main ref advanced and only expected head paths were deleted;
- six synthetic-test paths preserved;
- producer scientific record unchanged;
- three retained checkpoints force-downloaded and SHA256 verified.

After canary + batch01 + batch02, **23** periodic versions and **3,520,661,565 B** of corresponding LFS payload have been retired. **329** old proposal candidates remain untouched and require a new fresh closure before any further mutation.
