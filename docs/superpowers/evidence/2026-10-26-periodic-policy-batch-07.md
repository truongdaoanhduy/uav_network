# Periodic-policy cleanup batch 07 — verified

Updated: 2026-10-26T05:36:12.211269+00:00

## Scope

Batch 07 was bounded to producer `90690601`, selecting exactly four unaliased/unused MATD3 periodic policy versions `v0`–`v3` / **241,389,740 B**. The retained successor is aliased full-state `v4`. No production training or scientific configuration was changed.

## Execution and transport reconciliation

The proof-only phase freshly verified the post-batch06 W&B/HF state, all candidate pins, all current immutable tuples, legacy provenance protection and synthetic-test pollution exclusions. The apply command returned a Sandbox transport 502 to the caller, but fresh reconciliation showed that the command itself continued and completed normally. Completion, W&B deletion, HF head-commit and LFS purge receipts were present. Apply was not repeated.

## Fresh independent verification

Independent verification passed with exit code 0:

- exact selected W&B refs `90690601:v0`–`v3` are retired with `delete_aliases=false`;
- producer collection retains exactly successor `v4`;
- exactly **4** selected HF head paths and **4** LFS objects / **241,389,740 B** are removed;
- `rewrite_history=false`;
- public HF head is exactly `2c68392e5ccc500a2fc13d4d8a2f4b444175bcb8`;
- main HF repo is unchanged;
- only expected public paths were removed and the batch tombstone was added;
- all original Git revisions remain;
- fresh W&B model set is exactly **1,287** versions;
- **1,360** surviving immutable tuples verify;
- all **1,057** retained tuples verify;
- all **78** retained full-state manifests remain protected;
- all **187** repaired legacy W&B versions / **193** immutable legacy HF paths remain valid;
- all six synthetic-test pollution paths remain preserved;
- producer scientific record remains unchanged;
- three retained checkpoint samples were force-downloaded and SHA256 verified.

Batch 07 leaves **303** periodic-policy candidates untouched.

Cumulative periodic cleanup through batch 07: **49 retired versions / 6,671,326,747 B**.

Receipts: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-07/`.
