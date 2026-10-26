# Periodic-policy cleanup batch 06 — verified

Updated: 2026-10-26T05:22:24.380859+00:00

## Scope

Batch 06 was bounded to producer `7f79f516`, selecting exactly four unaliased/unused MATD3 periodic policy versions `v0`–`v3` / **222,612,972 B**. The retained successor is aliased full-state `v4`. No production training or scientific configuration was changed.

## Execution and transport reconciliation

The proof-only phase freshly verified the post-batch05 W&B/HF state, all candidate pins, all current immutable tuples, legacy provenance protection and synthetic-test pollution exclusions. The apply command returned a Sandbox transport 502 to the caller, but fresh reconciliation showed that the command itself had continued and completed normally. Completion, W&B deletion, HF head-commit and LFS purge receipts were present. Apply was not repeated.

## Fresh independent verification

Independent verification passed with exit code 0:

- exact selected W&B refs `7f79f516:v0`–`v3` are retired with `delete_aliases=false`;
- producer collection retains exactly successor `v4`;
- exactly **4** selected HF head paths and **4** LFS objects / **222,612,972 B** are removed;
- `rewrite_history=false`;
- public HF head is exactly `61756dc697cf95a9efb6a25c2bc0b736deb3fff3`;
- main HF repo is unchanged;
- only expected public paths were removed and the batch tombstone was added;
- all original Git revisions remain;
- fresh W&B model set is exactly **1,291** versions;
- **1,364** surviving immutable tuples verify;
- all **1,057** retained tuples verify;
- all **78** retained full-state manifests remain protected;
- all **187** repaired legacy W&B versions / **193** immutable legacy HF paths remain valid;
- all six synthetic-test pollution paths remain preserved;
- producer scientific record remains unchanged;
- three retained checkpoint samples were force-downloaded and SHA256 verified.

Batch 06 leaves **307** periodic-policy candidates untouched.

Cumulative periodic cleanup through batch 06: **45 retired versions / 6,429,937,007 B**.

Receipts: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-06/`.
