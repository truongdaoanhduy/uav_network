# Periodic-policy cleanup batch 05 — verified

Updated: 2026-10-26T04:36:53.617928+00:00

## Scope

Batch 05 was bounded to producer `78928740`, selecting exactly four unaliased/unused MATD3 periodic policy versions `v1`–`v4` / **224,201,964 B**. The retained successor is aliased full-state `v5`; historical `v0` had already been retired by the original three-version canary. No production training or scientific configuration was changed.

## Concurrency finding

A second batch-05 executor was already active. The local apply attempt was stopped by its fresh W&B model-set guard after the concurrent executor had retired the four W&B versions. Investigation showed the concurrent executor then completed the guarded HF head commit and four LFS purges. The failed local path performed no duplicate deletion.

Because destructive receipts already existed, recovery used verification only; the apply path was not repeated.

## Fresh independent verification

Verification-only recovery passed with exit code 0:

- exact selected W&B refs `78928740:v1`–`v4` are retired with `delete_aliases=false`;
- the producer collection now retains exactly `v5`;
- exactly **4** selected HF head paths and **4** LFS objects / **224,201,964 B** are removed;
- `rewrite_history=false`;
- public HF head is exactly `b1c70eca202481254e1640d25926105261eb2b78`;
- main HF repo is unchanged;
- only expected public paths were removed and the batch tombstone was added;
- all original Git revisions remain;
- fresh W&B model set is exactly **1,295** versions;
- **1,368** surviving immutable tuples verify;
- all **1,057** retained tuples verify;
- all **78** retained full-state manifests remain protected;
- all **187** repaired legacy W&B versions / **193** immutable legacy HF paths remain valid;
- all six synthetic-test pollution paths remain preserved;
- producer scientific record remains unchanged;
- three retained checkpoint samples were force-downloaded and SHA256 verified.

Batch 05 leaves **311** periodic-policy candidates untouched.

Cumulative periodic cleanup through batch 05: **41 retired versions / 6,207,324,035 B**.

Receipts: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-05/`.
