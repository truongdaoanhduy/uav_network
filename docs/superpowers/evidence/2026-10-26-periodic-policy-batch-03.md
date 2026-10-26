# Periodic-policy cleanup batch 03 — verified

Updated: 2026-10-26T04:16:18.890238+00:00

## Scope

Batch 03 was bounded to the 12 fresh-proved periodic policy versions of producer `01c9edab`. No production training or scientific configuration was changed.

The proof-only receipt selected exactly 12 refs / **2,051,703,428 B**, with no aliases, consumers, functional ref pins, SHA pins, or path pins.

## Reconciliation

When execution was resumed, destructive receipts already existed. The fail-closed apply guard refused to repeat the batch. The existing receipts showed that all 12 W&B version retirements, the guarded HF head commit, and all 12 LFS purges had already occurred. No destructive operation was repeated during recovery.

Recorded HF commit: `971b770ba903e06dec90c7baab30af12c2e7ce59`, parent `d6fc553f587508bf4f760e28449338eacbbb4eb4`, with tombstone `retirement/20261026-periodic-policy-batch-03.json`. `rewrite_history=false`.

## Fresh independent verification

Verification-only recovery passed with exit code 0:

- exactly **12** selected W&B versions are retired;
- exactly **12** selected LFS objects / **2,051,703,428 B** are gone;
- public HF head is exactly the recorded batch commit;
- main HF repo is byte/tree/ref/commit unchanged;
- only the 12 expected public head paths were removed and the audit tombstone added;
- all original Git revisions are retained;
- fresh W&B model set is exactly **1,301** versions;
- **1,374** surviving immutable tuples verify;
- all **1,057** retained tuples verify;
- all **78** retained full-state manifests remain protected;
- all **187** repaired legacy W&B versions / **193** immutable legacy HF paths remain valid;
- all six synthetic-test pollution paths remain preserved;
- producer scientific record remains unchanged;
- three retained checkpoint samples were force-downloaded and SHA256 verified.

Batch 03 leaves **317** periodic-policy candidates untouched.

Cumulative periodic cleanup through batch 03: **35 retired versions / 5,572,364,993 B** (3-version canary + batch01 + batch02 + batch03).

Receipts: `outputs/cuda-scenario-audit-20261026/storage-rebuild/policy-batch-03/`.
