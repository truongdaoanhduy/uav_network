# Legacy provenance family repair — COMPLETE

Updated: 2026-10-26T03:35:45.108376+00:00

## Scope

Completed the bounded same-collection provenance repair for all six historical checkpoint families. The previously verified one-version `ea73429e` canary was preserved and the remaining five families were repaired sequentially, never in parallel. No checkpoint payload was rewritten, no production training was launched, and source scientific metrics/configuration were not changed.

## Repaired families

- `uav-network-masac-b97d9f62`: 79 versions (`v0..v78`);
- `uav-network-masac-bcbd5a81`: 24 versions (`v0..v23`);
- `uav-network-masac-ea73429e`: 1 version (`v0`, prior canary);
- `uav-network-matd3-501d6d1a`: 1 version (`v0`);
- `uav-network-matd3-6c1b317d`: 79 versions (`v0..v78`);
- `uav-network-matd3-e0c7b655`: 3 versions (`v0..v2`).

Total: **187 W&B manifest-only versions** reconstructing the existing **193 immutable HF files**. The six extra files are the final-family `resume.pt` entries already present in the immutable archive.

## Repair procedure

For each family, before any source-summary update the repair rechecked that the source run was finished, config and scientific-summary hashes matched the read-only proof, expected W&B collection/version refs were absent, and immutable HF size/SHA metadata matched. Versions were then recreated in ascending original-version order using the original collection name so the repaired refs regained their original `:vN` identities. Each created artifact contains only `checkpoint.json`.

Source scalar archive mappings and `checkpoint_artifact` were updated only after the whole family had been published and representative immutable HF payloads were force-downloaded/hash-verified. Each family then received a separate fresh verification pass checking exact version count, manifest-only artifacts, source/config/scientific hashes, HF metadata, resolver behavior and final payload SHA.

The global closure also found one old fallback key on `e0c7b655` (`checkpoint_archives.<original-v0>`) still pointing to missing `:v3`. The scalar mapping already made normal resolution correct, but the stale fallback provenance value was repaired to restored `:v0` and rechecked without changing scientific summary/config hashes.

## Final global verification

`outputs/cuda-scenario-audit-20261026/legacy-provenance-repair/global-repair-verification.json` records `ok=true` with:

- **6/6 families**;
- **187/187 versions**;
- **193/193 immutable HF files** metadata-verified;
- every source still `finished`;
- all six config hashes unchanged;
- all six scientific-summary hashes unchanged;
- exact collection version counts;
- all recreated artifacts manifest-only;
- all scalar archive mappings point directly to restored original refs;
- all final `checkpoint_artifact` pointers point directly to restored final versions;
- legacy fallback mapping repaired.

Per-family receipts are under `outputs/cuda-scenario-audit-20261026/legacy-provenance-repair/family-*`; the original `ea73429e` canary remains under `canary-ea73429e/`.

## Ruling

Legacy W&B checkpoint provenance reconstruction is complete for the six proven families. The immutable HF archive remains the payload source and was not rewritten. The next storage task is a new fresh closure for additional periodic-policy cleanup candidates; the prior remaining-candidate count must not be treated as authorization without re-proof.
