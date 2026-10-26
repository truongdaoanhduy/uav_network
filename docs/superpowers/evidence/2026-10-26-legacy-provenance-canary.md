# Legacy provenance repair canary — verified

Updated: 2026-10-26T02:38:04.280701+00:00

## Scope

Executed exactly one same-collection manifest repair canary for historical source run `ea73429e`. No other legacy family was repaired, no checkpoint payload was rewritten, and no production training was launched.

## Pre-write gates

- source run `ea73429e` was freshly read as `finished`;
- source config SHA256 matched the read-only proof: `3a09b26585dd44092a6157cd59329bdafe9815ac130565c0a460e660e3fda845`;
- scientific summary SHA256 excluding checkpoint provenance matched: `f73a6f94f76dda320ec8710b72008ab772bbcc03f7cc0b8d12f66dbb131ea0ac`;
- original W&B ref `truongdaoanhduy-uit/uav_search_target/uav-network-masac-ea73429e:v0` was absent before the write;
- the previously broken mapped ref `...:v1` was absent;
- immutable HF revision `446dced7362791e68a027a64b8631545263f47b3` contained the expected `checkpoint.pt` and `resume.pt` size/SHA metadata.

## Write

Published exactly one manifest-only W&B artifact in the original collection name:

`truongdaoanhduy-uit/uav_search_target/uav-network-masac-ea73429e:v0`

The artifact contains only `checkpoint.json`, pointing to the existing immutable HF payload. The source summary provenance mapping and `checkpoint_artifact` were updated only after the new artifact and HF payload were verified.

Maintenance run: `repair-ea73429e-20261026`.

## Fresh independent verification

A separate post-write verification passed with exit code 0:

- source remains `finished`;
- source config hash unchanged;
- scientific summary hash unchanged;
- repaired collection has exactly one version and it is `v0`;
- W&B artifact is manifest-only;
- `checkpoint_archive_uav-network-masac-ea73429e_v0` points to repaired `v0`;
- `checkpoint_artifact` points to repaired `v0`;
- both HF checkpoint/resume metadata match immutable size/SHA values;
- `resolve_checkpoint_artifact()` resolves the original ref;
- real `checkpoint.pt` was downloaded and SHA256 verified as `7c7237868b4979679b0ee596e04626155409298f015f97c9b9df1dd8209642c1`;
- maintenance run is `finished` with `maintenance/status=verified` and repaired version count 1.

Receipts:

- `outputs/cuda-scenario-audit-20261026/legacy-provenance-repair/canary-ea73429e/completion.json`
- `outputs/cuda-scenario-audit-20261026/legacy-provenance-repair/canary-ea73429e/fresh-verification.json`

## Ruling

The one-version same-collection repair procedure is validated. The remaining legacy families are still untouched by this canary. The next task, only after user approval, is bounded family repair using the same fresh gate/verify/receipt pattern.
