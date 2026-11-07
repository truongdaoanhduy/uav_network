# Independent baseline Kaggle UI continuation — 2026-11-07

The user requested continuing the concrete baseline work in the previous
report. The existing independent-profile YAML and established manual Kaggle
UI workflow define the design; this step packages them without changing the
algorithm, scenario, reward, CBF or initialization.

## Deliverables and implementation choices

- Two separate self-contained notebooks: MASAC and MATD3, BASE, independent
  actors/critics, plain MLP, NOCHUNK, seed44, same1500-step scientific profile.
- Both embed the same pinned clean source archive and use the same existing
  HF-best/W&B training-only runner. No automatic cloud submission.
- Reuse the common bootstrap generator with optional archive input and
  explicit Kaggle Secrets credentials; legacy dataset credentials stay default.
- Build notebooks directly from repository sources, without requiring a
  previously downloaded notebook template.
- Editable train seed (default44); fixed common validation layout seeds.
  Names/tags reflect the resolved algorithm/profile/seed and use fresh run IDs.
- Create fresh per-run lineages; preserve historical SH and initialization
  notebooks. Distinct output file names.

## Tasks

- [x] Add failing tests for archive reuse, credential source, effective profile,
  edited seed/name agreement, fresh identity, serialized notebook validity.
- [x] Implement the minimal common-bootstrap options and repository UI builder.
- [x] Run focused contracts and CPU train/reload checks; run repository suite.
- [ ] Commit/push source before bundling, build only the two new notebooks.
- [ ] Inspect actual archives, checksums, source equality, resolved configs,
  compile cells and verify both saved copies. Record verification and commits.

## Review focus

- No SH/LayerNorm or Kaiming flags inherited from old notebook templates.
- MATD3 and MASAC keep equal scene/reward/evaluation protocol.
- A changed train seed cannot leave stale W&B labels or tags.
- UI credential access never embeds credentials or launches an API job.
- Archive checksum and claimed source commit match the built runtime source.

A CPU or schema pass is not evidence of T4 memory fit or convergence.

Broad run:1356 passed/103 skipped;4 static-guard failures were repaired.
Configuration/tracking formatting is AST-identical, archived history is byte-
identical, and13 guard/notebook tests plus578 subtests passed after refresh.
Independent read-only review approved; no algorithm changes in this export.
