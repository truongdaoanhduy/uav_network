# One W&B workspace and external checkpoints

The user approved implementing the preceding proposal on 2026-10-15 and additionally requires every run to appear in the main W&B workspace. Continue in the existing isolated branch; no additional approval cycle is needed.

- Main destination: `truongdaoanhduy-uit/uav_search_target`, resolved from the existing credential configuration for local and cloud execution. Cloud jobs explicitly carry entity/project; reject a mismatch rather than silently using a login's default team.
- Default Personal Workspace has no run-group, ID, search, or selection restriction. Preserve existing panels. Include training, validation, benchmark, continuation and CPU visualization runs; use run names and stage metadata to distinguish them. Reconcile visibility when each online run starts, without creating Saved Views.
- Hugging Face private checkpoint repository from the existing `huggingface` credential section holds policy and full replay/optimizer/RNG state. W&B artifacts store a small JSON manifest with immutable HF commit, file paths, sizes and SHA256. Never place tokens in configs, manifests, source, output or Git.
- New cloud jobs require the external store to pass preflight. GPU and CPU inputs receive only the single HF token in a verified owned private credential dataset. Never substitute a full `key.txt` input.
- Download code understands both legacy W&B binary artifacts and new manifest artifacts. Validate immutable revisions, file names, file sizes and SHA256 before existing algorithm/progress/state checks. GPU continuation retains full replay; CPU downloads only policy.
- CPU visualization opens and closes its own online W&B run in the same project, logs evaluation metrics and output locations, and keeps MP4/PNG outputs on Kaggle. It must remain CPU-only.
- Back up completed, unused legacy artifacts before removal. Verify a private remote immutable copy using file metadata and a fresh downloaded copy/checksum before deletion. Preserve current production artifacts and any immutable version referenced by an active continuation/CPU job. Save an archive index and replacement reference in W&B. Do not delete run histories, change training math, restart active production jobs or buy storage.
- Retention operates on superseded completed checkpoint versions after verified backup, keeping latest/best/final references available through manifest replacements. Existing pinned jobs keep their source and upload behavior; report that limitation.

Verification: fail-first tests at external service boundaries, existing cloud/continuation/tracking suite, real online CPU training/evaluation with an external checkpoint round-trip, W&B Personal Workspace readback, then archive verification and a fresh storage inventory.
