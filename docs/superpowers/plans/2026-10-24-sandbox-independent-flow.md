# Sandbox-independent flow implementation

Spec: ../specs/2026-10-24-sandbox-independent-flow-design.md.
BASE: 1b9a91d. Execute inline with one fresh final reviewer; preserve all active GPU jobs.

## Builder decisions
Builder check found two real gaps. Preserve all 30 configured evaluation episodes. CPU visualization gets a 39600-second bounded session, records completed episode/seed prefixes and resumes those records in a uniquely named subsequent CPU attempt when needed. A single episode that exceeds an entire session is an explicit blocker, not silently shortened. Measure CPU episode runtime. Completion requires both W&B media and immutable HF assets. Transient publication failures receive publication-only retries from saved Kaggle assets (maximum three recovery attempts), not another 30-episode evaluation.

Keep GPU recovery source4cd9358 for existing lineages; controller/CPU evaluation source is the new tested immutable commit. Existing GPU-produced CPU jobs are observed first; if their old one-hour limit is insufficient, recover on a uniquely named new CPU attempt. No overwrite.

## Tasks and interfaces
1. CPU evaluation: add optional completed_records/should_pause/on_checkpoint callbacks; default behavior unchanged. Validate exact seed/index prefix. Atomic small evaluation-progress.json is reusable without model/replay changes. CPU spec timeout39600; publication-only recovery. Tests first: deterministic tiny real policy resumed metrics equal uninterrupted evaluation, pause preserves records, invalid seed rejects, generated timeout is used.
2. Controller: CPU bootstrap stage, private credentials, bounded controller sessions, own lineage state; active GPU observation, validated terminal results, deterministic child adoption, same-account quota wait, explicit GPU source, final CPU progress/publication recovery and completion. Persist atomic state and fresh W&B heartbeat. Tests first: external I/O fake only; exercise actual decision functions and generated bootstrap.
3. CLI --auto-continue automatically seeds a controller after saving GPU manifest; failure keeps the GPU receipt and explicitly marks unattended readiness false. Seed five existing lineages only after source is published and smoke verified.
4. Verification/deployment: focused tests then full CPU suite; independent review; actual mini GPU -> CPU media E2E smoke on account35, controller handoff test, CPU episode timing. Production settings unchanged. Verify five cloud controllers and their W&B heartbeats; update secondary watcher to read-only when controller owns lineage.

Shared interfaces: controller downloads job/result/continuation/cpu/evaluation-progress/result JSON from terminal Kaggle kernels; uses existing next_stage, build_continuation_spec, submit_continuation and submit_visualization. Validation happens before any submission; existing owned refs are always adopted, never overwritten. Submission timeout ambiguity persists as a pending intent and is reconciled before further mutation.

Review focus: terminal-vs-running races, private missing-kernel ambiguity, source pinning, secrets, quota and session handoff, exact evaluation prefix, old CPU timeout recovery, partial publication, unexpected malformed control files. No claim that tiny smoke validates 20k policy quality or a 12h finish.

## Ledger
- Preflight: existing auto-submission and new controller share deterministic child refs; controller acts only after parent terminal and definitive owned-child reconciliation.
- Preflight: old CPU one-hour jobs cannot consume new resume callback; controller adopts, then recovers a failed terminal attempt with a new unique ref, retaining failed evidence.
- Task1: complete. RED3missingAPI/timeout then GREEN deterministic resumable evaluation. CPU runner pause/restore and real asset reload tested; final default behavior unchanged.
- Task2: complete. REDmissingcloudexecutor, GREEN14controllercases; active observation, quota wake, source pin, existing/ambiguous child adoption, CPU delivery, >3 progressing sessions, bounded handoff/retry, config drift and secret rejection.
- Task3: complete. RED2launchreadinesscases -> GREEN. GPU launch receipt is written before cloud controller submission; scheduling failure reports readiness false without cancelling GPU.
- Task4: in progress. Cloud/eval focused75passed28subtests; extra hardening16passed. Full suite running; independent review and real cloud smoke/deployment next.
