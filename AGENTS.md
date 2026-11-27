# Active repository engineering rules

- Only the current seven-radio scenario in research/zhang_xue is active.
- Old world and radio simulation configs must not be imported or reintroduced.
- Maintain MATD3 and MASAC loss, actor, critic and optimizer semantics.
- Keep every owned source file within 500 lines; organize by module/folder.
- Scientific settings must be validated through YAML, dataclasses and CLI.
- Apply RED -> GREEN -> regression before changing scientific logic.
- Keep environment truth private from decentralized Actor execution.
- Check packet custody, TTL, ACK deduplication, radio energy and physical safety.
- Store model weights, maps, plots, W&B and secrets outside Git.
- Do not claim full GPU training or live tracking when not implemented.
- Only push a dedicated branch with passing checks and verified remote SHA.
- Preserve prior experiment history in Git, not duplicated active files.

Repository contract: docs/PROJECT_ENGINEERING_CONTRACT_20261125.md.
