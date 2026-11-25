# CURRENT TASK — live index (2026-11-25)

## Priority and truth boundary

- Project branch in progress: `experiment/zhang-xue-v59-architecture-gates-20261125`.
- Baseline branch preserved: `experiment/zhang-xue-v58-audit-20261125` at commit
  `7d7ab1ccea3d31a121f9708e428bbf96c3c529a6`.
- **No MATD3/MASAC training, no Kaggle GPU experiment and no full CUDA parity
  were run by the v5.9 architecture task.** Do not present CPU tests as GPU
  or paper quality results.
- Baseline v5.8: 58 standalone tests passed. The v5.9 modular refactor
  preserves legacy import names and validates CPU action/network behavior.
- Source rules: [repository engineering contract](../PROJECT_ENGINEERING_CONTRACT_20261125.md)
  (original source: `docs/implementation-plan.md`, `docs/design.md`).
- SHA-pinned 18d756ef actor/critic files are not modified or repartitioned.

## History without duplicating code/docs in HEAD

All previous 1,219 lines of this file are preserved byte-for-byte in immutable
Git history at [v5.8 CURRENT_TASK](https://github.com/truongdaoanhduy/uav_network/blob/7d7ab1ccea3d31a121f9708e428bbf96c3c529a6/docs/superpowers/CURRENT_TASK.md).
Original SHA256: `4a9a1c177d9d6dda059930a91417091250d59b7afa6e5855832265dee7d7468c`.
Use that link to recover prior task-specific decisions. This short live index
avoids duplicating 1,219 historical lines in every new branch, but it does NOT
remove the original content from Git history.

## Gate status and continuation

1. Keep v5.8, v5.3, and main branches unchanged; use isolated v5.9 branch.
2. Governance changed-file limit and module-size test must pass. On request,
   all 25 historical Kaggle notebooks were retired from v5.9 HEAD, preserved
   in v5.8 immutable history. Full-tree policy remains RED due to inherited
   >500-line source and documentation files.
3. Validate scenario YAML/`--set` and `setup.sh` including pinned UavNetSim.
4. Verify Warp CPU reset, injection, two-hop DATA and E2E ACK after refactor.
5. Real CUDA/T4 E32→2048, per-event CPU–Warp parity and complete training
   coupling remain future blockers; never auto-launch a paid GPU task.
6. After push, record actual new SHA and repository links in the release docs.
