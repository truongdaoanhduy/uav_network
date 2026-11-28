# V121 — Kết quả từng bước (28/11/2026)

- [x] 01. Nhận đúng ZIP V120 gốc; kiểm SHA256 và giữ input, C01–C72, mọi lịch sử log.
- [x] 02. Khóa chế độ CPU canonical DS-only; test RED→GREEN. Chỉ giữ Bayes helper lịch sử ngoài runtime.
- [x] 03. Sửa lỗi đơn vị 1m/s-vs-15m/s của hai script hiệu chuẩn, chạy lại 45+45 CPU missions 3 seeds×300, cập nhật C04/C05/C12/C13/C23/C28/C35/C41/C66.
- [x] 04. Thêm kiểm tra CUDA full 4096×900 *không auto-reset*; hậu vá shield Base và RF pure-GCS 2/2 PASS, 0 safety/overflow/nonfinite, mọi world đủ 900.
- [x] 05. CPU MAC simultaneous due-event contention + HELLO telemetry airtime RED→GREEN.
- [x] 06. CPU/Warp shield near-tie: trước 16/900 mismatch max 2.8701m/s; sau 0/900 vượt sai số, max 3.86e-6m/s. Test fixture projection trước đó sai scope, đã RED→GREEN đúng geometry-only, không đổi safety rule.
- [x] 07. Actual constructive oracle feasibility witness 10/10 success (5 seed × Base+RF), 15/15 GCS confirmed each, không phải policy học hay non-oracle proof; full per-step JSONL.
- [x] 08. 82 conflict C01–C82, 72 historical records retained, report tiếng Việt, source diff và SHA256.
- [x] 09. Final full regression: **640 passed, 6 skipped, 152 subtests passed**; log `outputs/v121_scenario_audit/pytest_release_final.stdout.log`. Source manifest and project policy reverified after documentation update.
- [ ] Scientific freeze **không đạt**: DS temporal correlation, GCS false confirmation reliability, model RF/energy, strict upstream UavNetSim SimPy event parity and whole-world CPU–CUDA parity remain open or scope-limited.

**Quyết định:** `SCENARIO_FROZEN=false`, Phase2 MATD3/MASAC baseline chưa bắt đầu, thuật toán và 2 YAML active giữ nguyên byte, không git push.

**Lưu ý:** V120 tuyên bố chỉ hai GPU command cuối là đủ freeze đã bị bác bỏ bởi corrected calibrated FP và energy data. Đọc `V121_FULL_SCENARIO_AUDIT_AND_OUTSTANDING_GATES_VI.md` và `outputs/v121_scenario_audit/V121_CONFLICT_REGISTRY_C01_C82.json`.
