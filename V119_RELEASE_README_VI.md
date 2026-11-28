# V119 — ZIP FULL SOURCE + AUDIT DS-only

Đọc `docs/V119_INDEPENDENT_DS_SCENARIO_AUDIT_VI.md` trước. Code canonical V118 nguyên bản trừ đúng 2 file core đã sửa: `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py` (C38), `scripts/gpu/run_v115_canonical_ds_baseline.py` (C42 C48 C51). Scripts/tests/audit docs mới được thêm, algorithms/30 gốc giữ nguyên.

Tất cả 72 conflicts (C01–C72) trong `outputs/v119_reaudit/06_FULL_CONFLICTS_C01_C72_WITH_CURRENT_STATUS.csv`, gốc lịch sử nguyên trong `outputs/v117_audit_summary`. Paper/source in `07_PAPER_CODE_PROVENANCE_V119.csv`. V118 natural experiment summary under `outputs/v118_audit`; no original raw packet logs V117. Vast evidence (nếu đã copy) under `outputs/v119_reaudit/vast_raw/`. `SCENARIO_FROZEN=false`, no RL training and no S25 patch.

Lệnh demo stress `python scripts/gpu/v119_four_way_cuda_gate.py --dry-run --output-dir ...` và 4 stress GPU thực trong main report. Đường dẫn source/manifest tự đối chiếu từ root. No secret/key.txt/SSH credentials included.


**Vast 2026-11-28 — Kết quả V119 đã xác nhận:** full pytest `603 passed, 6 skipped, 120 subtests passed`; GPU action command CPU-vs-CUDA nominal max error `3.6e-7 m/s`, red V118 discrepancy `0.508486 m/s`; 4/4 stress CUDA post-patch `4096×900` PASS zero safety fail/overflow/NaNs (reset mode). Paired mission tự nhiên **6 case × 5 seeds × 4 worlds/seed ×900 first-terminal steps**, RAW JSON và log đều được chuyển và kiểm sha; file `25_V119_PAIRED_NATURAL_900_SUMMARY.csv`, `26_V119_PAIRED_DELIVERY_DELTAS_SEED_BOOTSTRAP_CI.csv`.

**Bảng cốt lõi:** base 98%; RF-only 56.667%; RF+PEER 24%; RF+PEER+MULE 24.667%; base+PEER 96.333%; DS0.99 95.333%. Các kết quả là heuristic controllers, **không phải RL trained results**. PEER/RF có tác động xấu trong setup này; chưa parity SimPy hay hiệu chuẩn DS. `SCENARIO_FROZEN=false`. Raw Vast **23 files** trong `outputs/v119_reaudit/vast_raw/`, không phải chỉ báo cáo tóm tắt.
