# UAV V84 — kịch bản và mã nguồn hiện hành

Kịch bản đang hoạt động nằm tại [experiments/v80_vast_20261126](experiments/v80_vast_20261126).
Tên thư mục được giữ để bảo toàn lịch sử; runner `run_v81_real_rollout.py` báo version V84.
Nhánh: `experiment/v84-v83-independent-validation-20261126`.
Commit mã đã kiểm: `13b6cc499883853f1178b5833c8492c23db534e9`.

Đọc [báo cáo V83→V84](docs/methods/v84_20261126/01_V83_AUDIT_V84_FIXES_VI.md),
[hướng dẫn chạy từng bước](docs/methods/v84_20261126/06_REPRODUCE_STEP_BY_STEP_VI.md)
và [trạng thái tiếp theo](docs/superpowers/CURRENT_TASK.md).

## Kịch bản

Map 1000×1000 m; 6 UAV và GCS; 15 targets; belief grid20×20.
UAV bay XY ở cao50m, dt1s, vmax15m/s, amax5m/s², safe distance50m.
Actor518 inputs không nhận oracle target truth. RF dùng custom microsecond discrete-event
CSMA/CA trên Warp, chưa chứng minh parity đầy đủ với upstream SimPy.
Các scenario cũ tại root là lịch sử, không phải entry point của V84.

## Kiểm tra

```bash
cd experiments/v80_vast_20261126
python3 scripts/maintenance/verify_source_manifests.py --check
python3 scripts/maintenance/check_project_policy.py --scope full
PYTHONPATH=.:research/zhang_xue/src pytest -q tests research/zhang_xue/tests
```

Source active301+122 checksums, policy303 files/0violations.
CPU439 passed/5skipped; suite bật CUDA440 passed/4skipped;100 subtests mỗi lượt.
Gói ZIP giữ full raw evidence, lịch sử Kaggle/Vast, patches và SHA inventory.
Git chỉ giữ mã, báo cáo và bảng đối chiếu; đường dẫn current_raw_evidence trong CSV thuộc gói ZIP.

V84 giữ sửa geometry L2 của V83; sửa replay cost/labels, warmup, sparse/error counters
và diagnostics SiLU. Raw vẫn là baseline; dual/executed là formulation thực nghiệm.
Không đổi learner equations hoặc hạ physics/safety thresholds.

Các cohort cuối seed44 gồm E64×300 raw/dual/raw_penalized, E64×900 dual và
E4096×3 raw/dual cho cả MATD3/MASAC. E64×900 dual: MATD3 có4 full15 successes
nhưng5 pair-safety failures; MASAC0 full15 và3 pair-safety failures.
Đây là online diagnostics, chưa phải held-out convergence, formal joint viability,
T4,2GPU hay fullE4096×900.
