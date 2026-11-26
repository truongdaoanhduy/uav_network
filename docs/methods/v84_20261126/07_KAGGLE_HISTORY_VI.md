# Lịch sử Kaggle được giữ trong gói V84

Có 64 record archive/member trong bảng, với 32 SHA nội dung khác nhau. Đây không phải 64 lượt training độc lập: archive master chứa lại nội dung từ các ZIP V77/V78; job summary và kết quả con cũng có thể mô tả cùng công việc.

| Archive | Số record tìm được |
|---|---:|
| uav_v77_kaggle2_logs.zip | 14 |
| uav_v78_kaggle2_logs.zip | 18 |
| UAV_FULL_CODE_KAGGLE_ALL_LOGS_RUN_EXPLANATION_EVIDENCE_20261126.zip | 32 |

| Status giữ nguyên từ bằng chứng | Record |
|---|---:|
| PASS | 48 |
| METADATA | 4 |
| DIAGNOSTICS_ONLY_NOT_FULL_RL | 2 |
| CAPACITY_PROBE_PASS_NOT_FULL_RL | 4 |
| FUNCTIONAL_PASS_NOT_FULL_RL | 2 |
| EXPERIMENTAL_INDEPENDENT_WARP_ROLLOUT_AND_LEARNER_PASS_NOT_PARITY | 4 |

Đọc `08_KAGGLE_HISTORICAL_RECORDS.csv`: mỗi dòng có archive, member, status, returncode và SHA. Các record có returncode được ghi đều bằng 0. Bảng không chứng minh mọi job trên tài khoản Kaggle từng chạy đều thành công; nó chỉ tổng hợp các bằng chứng hiện có trong các archive được tìm thấy.

`kaggle_history/v77` và `kaggle_history/v78` giữ notebook và raw logs. `history_archives` giữ nguyên ZIP V77, V78, V79 và archive master. `history_v82_reports/RELEASE_SOURCE_PROVENANCE.json` giữ SHA gói cũ; SOURCE_PROVENANCE.json hiện tại giữ SHA các archive.

Capacity probe, functional learner smoke và experimental independent Warp rollout không được đổi nhãn thành full training hoặc SimPy parity. Toàn bộ 30 cohort V83/V84 hiện tại chạy trên RTX 3060 Laptop ở Vast; không có job Kaggle/T4 mới, không dùng credential để tạo lượt chạy trùng lặp.
