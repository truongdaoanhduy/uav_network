# Kiểm tra GPU cùng config và metrics nhiệm vụ

Thời điểm job theo UTC:2026-10-14; source benchmark `fee499b82137819f8a75f62ed27d4404ce2e3e47`.
Kaggle: https://www.kaggle.com/code/haibro1234/uavnet-benchmark-20261014-170332-a09659 .
[Báo cáo máy đọc](benchmarks/training-gpu-comparison-20261014.json).

## Phép đo có kiểm soát

2×TeslaT4,14.562GiB/card, Torch2.11.0+cu128, CUDA12.8.
E4096,B8192,replay500000,N6,25 targets,15 obstacles,H1000,dt1s.
Warmup32 collections, rate8,40 collections và163840 joint transitions.
Mỗi case process riêng, tracking tắt trong measurement, giữ hidden256×256,
gamma/tau/lr/reward/physics và policy_delay. Toàn bộ6 cases passed, không OOM
và không giảm batch/capacity. MASAC72 critic/72 actor; MATD3 72 critic/36 actor.
Window gồm warmup+updates+drain và bỏ startup; không là steady-state timing.

| Thuật toán | Luồng | Thời gian(s) | Joint rows/s | Sampled VRAM max GPU0/GPU1(GiB) | Mean GPU util0/1(%) | Energy cả2 board(kJ) |
| --- | --- | --- | --- | --- | --- | --- |
| MASAC | legacy | 46.78 | 3503 | 11.04 / 0.00 | 93.8 / 0.0 | 3.31 |
| MASAC | pql-single | 46.29 | 3540 | 13.56 / 0.10 | 95.4 / 0.0 | 4.16 |
| MASAC | pql-dual | 40.58 | 4037 | 2.10 / 11.61 | 38.7 / 66.6 | 3.98 |
| MATD3 | legacy | 21.60 | 7586 | 7.19 / 0.00 | 95.6 / 0.0 | 1.49 |
| MATD3 | pql-single | 20.32 | 8062 | 8.16 / 0.10 | 96.3 / 0.0 | 1.74 |
| MATD3 | pql-dual | 17.64 | 9285 | 2.10 / 6.19 | 82.8 / 28.9 | 1.62 |

So với fixed legacy, PQL dual đạt throughput MASAC cao hơn15.27% và MATD3
cao hơn22.40% trong **một lượt đo ngắn này**. Không suy ra mức tăng cố định
cho mọi GPU/config hoặc chất lượng policy.

PQL single dùng nhiều VRAM hơn fixed legacy vì worker replicas/snapshots và
in-flight batches. PQL dual chuyển replay/learner sang GPU1, collectorGPU0 còn
khoảng2.1GiB sampled usage; đừng cộng các peak ở thời điểm khác nhau thành
simultaneous total peak.

GPU utilization là trung bình lấy mẫu của từng board, không phải chất lượng
model hoặc tỷ lệ điện năng dùng hiệu quả. Torch allocator peak/reserved được
giữ riêng trong JSON; Warp allocations không nằm hoàn toàn trong Torch counters.

Power.draw lấy mẫu được tích phân trapezoid, coverage100% ở cả6 cases; đây là
ước tính board Joule trong window, không phải đo tại ổ điện hoặc energy UAV.
Các GPU0-only cases vẫn có2 card vật lý trong session; PQL single quan sát card1
rảnh khoảng26W so với10–12W của legacy. Do đó bảng không dự đoán chi phí của
máy chỉ có một GPU. Tổng energy PQL dual cao hơn legacy khoảng20.5%(MASAC)
và8.7%(MATD3), dù wall time giảm. PQL có mục tiêu cân bằng collector/learner,
không bảo đảm tiết kiệm điện.

## Các kiểm tra đã chạy

CPU full suite tại source benchmark:307 tests,261 passed,46 CUDA skips,
148.380s. T4 gates:10 CUDA,8 mission,9 PQL tests đều passed.
Matched comparison kiểm tra đủ requested algorithms×3cases, completed update
counts và actual CUDA roles; runner interrupted không được giả thành success.

Các cases40 collections chưa đầy replay500000. Chưa đo long-run eviction
hoặc convergence. Không xếp MASAC/MATD3 theo rows/s thành chất lượng thuật toán.

## Code và protocol mới

Xem [hướng dẫn](training-flow-comparison-20261014.md) và
[7 nguồn nghiên cứu](algorithm-comparison-research-20261014.md).

CPU evaluation có first-observed target discovery/GCS delivery arrays, resolution
dt; capped delivery/AUC không bỏ targets thất bại. Safety đo actual swept
trajectory. UAV energy gồm propulsion+UAV RF, pooledJ/delivery tính cả failures.
Packet PDR vẫn là per-hop fragments; delay successful-only luôn có denominator.

CLI compare có same-scene seed pairing, nominal95% scenario-bootstrap CI,
train/validation seed exposure checks và JSON/CSV export. Giữ training seed44;
không gọi nhiều test episodes là nhiều independent training runs.
