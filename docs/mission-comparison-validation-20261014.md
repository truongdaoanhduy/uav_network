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

## PQL GPU→CPU handoff đã kiểm tra trên Kaggle

Source `62153252fecf0ced3602b60bd78f4073dc824dca`.
- GPU parent complete: https://www.kaggle.com/code/haibro1234/uavnet-validate-20261014-172327-596fc5
- CPU child complete: https://www.kaggle.com/code/haibro1234/uavnet-validate-20261014-172327-596fc5-cpu
- [Evidence machine-readable](benchmarks/pql-cpu-handoff-20261014.json).

Mỗi thuật toán:4 episodes×8 steps,4 env, batch8, replay256, warmup2/rate2,
hidden32×32. CollectorCUDA0; critic/replay/policyCUDA1;0 CPU env workers.
Cả hai32 joint transitions,14 critic updates; MASAC14 actor, MATD3 7 actor.
Đây là **functional smoke**, không phải production/convergence training.

Parent đã upload immutable W&B artifact:v0 và tự submit private CPU child.
API metadata của child:enable_gpu=false, chỉ một private W&B credential dataset;
hardware Torch2.11.0+cpu, CUDA unavailable. CPU download v2 checkpoint.pt nhẹ,
không tải replayv3, và giữ backend warp_packet chạy Warp CPU.

Cả hai có PNG và MP4 thật800×800,10fps,3 frames tại steps0/5/8. ffmpeg decode
thành công. Artifact MD5 và GPU-training model hash đều khớp checkpoint CPU.
Hai test scenes2000000044/2000000045, không trùng train/validation. Tạo
comparison.json/CSV với metrics mới. Với chỉ2 short scenes, không kết luận
MATD3 tốt hơn MASAC dù delivery/return của smoke khác nhau.

Sau parent submission thành công, train/upload/CPU submission/render chạy trên
Kaggle; không phụ thuộc laptop poller. Parent submit CPU trước khi parent thoát,
nên không bảo đảm strict nonoverlap giữa sessions.

Trong verification, selected-file Kaggle download đã timeout120s hai lần.
Đường archive của **private dataset được xác minh chỉ có đúng một credential
file** hoạt động; helper dùng đường này và reject archive có extra files.
Dataset nhiều file vẫn chỉ tải selected credential, không kéo toàn bộ bundle.

Sau fix archive, CPU full suite309 tests:263 pass,46 CUDA skips,146.023s.

## W&B online

Benchmark có tracking tắt chỉ trong timed cases để cùng instrumentation; actual
results đã upload online sau measurement và xác nhận hai runs finished:
- MASAC benchmark: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/51aada9f
- MATD3 benchmark: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/6c360084

PQL handoff training cũng online và completed:
- MASAC smoke: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/35dc7e61
- MATD3 smoke: https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/adb9d9ff
