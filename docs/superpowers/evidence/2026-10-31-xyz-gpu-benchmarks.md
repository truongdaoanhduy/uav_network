# Đo CUDA XYZ BASE/PQL — 31/10/2026

Source của hai lần preflight: `9e7b88c52637fc5ae3571e3e567271a93cd0c965`.
Runtime struct-reference fix: `5fee6c753b4e7cd86801a68ab5a0fb72521bda86`.
Hai Tesla T4 thực;6 UAV/map1km/20T/15O/horizon1200, FP32, shared actor+critic,
E4096/B8192/replay1M/Hybrid75, learned RF và XYZ Bayesian sensing, visualize2D.
Eager/compile-off là override preflight; không đổi YAML gốc.

## Kết quả đã chạy thật

| Gate | BASE | PQL |
| --- | --- | --- |
| CUDA (gồm4XYZ tests) |15 passed/100.993s|15 passed/96.023s|
| Scenario CUDA |4 passed/2.615s|4 passed/2.370s|
| Mission regressions |8 passed/3.034s|8 passed/2.788s|
| PQL smoke |Không áp dụng|11 passed/25.931s|

[BASE Kaggle](https://www.kaggle.com/code/kesfljhk/ben-g3-both-base-h75-10301844-63825e) và
[PQL Kaggle](https://www.kaggle.com/code/kesfljhk/ben-g3-both-pql-h75-10301852-575209)
đều `complete`; từng report được đọc và kiểm `status=passed`, actual devices,
E/B/replay cùng update counters. Không chỉ dựa vào trạng thái job.

| Mode | Algorithm | Collector steps | Measured critic/learner updates | Total actor updates |
| --- | --- | ---: | ---: | ---: |
| BASE | MASAC |4|4 (cộng2 warmup)|6|
| BASE | MATD3 |4|4 (cộng2 warmup)|3|
| PQL | MASAC |40|72|72|
| PQL | MATD3 |40|72|36|

Bốn test XYZ chạy thật kiểm Native-vs-CUDA camera/motion/belief/observation,
confirmation chỉ ở độ cao thấp, finite-height RF, exact next-draw snapshot và
masked reset. PQL learning metrics không có nonfinite flag khác0.
Đây là benchmark, không phải run20k hoặc bằng chứng hội tụ; chưa hoàn tất episode1200bước.

W&B đã được API xác nhận `finished`, không phải URL dự kiến từ receipt:
[MASAC PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/4d0c32a3),
[MATD3 PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/d342ba5a).
BASE benchmark không tự mở W&B run; không diễn giải hai BASE IDs trong receipt là run online.

## Timing và VRAM

BASE isolated learner throughput: MASAC1.6463 update/s; MATD3 6.8368 update/s.
PQL critic rate theo wall-time gồm collection: MASAC1.5032; MATD3 3.5012 update/s.
Hai mẫu số khác nhau và budgets4/40 khác nhau; không dùng bảng này để kết luận
PQL hay BASE tốt hơn hoặc suy thẳng runtime20k.

Collector ởcuda:0; BASE learner ởcuda:1; PQL critic/policy ởcuda:1.
Replay có750k CPU slots +250k hot slots trêncuda:1. Allocated replay12,985,000,000B
(gồm measurement arrays); Warp world2,849,393,696B.
Peak allocated PyTorch trên learner GPU: BASE MASAC4,032,137,728B,
MATD3 4,076,023,296B; PQL MASAC4,743,710,720B, MATD3 4,186,487,296B.
Số peak PyTorch không gồm Warp-owned allocations hoặc driver context; không gọi
nó là total peak VRAM. World arrays nằm trên collector GPU riêng.

BASE `cuda_pipeline_verified=false`/`dual_gpu_verified=false` được giữ nguyên:
định nghĩa hiện có đòi configured replay device thuần CUDA, trong khi Hybrid
có phần CPU và configured replay_device=cpu. Actual collector/learner dùng hai GPU.
PQL `cuda_pipeline_verified=true` kiểm ba vai compute; không yêu cầu mọi byte replay ở GPU.
`target_t4_scenario_verified=false` là stamp của scenario cũ25T/15O/1000bước/R500k,
không phải stamp cho profile XYZ hiện tại. Không sửa các cờ này thànhtrue.

## Kiểm tra tiếp phần replay thực sự được dùng

BASE mới ghi16,384 rows; PQL ghi163,840 rows, đều chưa chạm mốc750k CPU slots.
Cấp phát đầy capacity1M và learner update đã kiểm được; việc dùng hot quarter
và wrap toàn ring còn cần phép thử dài hơn. Dự kiến256 collector rounds:
1,048,576 inserted rows, vượt capacity1M, vẫn giữ E/B/replay/FP32.

Phát hiện cloud PQL trước đây bỏ qua `--benchmark-steps` lớn: yêu cầu256 nhưng
child command chỉ nhận40. Regression test RED40!=256, sau sửa GREEN5tests.
Commit `08676528d52b578063039379ede4eab9d7217292` cho phép budget yêu cầu lớn hơn
mức warmup tối thiểu; không đổi warmup, learning rate, algorithm hoặc model.
Full bare pytest sau sửa:1146 passed,102 skipped,8 warnings,1557subtests,258.23s.
Lượt full-ring tiếp theo sẽ dùng commit có sửa gate này, ghim cùng account_01,
kiểm quota/job trùng trước khi submit. Chưa submit20k production tại mốc này.

[Structured measurements](2026-10-31-xyz-gpu-benchmarks.json) |
[Compiler diagnosis](2026-10-31-xyz-cuda-preflight.md).

Full-ring readiness sẽ kiểm BASE/PQL cho cả3profile. Riêng PQL diagnostic dùng
`pql.warmup_collection_steps=248` để điền ring trước phần learner; training profile
vẫn warmup32. Không so throughput diagnostic này như một training study ngang budget.
