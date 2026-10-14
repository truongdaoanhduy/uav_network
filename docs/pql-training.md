# PQL-inspired training cho UAV MASAC/MATD3

## Đã triển khai
Đây là MASAC/MATD3 với kiến trúc huấn luyện lấy ý tưởng từ PQL; không đổi tên thuật toán thành PQL.
Giữ nguyên actor riêng từng UAV, CTDE, hybrid action, receiver mask, reward, sensing, safety, energy và packet simulation.
Default runner và `configs/motion-gpu.yaml` vẫn là control cố định 512 env.

Ba thành phần:
- Collector sở hữu Warp và actor chỉ dùng inference.
- Critic worker sở hữu replay, online/target critics và critic Adam.
- Policy worker sở hữu actors, alpha/Adam của MASAC và critic snapshot chỉ đọc.
Dùng hai worker thread và CUDA stream riêng; không thêm Ray hoặc một CPU process cho mỗi GPU env.

Collector tự chạy cho tới khi hàng đợi có giới hạn đầy. Critic chạy trên một replay do chính nó quản lý.
Policy chạy theo các sự kiện critic hoàn tất. Queue và feedback từ completed counters tự chặn bên nhanh hơn khi bên chậm chưa theo kịp.
Đây là biến thể dùng credit/backpressure; không copy bộ điều khiển sleep theo wall time của PQL nguyên xi.
Queue collection mặc định tối đa hai batch đang chờ, cộng một batch đang xử lý. Policy cũng có giới hạn tương tự.
Transition không bị drop; snapshot clone có version; CUDA hoàn tất publication trước khi consumer đọc.
Async policy/critic snapshot có độ trễ hữu hạn theo queue. Mỗi minibatch actor vẫn giữ đủ local observation, centralized state và joint-action context.

## Cấu hình mục tiêu
`configs/motion-pql.yaml`:
| Giá trị | Cấu hình |
|---|---:|
| UAV / targets / obstacles | 6 / 25 / 15 |
| Max steps | 1000 |
| Worlds | 4096 |
| Replay joint rows | 500000 |
| Batch target | 8192 |
| Warmup vector collections | 32 |
| Critic updates / eligible collection | 8 |
| Critic / policy MASAC | 1 |
| Critic / policy MATD3 | policy_delay = 2 |
| gamma / tau / learning rate | .99 / .005 / .0003 |
| Train / eval seed | 44 / 10044 |

Horizon cố định là một vector step. Trong PQL mode, `pql.warmup_collection_steps` và `pql.critic_sample_ratio` điều khiển learning.
`learning_starts` và `updates_per_step` là tham số của fixed runner; giá trị trong profile chỉ diễn tả cấu hình fallback.
Không cộng update credit cho dữ liệu warmup trước khi replay đủ batch.
MASAC cập nhật actor+alpha theo mỗi critic event. MATD3 giữ exact policy-delay clock, target actor và target smoothing.
Không copy PQL nstep3, tau.05, AdamW, bỏ target actor hoặc thay target noise của PQL vào các baseline.

Với E worlds, batch B, U critic updates trên mỗi collection:
- Một replay row là một JOINT transition, không nhân thêm số UAV.
- Sample usages/new row = U*B/E.
- 4096 / 8192 / 8 cho target 16 usages/new row.
- Baseline 512 / 512 / 16 cũng cho 16 usages/new row.
Vì vậy PQL không tự tăng reuse so với control hiện tại; lợi ích cần đo là pipeline, backpressure và throughput.

Một transition scenario hiện tại khoảng 9832 bytes; replay500k khoảng 4.58 GiB raw, chưa gồm model/Adam/activation/queues/metadata.
Replay5M của PQL gốc sẽ khoảng45.8GiB ở scenario này nên không copy.
Replay500k tại4096 worlds giữ khoảng122 vector steps; Bellman1-step không yêu cầu giữ đủ1000 bước của cả episode.

## Chạy và benchmark
T4 batch8192 là candidate, chưa được khẳng định chỉ bằng config hoặc CPU tests.
Benchmark giữ scenario thật, report allocated/reserved peak VRAM, rates, queue wait, snapshot lag, finite learner metrics và lỗi OOM.
`cuda_pipeline_verified` chỉ xác nhận pipeline CUDA của case đã đo; `target_t4_scenario_verified` đòi hỏi đúng4096/B8192/scenario25/15 trên haiT4 và mapping0→1.
Benchmark không thể pass nếu chưa hoàn tất cả critic lẫn policy updates.
Batch fallback phải được yêu cầu rõ qua `--batches`; chương trình không tự đổi batch sau OOM.

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_pql.py \
  --algorithms masac matd3 --envs 4096 --batches 8192 --steps 40 \
  --output outputs/pql-benchmark/report.json
```

Chỉ sau benchmark đúng hardware pass:
```bash
.venv/bin/python -m uav_network train --config configs/motion-pql.yaml \
  --algorithm masac --set runtime.output_dir=outputs/pql-masac
.venv/bin/python -m uav_network train --config configs/motion-pql.yaml \
  --algorithm matd3 --set runtime.output_dir=outputs/pql-matd3
```

Hai GPU: collector CUDA0, replay/critic/policy CUDA1. Một GPU: auto chọn CUDA0 cho cả ba, là functional fallback cần benchmark riêng.
Explicit `pql.collector_device/learner_device/policy_device=cuda:N` được kiểm tra và không silently fallback CPU.
Kaggle launcher PQL giữ cả hai GPU cho từng thuật toán; `both` chạy sequential, không partition thành một GPU cho mỗi algorithm.
Cloud `stage=train` chạy gate PQL benchmark trước khi train. `stage=benchmark` chỉ đo, không chạy production:
```bash
.venv/bin/python -m uav_network kaggle launch --stage benchmark \
  --config configs/motion-pql.yaml --algorithm both \
  --credential-dataset OWNER/VERIFIED_PRIVATE_WANDB_DATASET \
  --required-gpu-hours 1 --timeout-seconds 3600
```

## Checkpoint và determinism
`final.pt` / `episode_*.pt`: model nhỏ cho evaluation, video và W&B.
`resume_latest.pt`: model, targets, actor/critic/alpha Adam, replay+sampling RNG, global learning RNG, counters, rate credit và progress.
Resume:
```bash
.venv/bin/python -m uav_network train --config configs/motion-pql.yaml \
  --algorithm masac --set runtime.warm_start_checkpoint=outputs/pql-masac/resume_latest.pt \
  --set runtime.episodes=8192 --set runtime.output_dir=outputs/pql-masac-resumed
```
Resume giữ learner/replay nhưng reset simulator với next unused seeds; không phải tiếp tục chính xác giữa episode.
`checkpoint_replay=false` giữ Adam/counters nhưng refill replay với warmup mới; metadata phải phản ánh điều đó.
Checkpoint được coalesce theo collection boundary để không lưu nhiều bản giống nhau khi hàng nghìn worlds kết thúc cùng lúc.
File resume lớn lưu cục bộ; chỉ final artifact thêm entry resume.pt một lần. Periodic artifacts chỉ chứa checkpoint.pt nhỏ.
Cloud --warm-start-artifact của PQL tải resume.pt; baseline tải checkpoint.pt. Profile dành timeout900s cho final upload; soft deadline chừa thêm thời gian.
Khi resume artifact cũ chỉ có policy v2, chương trình báo thiếu full state thay vì silently reset Adam/replay.

`pql.mode=synchronous` cho fixed-schedule regression. Test chạy lặp cùng seed xác nhận hashes model/outcome/settings giống nhau.
Async giữ seeded numerical/environment operations nhưng thứ tự thread có thể thay đổi. Không cam kết bitwise determinism cho async.

## Chẩn đoán thay vì đoán
Metrics mới gồm rates, actual critic/collector và critic/actor ratios, queued work, wait time, snapshot lag,
sample usages/new row, replay sample age và actual unsampled evictions.
Warmup nằm trong denominator sample-usage toàn run; tỷ lệ steady-state16 không có nghĩa metric phải bằng16 ngay ở collection40.

Tăng replay chỉ kéo dài residence time C/insertion_rate. Ở steady uniform replay, expected usages trước eviction xấp xỉ B*critic_rate/insertion_rate.
Nếu learner không theo kịp: throttle collector hoặc tăng learner throughput khả thi; thêm replay một mình không tạo thêm optimizer updates.
PQL paper cho thấy replay quay vòng nhanh vẫn học tốt trên các task họ thử, nên không thể kết luận overwrite là nguyên nhân duy nhất.
High reuse cũng có thể đi kèm Q divergence, gradient lớn, alpha instability hoặc primacy bias.
Giữ Q/TD/gradient/alpha, safety, receiver rejection, search-delivery gap và energy để kiểm tra.
Không thêm normalization, network resets, reward shaping hoặc n-step như những thay đổi đồng thời thiếu ablation.

## Nguồn primary
- PQL code: https://github.com/Improbable-AI/pql/tree/93c2b583b6e9999dd45c77f3f8342ed538d73fb5 .
- Paper: https://proceedings.mlr.press/v202/li23f.html ; buffer discussion §4.4.4, architecture §3.
- Primacy bias: https://proceedings.mlr.press/v162/nikishin22a.html .
- High-update value divergence: https://arxiv.org/abs/2403.05996 .
- CUDA stream/lifetime semantics: https://pytorch.org/docs/stable/notes/cuda.html .

## Verification
Pre-refactor oracle frozen at afeaa3d compares exact losses, weights, target networks, Adam and RNG for both critic implementations.
Tests cover stage ownership, warmup/fractional rate, delayed actor count, immutable snapshots, replay ring+unsampled eviction,
worker errors, full state restore, CPU Warp training, synchronous repeatability, target config, CUDA paths when available,
cloud GPU allocation and refusing failed benchmark.
CPU smoke proves functional behavior, not T4 capacity or policy convergence. Compare outcome vs transitions AND wall time after GPU validation.
