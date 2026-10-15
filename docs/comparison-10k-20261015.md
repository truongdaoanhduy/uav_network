# So sánh BASE và PQL: bốn account, 10.000 episode

Mỗi series dùng một account Kaggle riêng. Profile chung:
`configs/comparison-10k.yaml`. BASE giữ fixed-schedule collector/learner;
PQL bật collector và learner chạy song song có rate control.

| Tham số | Cả bốn run |
| --- | --- |
| UAV / target / obstacle | 6 / 25 / 15 |
| Backend / motion | warp_packet / dynamic_window |
| Episode horizon / dt | 1.000 bước / 1 giây |
| Env / seed | 4.096 / 44 |
| Episode budget | 10.000 episode đã hoàn thành |
| Hidden / batch / replay | 256×256 / 8.192 / 500.000 |
| Fixed update / PQL ratio | 8 update mỗi vector collection |
| Warmup | 131.072 joint transitions = 32 collections |
| Evaluation | 30 scenario từ seed 2.000.000.044 |
| Tracking | W&B online, group uav-network-base-pql-10k-seed44 |

Trong cùng một thuật toán, config chỉ đổi `pql.enabled`. MASAC và MATD3
dùng các hyperparameter riêng vốn có của thuật toán, cùng trong profile này.
Cả hai mode yêu cầu cùng loại session T4; BASE sử dụng GPU0 cho
collector/learner/replay, PQL dùng GPU0 collector và GPU1 learner/replay/policy.
Vì vậy so sánh tốc độ của cấu hình này bao gồm cách phân bổ hai GPU khác nhau.

## Tự chạy khi tắt máy

1. Saved script trên Kaggle chạy training và ghi W&B online.
2. Trước deadline an toàn, trainer lưu policy v2 và full training state v3;
   upload artifact có version cố định. Full state gồm weights/target networks,
   Adam, replay, RNG và counters.
3. Nếu dừng vì time budget, chưa đủ episode và có tiến bộ thật, parent tự gửi
   GPU child cùng account, source, config và tổng budget. Child khôi phục
   full state. W&B segment mới có `root_kernel_ref` và `continuation_index`.
4. Khi đã ghi đúng 10.000 episode, parent gửi CPU child có URL được ghi trong `cpu-visualization.json`.
   Slug được rút ngắn khi cần để vừa giới hạn tên; dùng URL trong manifest.
   CPU tải policy nhẹ, sinh PNG/MP4/metrics trong Output.
   Metadata CPU: `enable_gpu=false`, `enable_tpu=false`.

Không có local watcher. Máy người dùng chỉ gửi job và đọc trạng thái.
Parent gửi next job trước khi thoát; hai session có thể overlap ngắn lúc
submission, nhưng evaluate/render chạy trong CPU session.

Mỗi GPU segment kiểm tra quota, chừa 15 phút quota đệm và thời gian upload;
timeout session tối đa 39.600 giây. Thiếu quota, upload lỗi, checkpoint stale,
không có episode tiến bộ hoặc child submission lỗi đều ghi failure rõ ràng,
giữ summary/checkpoint. Flow không tự chờ tuần reset quota hoặc sửa mọi lỗi dịch vụ.

Resume khởi tạo episode simulator mới theo seed tiếp theo, giữ learner/replay
và RNG đã lưu. Các episode đang chạy khi segment dừng được bỏ dở.
Không cam kết trajectory bitwise giống một session liên tục.
Batch vector cuối có thể tạo thêm transition; chỉ ghi đủ episode prefix đã yêu cầu.

## Gửi một series

Ví dụ BASE MASAC, với dataset đã xác minh thuộc account được chọn:

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/comparison-10k.yaml --algorithm masac \
  --account account_08 --required-gpu-hours 12 --timeout-seconds 39600 \
  --credential-dataset fasfwejlklfv/uav-cpu-wandb-credential \
  --set pql.enabled=false --auto-continue
```

PQL dùng `--set pql.enabled=true`; MATD3 dùng `--algorithm matd3`.
Mỗi run phải chọn account/dataset của đúng chủ sở hữu.

Host hiện có đường IPv6 tới Google Storage bị timeout. Workaround tùy chọn
chỉ áp dụng vào process gửi job và CLI con, không đổi mạng toàn máy:

```bash
PYTHONPATH=outputs/kaggle_ipv4:. .venv/bin/python -m uav_network kaggle launch ...
```

Helper `outputs/kaggle_ipv4/sitecustomize.py` dùng urllib3 với IPv4.
Kaggle cloud script độc lập với helper local này.

## Đọc kết quả

Trong CPU session Output:
`visualization-result.json`, `hardware.json`,
`visualize/<algorithm>/final_episode.mp4`, `final_episode.png`, `metrics.json`.
Mỗi series có CPU URL riêng gắn với root, kể cả đã nối nhiều GPU segment.

So BASE với PQL của cùng thuật toán:
- Thời gian toàn series tới 10.000 episode, transitions/s, critic/actor updates/s.
- Số sample usages/transition, replay reuse/eviction khi có instrumentation,
  backlog, policy lag, GPU memory/utilization.
- Held-out discovery/delivery/success, completion time, delivered AUC,
  propulsion+RF energy, joules/delivery, safety violations.

Dùng cùng evaluation bank và paired scenario comparison trong
`uav_network compare` khi so hai checkpoint. CI theo scenario mô tả checkpoint
pair ở seed44; đánh giá độ ổn định của thuật toán cần thêm training seeds.
Tốc độ cao hơn và mission quality tốt hơn là hai kết quả cần đo riêng.
Các lượt smoke kiểm tra flow/checkpoint/device, không kết luận hội tụ.

## Benchmark thực tế ngày 15/10/2026

Kaggle [matched benchmark](https://www.kaggle.com/code/trteht/uavnet-benchmark-20261015-023236-779b9a)
đã hoàn tất 6/6 trường hợp trên hai Tesla T4. Mỗi case chạy trong process mới,
40 vector collections, 163.840 joint transitions và 72 critic updates.
MASAC có 72 actor updates; MATD3 có 36 do policy delay. Không giảm env,
batch hay replay để tránh OOM. Bảng đo từ khi collect đến lúc drain learner,
không gồm startup, evaluation, checkpoint hay upload.

| Thuật toán | Mode | Joint transitions/s | So BASE | Tổng điện hai card | So BASE |
| --- | --- | ---: | ---: | ---: | ---: |
| MASAC | BASE, GPU0 | 3.427,23 | — | 3.386,8 J | — |
| MASAC | PQL, GPU0 | 3.282,87 | −4,21% | 4.530,9 J | +33,78% |
| MASAC | PQL, GPU0+1 | 3.868,83 | +12,88% | 4.253,7 J | +25,59% |
| MATD3 | BASE, GPU0 | 7.289,89 | — | 1.629,8 J | — |
| MATD3 | PQL, GPU0 | 7.444,37 | +2,12% | 2.003,6 J | +22,94% |
| MATD3 | PQL, GPU0+1 | 9.172,56 | +25,83% | 1.741,1 J | +6,83% |

Đây là một lượt đo ngắn mỗi case, chưa có CI hay kết luận hội tụ.
Physical board energy gồm cả idle power của hai card được cấp; nó không
phải năng lượng riêng của PyTorch. Mỗi cửa sổ có power coverage 100%.
PQL một GPU không luôn nhanh hơn BASE; PQL hai GPU cần thêm tài nguyên
để đạt tốc độ này. Peak memory, utilization, device roles và số đo đầy đủ:
[comparison-10k-t4-20261015.json](benchmarks/comparison-10k-t4-20261015.json).

## Kiểm chứng flow và lỗi preflight

Hai smoke MATD3 BASE/PQL hoàn thành đúng 7 episode dù num_envs=4,
tự tạo CPU session, tải đúng immutable policy artifact và sinh MP4 H.264
800×800 đọc được bằng ffprobe. CPU dùng torch 2.11.0+cpu,
CUDA unavailable và enable_gpu=false. Bằng chứng:
[cpu-handoff-smoke-20261015.json](benchmarks/cpu-handoff-smoke-20261015.json).

Smoke MASAC ép dừng theo time budget đã chứng minh GPU child tự khôi phục
optimizer/replay/RNG/counters: BASE tiến 16→32→48→64 episode;
PQL tiến 28→56→80. Simulator mở episode mới khi resume như mô tả ở trên.
Trạng thái cuối của smoke và bốn lượt train chính được ghi riêng trong báo cáo chạy.

Lượt production BASE đầu tiên dừng trong benchmark preflight trước khi
mở W&B: benchmark dùng NumPy random_actions cho observation tensor trên CUDA.
Sửa preflight dùng WarpVectorEnv.random_actions và reset mask đúng device,
giống trainer/legacy comparison benchmark. Regression chạy collect, reset
và update cả MASAC/MATD3 trên Warp CPU, đồng thời chạy lại trên CUDA trong
preflight Kaggle. Không đổi trainer, simulator, reward hoặc config train.
