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
4. Khi đã ghi đúng 10.000 episode, parent gửi CPU child `<root-ref>-cpu`.
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
