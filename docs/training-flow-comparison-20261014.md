# So sánh luồng train, UavNetSim GPU và đánh giá thuật toán

Ngày 2026-10-14. Nhánh triển khai: `pql-async-20261014`.
Mốc trước PQL: `afeaa3dba27edd736be13f84b602f84c268e3a22`.
Nghiên cứu metrics và bảy paper: [algorithm-comparison-research-20261014.md](algorithm-comparison-research-20261014.md).

## Code trước và sau PQL

| Thành phần | Trước PQL | PQL-inspired hiện tại |
| --- | --- | --- |
| Mô phỏng mission | Warp packet backend hoặc native đã chọn trong config | Cùng backend và quy tắc |
| Vòng điều phối | Collect → replay → combined learner update, có overlap khi bật | Collector, critic worker, policy worker có hàng đợi giới hạn |
| Phân bổ trên 2 GPU | Một learner cùng env/replay/model trên một GPU; hai thuật toán có thể chạy độc lập | GPU0 collector; GPU1 critic/replay/policy; hai thuật toán chạy nối tiếp |
| Update | Actor/critic/alpha/targets trong combined wrapper | Tách ownership/update stage, giữ equations và target clocks |
| Giới hạn data | Fixed updates_per_step | Credits tính từ completed collections, queue backpressure |
| MASAC actor schedule | Mỗi critic update | Mỗi critic update |
| MATD3 actor schedule | policy_delay=2 | policy_delay=2 |
| Resume | Policy warm start; rebuild Adam/replay | v3 lưu Adam/replay/RNG/controller; env reset bằng seed chưa dùng |
| Determinism | Cùng phép toán và thứ tự cố định | Async không bảo đảm bitwise; synchronous debug có kiểm tra |

Các tests frozen legacy update so sánh losses, model/targets, Adam và RNG để
giữ toán học cũ. Tách lịch chạy không giữ nguyên toàn bộ quá trình học:
policy snapshots có độ trễ, batch/RNG order và trajectories có thể khác.

Với E env, B batch và U critic updates mỗi collection, sample usages trên mỗi
joint transition mới xấp xỉ U×B/E sau warmup. Đây là số lần dùng sample trung
bình, không bảo đảm từng transition đều được sample.

- E4096/B512/U16: 2 usages/row.
- E4096/B8192/U8: 16 usages/row.
- Capacity500000/E4096 giữ khoảng122.07 collections; tăng capacity riêng lẻ
  không tăng tỷ lệ usages.
- Collector bị chặn khi critic/policy tụt sau giới hạn queue. PQL không bảo đảm
  reward hoặc delivery luôn tốt lên.

## GPU port so với UavNetSim nguyên bản

Upstream được pin tại `04daafb815eb377409b40b285574eeb62b9a8d58`.
Repo gốc: https://github.com/Zihao-Felix-Zhou/UavNetSim .
Paper gốc: https://arxiv.org/html/2507.09852v1 .

Đây là backend packet được viết lại bằng NVIDIA Warp, đối chiếu với **mission
bridge native**. Không phải nguyên SimPy code chạy trên CUDA. Upstream đã có
SionnaRT; profile nhiệm vụ chọn analytical A2A trên cả hai backend.

| Phần | Phạm vi tương đương / khác biệt |
| --- | --- |
| CSMA/CA | CCA, DIFS/backoff freeze, DATA/ACK, retry, airtime, collisions/interference |
| Packet/report | FIFO, fragments/reassembly, custody, receiver admission, packet lifetime và report TTL |
| PHY profile | LOS/NLOS urban, selected power, noise/SINR, obstacle attenuation, link cache |
| Scheduler | Thay SimPy callbacks bằng heap timestamp/priority/insertion-order và explicit state |
| Song song | Một thread xử lý radio tuần tự trong một world; nhiều world chạy song song |
| Ngoài profile | Không port toàn bộ RT modes, calibration, multicast/broadcast và routing protocols nguyên bản |
| Mission | Sensing sinh report, action chọn next hop; movement XY, altitude cố định |
| Flight energy | Mô hình mission idle+hover+v²+a²; không thay bằng toàn bộ Zeng rotor model |
| Capacity | Fixed device arrays; overflow báo lỗi rõ ràng |
| Precision | Geometry/time/energy FP64; observation/state/reward FP32 |

Tests parity bao phủ các cases đã kiểm tra và numerical tolerances; không chứng
minh mọi configuration hay tất cả ordered event traces đều tương đương.
Xem [bằng chứng GPU packet](gpu-packet-results-20261014.md).

Tăng tốc thực sự dùng:
- Warp 1.18 JIT kernels cho world/motion/sensing/packet/reports/observations.
- Device arrays, CUDA action/observation/replay; không tạo CPU simulator workers.
- `wp.from_torch/wp.to_torch` views cùng storage, Torch/Warp stream ordering.
- Batched actor inference và `torch.func.vmap` cho các critic riêng biệt.
- PQL worker threads/CUDA streams, immutable snapshots và bounded queues.
- Idle poll coalescing khi radio hoàn toàn rảnh.

`torch.compile`, fused optimizer và CUDA graph capture không bật trong profile
motion-pql. Không dùng Numba/Triton/JAX/VMAS/WarpDrive để chạy environment này.
Python vẫn launch kernels, reset seed và đọc scalar diagnostics; không phải
mọi tác vụ trên host đều đã biến mất.
Warp interop docs: https://nvidia.github.io/warp/stable/user_guide/interoperability/pytorch.html .

## Benchmark GPU với cùng config

Chạy trong worktree:
`/home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014`.

```bash
.venv/bin/python scripts/compare_training_gpu.py \
  --config configs/motion-pql.yaml --algorithms masac matd3 \
  --steps 40 --output outputs/gpu-comparison/report.json
```

Hoặc cloud pinned source, chỉ benchmark:

```bash
.venv/bin/python -m uav_network kaggle launch --stage benchmark \
  --compare-legacy --config configs/motion-pql.yaml --account account_02 \
  --required-gpu-hours 1 --timeout-seconds 3600 \
  --credential-dataset haibro1234/uav-wandb-credential-masac50k-d59860c
```

Mỗi algorithm có legacy/single-GPU-PQL/dual-GPU-PQL trong process mới.
E4096/B8192/C500000, warmup32/rate8,40 collections:72 critic updates,
MASAC72 actor, MATD3 36 actor. Window gồm warmup+learning+drain, loại startup;
không phải steady-state hoặc convergence experiment.

Torch peak allocated/reserved chỉ đếm allocator Torch và bỏ sót Warp.
`gpu_resources` lấy tất cả board vật lý bằng nvidia-smi: sampled VRAM và GPU
utilization; power.draw được tích phân theo khoảng sample để ước tính Joule.
Ghi coverage, không ngoại suy hoặc thay power unavailable bằng0; không tích
phân qua poll gap dài hơn max(1s,4×sample_interval). Tổng board
có thể gồm process khác. GPU Joule hoàn toàn khác năng lượng UAV mô phỏng.
OOM giữ nguyên batch/capacity và ghi rõ failed case; không hạ batch âm thầm.

## Metrics mới và cách dùng

```bash
.venv/bin/python -m uav_network compare \
  --left /path/to/masac/final.pt --right /path/to/matd3/final.pt \
  --episodes 30 --seed 2000000044 --animate \
  --output-dir outputs/mission-comparison
```

Kết quả: `comparison.json`, `comparison.csv`, metrics/PNG và tùy chọn MP4/GIF
của mỗi bên. Hai checkpoint phải cùng world/sensor/network/energy/reward/backend
và cùng test seeds. Policy deterministic, không cập nhật weights.

| Metric | Diễn giải | Mong muốn |
| --- | --- | --- |
| gcs_delivery_fraction | Mục tiêu riêng biệt báo đầy đủ về GCS / K | cao |
| target_discovery_fraction | Mục tiêu ít nhất một UAV biết / K | cao |
| delivery_auc | Diện tích delivery theo thời gian / H; failures tính tới deadline | cao |
| target_delivery_time_capped_mean_s | Mean first GCS observed time; mục tiêu chưa giao dùng H | thấp |
| mission_success | Toàn bộ K mục tiêu đã báo về GCS | cao |
| mission_completion_time_capped_s | Thời gian success; failure dùng H kể cả hết pin sớm | thấp |
| pooled_uav_j_per_gcs_delivery | Tổng propulsion+UAV RF qua tất cả episodes / tổng delivered targets | thấp cùng delivery tương đương |
| safe_episode và swept violation counts | Kiểm tra actual movement segments, peer/obstacle/boundary thresholds | an toàn, ít violation |
| report_delay_delivered_mean_s | Delay của report thắng, chỉ các deliveries; null khi không có | thấp, phải xem cùng delivery |
| packet_pdr_fraction | Packet theo hop, gồm UAV relay; không gọi là report/GCS PDR | metric mạng phụ |

Per-target timestamps mới được **quan sát ở boundary của step CPU evaluation**,
độ phân giải dt (1s trong profile). Không ghi timestamp packet chính xác hoặc
sửa simulator state. Delay packet/report gốc vẫn giữ độ phân giải của network.
Đây là phương án ít thay đổi nhất, không tăng overhead GPU training.

Seed2000000044 là bank final-test khởi đầu được chọn cho CLI comparison;
không mặc định coi nó disjoint. Protocol kiểm tra conservative training range
từ seed/record cũ nhỏ nhất tới next_seed-1, và bank validation định kỳ trong
saved config. Default visualize seed10044 vẫn được giữ; trùng train/validation
hoặc thiếu lịch sử exposure khi resume thì comparison ghi exploratory. Thiếu
progress không được diễn giải rằng test bank chắc chắn chưa được dùng.

Paired bootstrap dùng cùng indices scenario cho hai checkpoint, báo mean
difference và nominal95% CI từng metric. Conditional metrics ghi số episode
đủ dữ liệu; zero delivery không thành zero delay/Joule. CI chứa0 được ghi
inconclusive. Không ghép các metric khác đơn vị thành một winner tự động.
Các CI chỉ mô tả scenario variation của checkpoint pair hiện tại; train seed44
vẫn được giữ. Muốn kết luận độ ổn định của **thuật toán** cần các training runs
độc lập, không xem30 test episodes như30 training seeds.

## GPU train → CPU visualize khi tắt máy

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/motion-pql.yaml --account account_02 \
  --required-gpu-hours 22 --timeout-seconds 39600 \
  --credential-dataset haibro1234/uav-wandb-credential-masac50k-d59860c \
  --set runtime.eval_episodes=30 --set runtime.eval_seed=2000000044 \
  --cpu-visualize
```

`--required-gpu-hours` là ngưỡng quota tối thiểu để chọn account, **không phải**
giới hạn thời gian22 giờ; thời gian thực phụ thuộc job/session.

Sau khi parent submission thành công, Kaggle chạy train, W&B upload immutable
checkpoint version và submit CPU child trên server; laptop có thể tắt.
CPU child có enable_gpu=false, tải checkpoint.pt nhẹ, dùng backend đã lưu
(`warp_packet` chạy Warp CPU), sinh video/metrics và comparison khi đủ hai bên.
GPU v3 resume.pt đầy đủ vẫn lưu riêng trong artifact, không cần tải để render.

Parent phải submit CPU child trước khi parent thoát. CPU có thể bắt đầu lúc
GPU parent còn đang kết thúc; chưa có bảo đảm strict nonoverlap giữa session.
Không cần laptop poller để thực hiện handoff.

Các lỗi submission/upload/render được ghi rõ và giữ lại training results.
Đã sửa mất summary/handoff của wave trước khi wave sau lỗi deadline/startup;
CPU credential JSON giờ chỉ nhận api_key, không mang các secret không liên quan.

## Kết quả thực tế đã lưu

[Báo cáo resource cùng config và handoff đã xác minh](mission-comparison-validation-20261014.md).
Benchmark6/6 T4 cases passed; PQL dual throughput tăng15.27% MASAC và22.40%
MATD3 trong lượt đo ngắn. Energy cả2 board vẫn cao hơn legacy.
PQL GPU→CPU smoke cũng completed và tạo decoded MP4/PNG/paired metrics.
