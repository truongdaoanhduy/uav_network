# UAV network: tìm kiếm target, pin và truyền report

Mặc định **6 UAV đồng nhất**, map 1000 × 1000 m, 10 target và 8 vật cản tròn.
Chuyển động XY, camera FOV, pin hữu hạn, lựa chọn relay/GCS và công suất phát.
Report đi qua **UavNetSim thật**: CSMA/CA, packet, ACK, retry và SINR.
Train bằng **Hybrid MASAC hoặc Hybrid MATD3**, với 6 actor riêng và critic tập trung.

## Cài đặt và kiểm tra

```bash
cd /home/aduy/Documents/NCKH/uav_network
bash setup.sh
# Nếu máy đã có PyTorch và dependency lớn:
# bash setup.sh --system-site-packages
.venv/bin/python -m uav_network hardware
.venv/bin/python -m unittest discover -s tests -v
```

`setup.sh` tạo venv, cài dependency và tải sparse checkout UavNetSim đã ghim
commit. Scenario được sinh theo seed; không cần tải dataset target.
`third_party/`, `.venv/`, `outputs/` và cache được Git bỏ qua.

## Cấu trúc và thứ tự đọc

| Folder | Chức năng |
| --- | --- |
| `configs/` | Default, smoke, smoke song song và workstation |
| `uav_network/configuration.py` | Dataclass, validate và YAML/CLI overrides |
| `uav_network/environment/` | World, XY/safety, pin, FOV, observation và Gym adapter |
| `uav_network/communication/` | Report/buffer/TTL, native PHY budget, NLOS và selected routing |
| `uav_network/algorithms/` | Actor/critic, MASAC, MATD3, tensor replay, checkpoint |
| `uav_network/experiments/` | Train, evaluate, hardware, benchmark và ảnh quỹ đạo |
| `uav_network/cloud/` | Đọc key, kiểm tra quota, tạo và theo dõi job Kaggle |
| `tests/` | Môi trường, packet thật, learner, worker/reset và CUDA |
| `scripts/` | Dependency mạng và revision upstream |
| `docs/` | Thiết kế, quá trình thực hiện, bằng chứng và nghiên cứu GPU |

Đọc theo thứ tự: `configuration → world → motion/energy → sensing → reports
→ communication/backend → observations/env → models → masac/matd3 → vector
→ replay → training`. Mỗi file project không quá 500 dòng.

## Action và observation

| Thành phần action mỗi UAV | Ý nghĩa |
| --- | --- |
| `continuous[:,0:2]`, trong `[-1,1]` | Gia tốc X/Y, nhân `world.max_accel_mps2` |
| `destination`, integer `0..N` | 0: silent; 1..N−1: peer theo ID tăng dần; N: GCS |
| `continuous[:,2]`, trong `[-1,1]` | Công suất phát, ánh xạ sang min/max watt trong config |

Với 6 UAV, UAV2 có danh sách `[silent, UAV0, UAV1, UAV3, UAV4, UAV5, GCS]`.
Policy chọn một next hop; không có greedy route thay quyết định của policy.
Mask cho phép silent, peer còn pin trong range và GCS trong range khi có report.
Sau khi phát hiện target, observation bước tiếp theo cho phép policy chọn đích.
Peer đang đầy buffer/đã biết report có thể từ chối; sender giữ custody để thử lại.

Actor nhận vị trí/vận tốc bản thân, GCS tương đối, thời gian, buffer, **pin và
active**, target đã biết/tuổi report, target/vật cản trong FOV, beacon peer và mask.
Target ngoài FOV không lộ tọa độ cho actor. Không có belief map.
Beacon vị trí/pin là control plane lý tưởng; report dùng mạng packet.
Critic train được thấy mission state đầy đủ. Default observation/state là 95/217.

## Pin và vật cản

Mỗi UAV có 277200 J = 77 Wh, chỉnh bằng `energy.battery_capacity_j`.
Năng lượng bay một bước:

`(idle_power + hover_power + c_v * ||v_real||² + c_a * ||a_real||²) * dt`.

Vận tốc/gia tốc dùng chuyển động thực tế sau xử lý safety. Nếu không đủ pin
cho trọn bước, UAV đứng tại chỗ, dùng hết phần pin còn lại và ngừng hoạt động.
Safety được kiểm tra lại với UAV vừa dừng. Đây là mô hình năng lượng theo bước;
không mô phỏng động lực học hạ cánh hoặc sạc pin.

Mỗi lần native PHY phát trừ `power_w * frame_bits / bit_rate_bps`, gồm retry,
frame lỗi và ACK. Frame không đủ pin không được phát. UAV hết pin không bay,
quan sát, gửi/nhận hoặc giữ vai trò relay mới. GCS dùng nguồn điện riêng.
Metrics tách năng lượng bay, RF của UAV, RF của GCS và số lần phát bị chặn do pin.
Hết pin cả swarm kết thúc thất bại; mọi target về GCS kết thúc thành công.

Safety mặc định dùng `world.motion_safety=dynamic_window`: giữ lệnh an toàn,
kiểm tra cả đường phanh tới khi dừng, giảm tốc/đi vòng tiếp tuyến khi cần.
Không dùng lực đẩy APF đẩy UAV xa target. Bộ lọc không nhận tọa độ target;
actor vẫn có FOV/occlusion và action/observation cũ. Exact swept guard được
giữ cho trạng thái bất khả thi hoặc peer bị pin buộc dừng.
Chọn `world.motion_safety=reject` để so với cơ chế hủy bước cũ; checkpoint
thiếu field này tự giữ reject khi evaluate lịch sử.

Console từng episode ghi pin, termination, motion và receiver counts.
Xem [nghiên cứu và kiểm chứng avoidance](docs/obstacle-avoidance-single-seed.md)
cho target sát obstacle, giới hạn phương pháp và so sánh checkpoint cùng seed.
Script `scripts/compare_motion_safety.py` so sánh policy đóng băng;
`scripts/check_motion_repeatability.py` lặp native smoke hai thuật toán.
Profile `configs/motion-study.yaml` dùng fresh seed44/100 episode cho follow-up.

Vật cản được chuyển thành building của airspace UavNetSim. Link rõ dùng LOS
free-space, link bị che dùng NLOS urban nên suy hao và có thể lỗi SINR.
`network.obstacle_extra_loss_db` thêm loss cho NLOS, default 0 để tránh cộng
trùng với urban loss. Vật cản áp dụng cho UAV→UAV và UAV→GCS.
Link NLOS vẫn được policy thử: kết quả phụ thuộc công suất, nhiễu và SINR.

## Chạy từng phần

```bash
# Xem config, chưa train.
.venv/bin/python -m uav_network config --config configs/default.yaml

# Smoke một env; chỉ kiểm tra luồng/gradient.
.venv/bin/python -m uav_network train --config configs/smoke.yaml \
  --algorithm masac --set runtime.output_dir=outputs/masac-smoke
.venv/bin/python -m uav_network train --config configs/smoke.yaml \
  --algorithm matd3 --set runtime.output_dir=outputs/matd3-smoke

# 2 env × 6 UAV, có reset từng worker.
.venv/bin/python -m uav_network train --config configs/smoke-parallel.yaml \
  --algorithm matd3 --set runtime.output_dir=outputs/matd3-parallel-smoke

.venv/bin/python -m uav_network evaluate \
  --checkpoint outputs/matd3-parallel-smoke/final.pt --episodes 3
```

Output folder phải trống. Kết quả gồm resolved config, JSONL từng episode,
summary, policy checkpoint; evaluate thêm metrics và PNG có pin từng UAV.
Checkpoint v2 ghi algorithm/config/dimensions, dùng để evaluate. Thêm pin/mask
đã thay observation: checkpoint v1 cần train lại. Chưa có resume optimizer/replay.

## Chạy trên CPU mạnh + GPU

`runtime.num_envs=1` dùng một env. Khi >1, Gymnasium **AsyncVectorEnv + spawn**
chạy một swarm hoàn chỉnh trong mỗi tiến trình CPU; mỗi process có UavNetSim riêng.
Một learner ở parent dùng CUDA cho actor/critic/target/Adam và có thể giữ replay
trên GPU. Sáu actor suy luận từng batch `[num_envs, observation_dim]`; action chỉ
chuyển về CPU một lần mỗi batch. `step_async` cho CPU mô phỏng trong lúc learner
cập nhật từ replay trước đó. Batch cuối được cập nhật trước khi lưu checkpoint.
SimPy/packet simulation vẫn chạy trên CPU; không chuyển thành CUDA env.

`runtime.coalesce_idle_polls=true` mặc định gộp tick kiểm tra queue rỗng trong
bước hoàn toàn không có traffic/MAC/ACK/PHY đang chờ. Giữ chu kỳ polling và tick
cuối trước ranh giới bước; khi radio bận hoặc clock/poll có phần lẻ, chạy lịch cũ.
Sáu actor vẫn có weights, gradient clipping và Adam riêng; kiểm tra finite của
nhóm gradient một lần trước khi cập nhật bất kỳ actor nào.
`runtime.fused_optimizers=true` tùy chọn dùng fused Adam trên CUDA, CPU dùng Adam
thường. Có thể khác rounding với Adam cũ; đo trên đúng GPU trước khi chọn.
`runtime.vectorize_critics=true` gom sáu critic bằng `torch.func.vmap`; mỗi
critic vẫn có weights/gradient riêng. Lấy weights hiện tại mỗi forward và truyền
gradient về leaf gốc, giữ checkpoint/Adam cũ. Chế độ này thay thứ tự phép tính
ma trận FP32 nên so sánh theo tolerance; repeated-seed được kiểm tra riêng.
Chọn `compile_models=false` khi bật `vectorize_critics`.
Không đổi reward, mô phỏng native, batch, replay hay số update để tăng utilization.

```bash
# Đo cùng cấu hình, đủ warmup, không train dài.
PYTHONPATH=. .venv/bin/python scripts/benchmark_acceleration.py \
  --label optimized --device cuda --output outputs/speed/optimized.json
```

Driver lặp cùng root seed44 để lấy median, không phải nhiều research seeds;
pipeline probe dùng deadline384/warmup512 được ghi riêng trong report.
Nguồn phương pháp: [nghiên cứu CPU/GPU](docs/gpu-throughput-options-20261013.md).

Profile `configs/motion-study-fast.yaml` bật critic vmap, giữ nguyên các tham số
khoa học của `motion-study.yaml`; chỉ đổi execution mode, output và tracking tag.
Trên T4, vmap tăng thêm throughput pipeline ngắn 15,3%/26,0% (MASAC/MATD3) so
với bản đã tối ưu collector. Bản mặc định tăng 18,8%/37,2% so với snapshot bàn
giao; đây là hai phép đo riêng. CUDA source kiểm thử có 139/139 tests pass.
Chi tiết cấu hình, hashes, giới hạn và job:
[training-acceleration-20261013.md](docs/training-acceleration-20261013.md).

```bash
# Máy có CUDA; với Kaggle dùng profile này trong lệnh launch hiện có.
.venv/bin/python -m uav_network train --config configs/motion-study-fast.yaml \
  --algorithm masac --set runtime.output_dir=outputs/motion-study-fast-masac
```

Chọn số env theo benchmark trên **đúng máy train**:

```bash
.venv/bin/python -m uav_network benchmark --config configs/workstation.yaml \
  --env-counts 1 2 4 8 16 --steps 128 --learner-updates 20 \
  --output outputs/workstation-benchmark.json

.venv/bin/python -m uav_network train --config configs/workstation.yaml \
  --algorithm masac --num-envs 16 --device cuda \
  --set runtime.output_dir=outputs/workstation-masac
.venv/bin/python -m uav_network train --config configs/workstation.yaml \
  --algorithm matd3 --num-envs 16 --device cuda \
  --set runtime.output_dir=outputs/workstation-matd3
```

16 env nghĩa là **16 scenario độc lập, mỗi scenario có 6 UAV**, không phải chia
6 UAV của một scenario thành 16 process. Workstation profile là điểm bắt đầu:
batch1024, replay200000, warmup8192, 4 update/lần collect. Giảm env nếu quá tải CPU;
đổi `runtime.replay_device=cpu`/giảm replay khi VRAM không đủ. CPU replay→CUDA dùng
pinned samples; CUDA replay→CPU dùng copy blocking để CPU đọc dữ liệu đã hoàn tất.

`updates_per_step` là số update cho mỗi **vector collection**, còn `global_step`
đếm joint transitions: E env đóng góp E transition/lần. Ratio danh nghĩa là
`updates_per_step * batch_size / num_envs`. Default 1×256/1 và workstation
4×1024/16 đều là 256; summary ghi ratio thực tế sau warmup.
Số episode hoàn tất có thể vượt ngân sách tối đa E−1 do nhiều worker kết thúc
cùng lượt; summary ghi cả số yêu cầu lẫn số thực tế.

Gradient không hữu hạn bị chặn trước mọi bước Adam. Scalar loss/metrics chạy
theo log interval, kèm kiểm tra tham số khi lưu. Distribution của policy dùng các tham số đã
clamp/mask để giảm validation sync trên GPU. Optional `runtime.compile_models=true`
compile phần MLP, default tắt; benchmark sau warmup và so với eager.
`runtime.deterministic=false` là lựa chọn tốc độ cần so sánh riêng với baseline.

Trên máy CUDA, chạy thêm:

```bash
.venv/bin/python -m unittest discover -s tests -p test_cuda.py -v
```

Các test CUDA báo skip rõ ràng nếu host không có CUDA. Xem kết quả CPU thực tế
và phạm vi đã kiểm tra trong [verification-v02.md](docs/verification-v02.md).

## Kaggle CUDA và W&B online

`key.txt` giữ nguyên schema project cũ, đã được chép vào root project mới.
File này được Git bỏ qua. Cài thêm CLI/SDK bằng:

```bash
bash setup.sh --system-site-packages --cloud
.venv/bin/python -m uav_network kaggle accounts

# Kiểm tra CUDA và smoke hai thuật toán trước khi train dài.
.venv/bin/python -m uav_network kaggle launch --stage validate \
  --account account_07 \
  --credential-dataset bduy2378/uav-wandb-credential-arch-ablation

# Train 1000 episode mỗi thuật toán; chỉnh bằng --set nếu cần.
.venv/bin/python -m uav_network kaggle launch --stage train \
  --account account_07 --required-gpu-hours 22 --timeout-seconds 39600 \
  --credential-dataset bduy2378/uav-wandb-credential-arch-ablation

.venv/bin/python -m uav_network kaggle status \
  --manifest outputs/kaggle/<job-folder>/launch.json

# Thiết lập/cập nhật Personal Workspace mặc định với đúng metric.
.venv/bin/python -m uav_network dashboard --config configs/kaggle.yaml \
  --run-ids <masac-run-id> <matd3-run-id>
# Không cần chạy lại khi có run mới: workspace tự lấy các run mới nhất.
# --run-ids chỉ kiểm tra run và đặt số đường hiển thị; không ghim run.
# Chỉ thêm --workspace-url <saved-url> khi chủ động muốn sửa một Saved view.
```

Launcher kiểm tra quota và dataset credential riêng tư thuộc đúng account;
source tree phải được commit/push trước vì job ghim chính xác SHA của code.
Mặc định tự chọn account còn nhiều giờ nhất nếu bỏ `--account`; dataset phải
thuộc account được chọn. Output ghi URL Kaggle và hai URL W&B riêng.
W&B dùng entity/project trong `key.txt`, ghi progress, reward, tìm kiếm, delivery,
pin, safety, packet/delay/throughput, loss và artifact checkpoint định kỳ/cuối.
W&B cũng theo dõi CPU/GPU/VRAM; tắt bằng `tracking.system_metrics=false` khi cần.
Chạy local với `--set tracking.enabled=true` cũng dùng key này.

Profile Kaggle dùng **2 env CPU cho mỗi learner, batch512 và replay CUDA**.
Với T4 ×2: MASAC ở GPU0, MATD3 ở GPU1, train độc lập; một GPU thì chạy lần lượt.
Job dài chạy CUDA regression và benchmark config thật trước khi mở learner;
Kaggle output giữ `cuda-tests.log`, hardware và benchmark JSON của hai thuật toán.
Đây là hai thí nghiệm, không phải chia một learner lên hai GPU.
Kernel private lấy W&B key từ dataset credential riêng tư đã có; toàn bộ
`key.txt` chỉ giữ ở local. Hỗ trợ policy warm-start với model/target/update counter và lịch sử episode;
Adam, replay và simulator được khởi tạo lại. Không phải full-state resume. Xem [kaggle-wandb-plan.md](docs/kaggle-wandb-plan.md) cho kiểm chứng.
Sau khi submit thành công, Kaggle và W&B chạy trên máy chủ; có thể tắt laptop.
Mỗi learner lưu policy mỗi 10 episode theo profile Kaggle. Một learner lỗi không
dừng learner còn lại; kết quả train được lưu trước evaluate, kể cả khi evaluate lỗi.
Personal Workspace mặc định dùng `global_step`, lọc group `uav-network-2d`,
không ghim run và có 18 biểu đồ. Mặc định hiển thị hai run mới nhất; run train mới
tự xuất hiện trong bố cục đã lưu, không cần chọn/copy Saved view sau mỗi lần chạy.
Mỗi lần train online in link Personal Workspace trong console và lưu
`wandb_workspace_url` trong `wandb_run.json`/kết quả train; summary W&B cũng có
`workspace_url`. Các lần train dùng cùng W&B entity/project/group trong config.
Lệnh `dashboard` mặc định cập nhật workspace của người sở hữu API key; chỉ sửa
Saved view khi truyền `--workspace-url`. Các bố cục được lưu ở W&B nên không cần
ghi lại chúng ở mỗi lần train.
Lỗi masked entropy khi alpha > 1 được kiểm tra bằng regression CPU/CUDA;
zero-probability terms được xóa trước nhân alpha, gradient luôn kiểm tra finite.
Lịch sử job, URL W&B và số liệu CUDA thực tế:
[verification-online-training.md](docs/verification-online-training.md).
Snapshot trước khi sửa lỗi masked entropy:
[verification-kaggle-wandb.md](docs/verification-kaggle-wandb.md).

## Thuật toán và tài liệu

MASAC dùng Gaussian/tanh cho XY/power và categorical có mask cho next hop.
Power chỉ tính entropy khi truyền; dùng kỳ vọng chính xác trên đích của actor
đang cập nhật, sample UAV khác. MATD3 dùng clipped double Q, continuous target
smoothing, actor/target update trễ và categorical straight-through softmax.
Gaussian exploration cho continuous, epsilon greedy trên đích hợp lệ.
Đây là các bản thích nghi cho action lai của mission này.

GPU research so sánh PyTorch, Gymnasium, TorchRL, EnvPool, Sample Factory,
Isaac Lab và Warp: [gpu-acceleration-research.md](docs/gpu-acceleration-research.md).
Thiết kế: [design.md](docs/design.md); quá trình: [progress.md](docs/progress.md).
Lượt smoke chứng minh wiring/gradient, không chứng minh policy đã hội tụ.

Tham khảo [UavNetSim](https://github.com/Zihao-Felix-Zhou/UavNetSim),
[DroneWatch](https://github.com/Bussler/DroneWatch), [GPMF](https://github.com/yangmin32/GPMF)
và [MATD3](https://arxiv.org/abs/1910.01465).

## Tiếp tục sau giới hạn Kaggle

Job mặc định tối đa 39.600 giây; profile train chủ động dừng sau 36.000 giây
để còn thời gian lưu/upload/evaluate. Với một GPU, ngân sách chia cho các lượt.
Mục tiêu episode là tổng tích lũy, không phải số episode thêm.
Policy upload mỗi 10 episode và chờ artifact commit trước khi tiếp tục.
Summary local được ghi trước upload cuối; upload lỗi được báo riêng và kết quả giữ lại.

```bash
python -m uav_network train --config configs/kaggle.yaml   --warm-start outputs/checkpoint-recovery/masac/checkpoint.pt   --set runtime.algorithm=masac --set runtime.episodes=1000   --set runtime.output_dir=outputs/masac-continuation
python -m uav_network kaggle launch --stage train --account account_07   --required-gpu-hours 10.75 --timeout-seconds 39600   --credential-dataset bduy2378/uav-wandb-credential-arch-ablation   --warm-start-artifact masac=truongdaoanhduy-uit/uav_search_target/uav-network-masac-f8524504-recovered:v0   --warm-start-artifact matd3=truongdaoanhduy-uit/uav_search_target/uav-network-matd3-971920c8-recovered:v0
```

Phải refill replay bằng 4.000 transition mới trước khi update; checkpoint cũ
không chứa Adam/replay nên không thể tái tạo chính xác trajectory trước đó.
Chi tiết phục hồi: [verification-continuation.md](docs/verification-continuation.md).

## W&B diagnostics

Biểu đồ, cảnh báo từ dữ liệu thực, đánh giá cùng seed và cách đọc lỗi:
[training-diagnostics.md](docs/training-diagnostics.md). Cấu hình ở tracking.
Console được capture online; failure.json và failed_batch.pt giúp tái hiện lỗi.
Các run đang chạy ghim SHA cũ; metric mới áp dụng khi launch source mới.

## Completed single-seed training audit

The completed motion-study runs and executable failure reproductions are in
[the training audit](docs/completed-training-audit-20261013.md).
The mission's observed max_steps deadline is terminal for Bellman updates.
An external max_train_seconds cutoff remains a separate runner limit.

Fresh configs enable network.rediscovery_recovery and network.ack_uses_data_power.
After local TTL/buffer loss, a fresh visible target observation can regenerate
an undelivered report when space becomes available; relaying preserves TTL,
custody transfer does not generate another copy, and discovery is not rewarded
twice. The existing report-status feature is remaining TTL (>0), no custody (0),
or local loss needing a fresh camera observation (-1). Dimensions are unchanged.
ACK uses the received data frame's selected power, with real PHY battery/RF costs.
Legacy checkpoints missing these fields retain both original network behaviors.
Console also warns about intervention, saturation and stalls even with zero hard
blocks. Kaggle startup gates both CUDA tests and native mission regressions.

Existing policies still need fresh training and same-seed evaluation after these
fixes. Successful packet tests and finite CUDA updates do not prove convergence
or energy optimality.

## Train dài và visualize trong phiên CPU riêng

`configs/motion-long.yaml`: 500 episode tổng mỗi thuật toán, tối đa 1000
step/episode; giữ 2 env/learner và critic vmap của profile fast. Profile 100
episode cũ vẫn dùng cho đối chiếu. Giới hạn thời gian train 10 giờ có thể dừng
trước ngân sách episode; summary ghi rõ trạng thái.

```bash
.venv/bin/python -m uav_network kaggle launch --stage train \
  --config configs/motion-long.yaml --account account_02 \
  --required-gpu-hours 22 --timeout-seconds 39600 \
  --credential-dataset haibro1234/uav-wandb-credential-masac50k-d59860c \
  --cpu-visualize
```

GPU train/upload checkpoint W&B → tự submit kernel CPU riêng → PNG, MP4/GIF,
metrics. Lấy URL CPU trong `cpu-visualization.json` của output job GPU.
Chế độ này tắt evaluate giữa buổi và không render trong phiên GPU.
Xem [luồng, cấu hình và số liệu tài nguyên](docs/gpu-cpu-visualization-20261014.md).

Visualize checkpoint trên CPU local:

```bash
.venv/bin/python -m uav_network evaluate --checkpoint /path/to/final.pt \
  --episodes 1 --seed 10044 --animate --frame-stride 5 --fps 10 \
  --video-format auto --output-dir outputs/cpu-preview
```
