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

Finite/gradient checks và chuyển metrics thành số CPU chạy theo log interval,
kèm kiểm tra tham số khi lưu. Distribution của policy dùng các tham số đã
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
  --account account_02 \
  --credential-dataset haibro1234/uav-wandb-credential-arch-ablation \
  --timeout-seconds 1800

# Train 1000 episode mỗi thuật toán; chỉnh bằng --set nếu cần.
.venv/bin/python -m uav_network kaggle launch --stage train \
  --account account_02 --required-gpu-hours 4 \
  --credential-dataset haibro1234/uav-wandb-credential-arch-ablation

.venv/bin/python -m uav_network kaggle status \
  --manifest outputs/kaggle/<job-folder>/launch.json
```

Launcher kiểm tra quota và dataset credential riêng tư thuộc đúng account;
source tree phải được commit/push trước vì job ghim chính xác SHA của code.
Mặc định tự chọn account còn nhiều giờ nhất nếu bỏ `--account`; dataset phải
thuộc account được chọn. Output ghi URL Kaggle và hai URL W&B riêng.
W&B dùng entity/project trong `key.txt`, ghi progress, reward, tìm kiếm, delivery,
pin, safety, packet/delay/throughput, loss và artifact checkpoint cuối.
W&B cũng theo dõi CPU/GPU/VRAM; tắt bằng `tracking.system_metrics=false` khi cần.
Chạy local với `--set tracking.enabled=true` cũng dùng key này.

Profile Kaggle dùng **2 env CPU cho mỗi learner, batch512 và replay CUDA**.
Với T4 ×2: MASAC ở GPU0, MATD3 ở GPU1, train độc lập; một GPU thì chạy lần lượt.
Job dài chạy CUDA regression và benchmark config thật trước khi mở learner;
Kaggle output giữ `cuda-tests.log`, hardware và benchmark JSON của hai thuật toán.
Đây là hai thí nghiệm, không phải chia một learner lên hai GPU.
Kernel private lấy W&B key từ dataset credential riêng tư đã có; toàn bộ
`key.txt` chỉ giữ ở local. Chưa hỗ trợ resume optimizer/replay khi hết thời gian
Kaggle. Xem [kaggle-wandb-plan.md](docs/kaggle-wandb-plan.md) cho kiểm chứng.

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
