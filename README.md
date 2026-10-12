# UAV network: swarm tìm target trong 2D

Project mới, không import notebook hay module của `uav_search_target`.
Mặc định **6 UAV đồng nhất**, 10 target, 8 vật cản tròn, map 1000 × 1000 m.
UAV tìm target bằng camera FOV, tạo report và truyền về GCS qua UavNetSim thật.
MASAC dùng 6 actor riêng, observation cục bộ và critic tập trung.

## Cấu trúc và thứ tự đọc

| Folder/file | Trách nhiệm |
| --- | --- |
| `configs/` | Cấu hình experiment; `default.yaml` và lượt kiểm tra `smoke.yaml` |
| `uav_network/configuration.py` | Dataclass, giá trị mặc định, kiểm tra config/CLI |
| `uav_network/environment/world.py` | Sinh vị trí UAV, target, vật cản theo seed |
| `uav_network/environment/geometry.py`, `motion.py` | Chuyển động XY, giới hạn tốc độ, chặn va chạm trên cả đoạn đường |
| `uav_network/environment/sensing.py` | FOV camera và che khuất |
| `uav_network/environment/observations.py` | Observation từng UAV và state cho critic |
| `uav_network/environment/env.py` | Luồng `reset → motion → sensing → network → reward` |
| `uav_network/communication/reports.py` | Report, buffer, TTL, quyền giữ report |
| `uav_network/communication/upstream.py`, `routing.py` | Tích hợp upstream, CSMA/CA, ACK và next hop do policy chọn |
| `uav_network/communication/backend.py` | Fragment, công suất phát, relay, GCS và metrics |
| `uav_network/algorithms/` | Actor/critic, replay và cập nhật Hybrid MASAC |
| `uav_network/experiments/` | CLI, train, evaluate CPU và ảnh quỹ đạo |
| `tests/` | Kiểm thử môi trường, mạng thật và thuật toán |
| `scripts/` | Dependency mạng và commit upstream đã ghim |
| `docs/` | Thiết kế, kế hoạch và bằng chứng kiểm tra |
| `third_party/`, `outputs/`, `.venv/` | Dữ liệu tải về/kết quả cục bộ; được Git bỏ qua |

Đọc lần lượt `configuration → world → motion → sensing → reports → network
→ observations → env → models → replay → masac → training` để hiểu từng khối.
Mỗi file do project quản lý dưới 500 dòng.

## Action mỗi UAV

| Thành phần | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `continuous[:,0]` | `[-1,1]` | Gia tốc X, nhân `world.max_accel_mps2` |
| `continuous[:,1]` | `[-1,1]` | Gia tốc Y |
| `destination` | Integer `0..6` | `0`: silent; `1..5`: các UAV khác theo ID tăng dần; `6`: GCS |
| `continuous[:,2]` | `[-1,1]` | Công suất phát, ánh xạ sang `0.1..0.4 W` |

Ví dụ UAV 2 có các đích `[silent, UAV0, UAV1, UAV3, UAV4, UAV5, GCS]`.
Tự động điều chỉnh số đích khi đổi `world.num_uavs`.
Chuyển động chỉ XY. Độ cao camera cố định 100 m, FOV đầy đủ 90°,
footprint tròn có bán kính `altitude * tan(FOV/2)` = 100 m.
Không có belief map. Actor không nhận tọa độ target nằm ngoài FOV.

Observation gồm vị trí/vận tốc bản thân, hướng tương đối tới GCS, thời gian,
target đã biết, tuổi report đang giữ, target/vật cản trong FOV và beacon UAV
lân cận. Beacon là giả định control plane lý tưởng; report target đi qua mạng
packet. Critic được dùng trạng thái mission thật trong lúc train.

## Cài đặt

```bash
cd /home/aduy/Documents/NCKH/uav_network
bash setup.sh
```

Máy đã có PyTorch và các dependency lớn có thể dùng:

```bash
bash setup.sh --system-site-packages
```

Script tạo venv riêng, tải những folder runtime cần thiết của UavNetSim,
ghim commit trong `scripts/upstream-revision.txt`. Không tải frontend hay ảnh,
không cần dataset target: môi trường sinh scenario trực tiếp theo seed.
Kênh analytical A2A chạy trên CPU; không cần Sionna ray tracing.
Một process chỉ giữ một simulator mạng đang sống vì upstream dùng config global.
`setup.sh` tải đúng commit đã ghim. Nếu tự đổi `network.upstream_path` sang một
checkout khác, cần tự bảo đảm checkout đó cùng commit; loader hiện tin đường
dẫn source đã cung cấp.

## Chạy theo từng phần

```bash
# In config đã resolve, chưa train.
.venv/bin/python -m uav_network config --config configs/default.yaml

# Các kiểm thử, bao gồm truyền mạng thật và Adam cập nhật hai nhánh action.
.venv/bin/python -m unittest discover -s tests -v

# Lượt nhỏ: 6 UAV, 2 episode, 40 step/episode; kiểm tra toàn bộ luồng.
.venv/bin/python -m uav_network train --config configs/smoke.yaml

# Evaluate checkpoint trên CPU, dùng đúng scenario config trong checkpoint.
.venv/bin/python -m uav_network evaluate \
  --checkpoint outputs/smoke/final.pt --episodes 1
```

Train đầy đủ hoặc đổi scenario chỉ bằng config/args:

```bash
.venv/bin/python -m uav_network train --config configs/default.yaml \
  --set world.map_size_m=3000 \
  --set world.num_targets=50 \
  --set world.num_obstacles=30 \
  --set world.max_steps=3000 \
  --set runtime.episodes=20000 \
  --set runtime.output_dir=outputs/paper-run
```

Kết quả gồm `resolved_config.yaml`, `episodes.jsonl`, `summary.json`,
`final.pt`; evaluation thêm `metrics.json` và `final_episode.png`.
Output folder phải trống để tránh ghi đè experiment cũ. Checkpoint hiện dùng
để evaluate; không chứa replay/RNG/optimizer để resume chính xác.

## Những quy ước cần biết

- Discovery và delivery là hai metrics riêng; thành công khi mọi target đến GCS.
- Report chỉ đổi quyền giữ sau khi đủ fragment. TTL giữ timestamp nguồn gốc
  qua các relay; phát thất bại vẫn tiêu hao năng lượng.
- Action chọn next hop tại lúc enqueue. Packet đang ở MAC queue vẫn tiếp tục
  chạy nếu bước sau chọn silent. Không dùng greedy routing thay action.
- Queue poll 1000 µs được cấu hình bằng `network.queue_poll_us`; MAC backoff,
  retry, SINR, interference và ACK vẫn do UavNetSim xử lý.
- Reset tạo observation/detection ban đầu. Trong mỗi step: di chuyển và sensing
  ở đầu khoảng mạng, chạy network `dt_s`, rồi tăng clock và kiểm tra TTL.
- MASAC là bản thích nghi cho action lai Gaussian + categorical, dùng kỳ vọng
  chính xác trên các đích của actor đang cập nhật và sample action UAV khác.
  Sáu actor và sáu cặp critic riêng; ReLU MLP `(256,256)`, Adam và soft update.
- Lượt smoke xác nhận code chạy và có gradient; không đánh giá hội tụ policy.

Tham khảo [UavNetSim](https://github.com/Zihao-Felix-Zhou/UavNetSim),
[DroneWatch](https://github.com/Bussler/DroneWatch) và
[GPMF](https://github.com/yangmin32/GPMF). Chi tiết trong `docs/design.md`.
