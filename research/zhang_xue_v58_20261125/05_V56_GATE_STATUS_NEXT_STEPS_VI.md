# Kiểm định v5.6 — Tổng kết theo Gate, 25/11/2026

## Kết luận trung thực

Đây là **mã nghiên cứu đã có Warp kernel CSMA/CA 7-radio thực thi**, KHÔNG còn chỉ là 80 mảng Warp. Tuy nhiên **chưa phải pipeline GPU hoàn chỉnh cho MATD3/MASAC E=4096**. Hai mệnh đề này khác nhau và phải được giữ nhất quán trong mọi báo cáo/thuyết minh.

## Gate 1 — CPU 7-radio / 6 UAV + 1 GCS

- Có mạng sự kiện gói tin `src/uav_v54_7radio_csmaca_sim.py`, có MAC ACK, E2E ACK, Q64, TTL10s, anti-duplicate, blackout/3D attenuation, relay.
- 80 bước seed=20261125: zero-comm 0 DATA GCS, full-comm 474 DATA GCS, 948 MAC ACK, 474 E2E ACK, xác nhận 1/15 TP, coverage đã nhận 6.5%, return -47.81 vs -77.63. Không xem là thành tích MARL hay so trực tiếp với v5.4 gốc khác động lực/reward.
- Regression trong ZIP: 35 PASS, 2 SKIP (hai test Warp cần có `warp-lang` trong container dựng ZIP); các bài kiểm thử Warp CPU thật đã chạy ở sandbox máy người dùng và được ghi log riêng.

## Gate 2 — Actor Observation → DS+AoI → Reward

**Actor:** 6 x 514 = 6 x (14 local telemetry + 500 local map) ; 5 kênh 10x10: DS target/empty/unknown, source-generation AoI, vị trí UAV. Không lộ target truth / GCS global ledger vào Actor. Nhưng compression 20x20 -> 10x10 chưa có ablation.

**DS:** Hợp đồng `zhang_max` theo Zhang 2024 Eq16 so với `sum` theo origin, không cộng gói retransmit thành observation mới; Dempster normalization ổn định. Chế độ `paper_fixed` (0.8/0.8), `sensor_likelihood_ratio` chỉ là bản đề xuất cần kiểm chuẩn thống kê. Mass vacuous (0,0,1) KHÔNG phải Bayesian prior P(target)=0.

**AoI:** `exp(-eta*delay)` trên source snapshot khi đến receiver, KHÔNG phải per-observation buffer aging như Xue 2026 Eq9. Chỉ số toàn bản đồ `mean_gcs_aoi_mapwide_censored_s` dùng quy ước age=t cho ô chưa nhận. Đã loại bỏ tên gây hiểu sai “unbiased”.

**Reward:** `r_peer_information` chỉ tính evidence thực nhận và mới tại peer, cap 12 cells/step. `r_gcs_tp` và `r_gcs_aoi` chỉ khi GCS xác nhận thật; action SEND không tự sinh reward. Weight chưa được ablation. Không tự nhận đã tối ưu reward.

## Gate 3 — Warp CPU-backed 7-radio packet simulation

- Các kernel thực thi trong `src/v56_warp_*`: 7 radio, state FSM, CCA/DIFS/backoff/freeze, DATA, ACK, queues, HELLO routing, multi-hop, provenance/DS ledger, E2E ACK. 1 Warp lane quản lý 1 environment; không dùng Python loop từng packet để tráo làm GPU.
- `warp-lang 1.18.0` trên sandbox, CPU backend: **1-hop 1 GCS DATA + 2 MAC ACK + 1 E2E ACK**; không overflow.
- **2 environments** với 2-hop `0 -> 1 -> 6` trong world0 và world1 mất kết nối: 1 GCS DATA, 4 MAC ACK, 1 E2E ACK; không hòa belief trên relay; world1 không nhận nhầm; reset riêng world0; `exit code 0`.
- Tìm và sửa hai lỗi quan trọng nhờ compiler + test: (a) HELLO khởi đồng bộ khiến hidden terminals 0 và 6 xung đột bền vững, cần stagger 2ms/radio; (b) kernel reset pending `%7` trong khi có 6 actor gây OOB khi E>=2, crash lúc free; sửa `%6`, test sạch.
- Các kernel reset/inject/event đã biên dịch và chạy trên Warp CPU; module event compile mới lần đầu tốn ~69s CPU. Không có driver CUDA trên sandbox: không được viết “đã chạy T4”. E=32 Warp CPU scaling cũng đã thử với giới hạn 18s và 75s nhưng đều timeout (không có output hoàn tất), **chưa có số đo throughput**; test E=2 PASS không chứng minh E=4096 phù hợp.

## Các điểm chưa đạt: KHÔNG được mở Gate production

1. **Parity CPU–Warp:** CPU HELLO truyền snapshot của packet, Warp đọc current neighbor DV tại receive. Còn khác đường random/backoff và PHY, không thể kỳ vọng bitwise match. Phải khóa event model thống nhất theo immutable HELLO payload, đối chiếu byte/timestamp nguồn, receiver, MAC ACK, queue và first-divergence traces.
2. **Bài 3D NLoS hiện mới ở CPU:** Warp nhận ma trận channel gain; chưa có toàn bộ building occlusion/motion update thành GPU kernel. Chưa có parity blackout và 3D building geometry.
3. **Hàng loạt trường hợp biên mạng:** collision hidden terminal, capture, mất MAC ACK, retransmit, loop, TTL, packet dedup overflow, queue full, E2E after expiry, 1,500s clock overflow, per-world RNG.
4. **Bay/động học:** veto chuyển động đưa vận tốc về 0 tức thì, có thể vượt gia tốc giới hạn a_max=5 m/s². Nên dùng dynamic window/reachable stopping set và test trajectory, đồng bộ cost pin/RF/idle/RX; chưa thể tuyên bố bảo đảm an toàn vật lý.
5. **T4 và 4096 worlds:** chưa profile allocator/throughput; mạng SoA ~0.56MB/env chỉ là lower bound. Với 2048 env/T4 ~1.07GiB mảng state không gồm replay và optimizer. 1M-transition replay `obs_dim=514` dù bỏ state dup vẫn lớn; phải đo memory từng GPU, chọn capacity thực tế, batch và chunk kernels trước training.
6. **Training:** chưa nối thực sự Warp `env.step()` với actor->GCS belief observation/replay->learner trên T4; user code `src/train_v55_packet_cpu.py` chỉ là bounded gradient CPU smoke. Hiện chưa chạy trainer production, checkpoint/W&B/HF/eval định kỳ trên 4096 GPU envs.
7. **Kaggle:** người dùng có đề nghị dùng account2; từ môi trường này không có CUDA, thao tác dùng credentials với Kaggle bị chặn bởi công cụ. Không vượt qua bước xác thực, không tuyên bố có GPU/T4 result.

## Tiêu chuẩn nghiệm thu tiếp — không được bỏ qua

A. Warp event1hop, 2hop, disconnected, hidden-terminal, queue/TTL, reset E2, E32/E256/E2048: xác nhận test log, process exit=0.

B. Same-seed CPU–Warp **parity** per-hop: compare packet generation time, src/dst/seq, queue transitions, MAC retry, application ACK, source evidence; trace first divergence, fix before retry.

C. NVIDIA T4 `scripts/v56_cuda_preflight.py --device cuda:0 --envs 32 --allow-experimental`: compile, run, record allocation and timings. Tăng 256/1024/2048 chỉ nếu pass và VRAM còn an toàn.

D. Nối nguyên bộ: GPU geometry + sensing 2D + provenance DS/AoI + network7 + TX power + battery + action mask + reward + MATD3/MASAC + replay gradient. Kiểm định closed-loop random actions và 100 steps trước 1500, sau đó mới chạy 20k episodes.

E. Đánh giá fixed holdout seeds, 50/100 episodes; riêng MAC metrics: GCS delivery %, discovery %, coverage %, AUC, AoI mapwide, latency, retransmit, queue drop, safety blocks, energy, mean return. So sánh `No-Comm`, `Ideal-Sync`, Zhang-max DS, AoI-off và v5.6.

**Không được tự hạ yêu cầu Gate bằng cách thay UART/packet network thành oracle graph hoặc không đo CUDA mà gọi là GPU training.**

## Kịch bản tích hợp GitHub

ZIP cung cấp `stage_in_repo.sh`, yêu cầu checkout v5.3 sạch, tạo nhánh `experiment/zhang-xue-v56-sevenradio-gate3-20261125` và stage mã *mà không thay* nhánh gốc. Trong phiên làm việc này **chưa commit/push** nhánh mới vì code ZIP nằm trong container dựng artifact độc lập với sandbox máy người dùng. Không khẳng định đã tích hợp nếu chưa có commit GitHub rõ ràng.
