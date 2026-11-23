# Đặc tả Kịch bản & Kiến trúc Hệ thống (Chuẩn Q1/A* — Cách B: Slotted Vectorized UavNetSim trên NVIDIA Warp + Vanilla MATD3/MASAC)

## 1. Mục tiêu cốt lõi & Sợi dây liên kết bài báo
Thay vì trục đánh đổi yếu về vật lý (*"relay để tiết kiệm pin RF"* — nơi RF chỉ chiếm <0,0002% pin bay), kịch bản này khóa chặt mọi thành phần vào một bài toán nhất quán:
> **6 UAV đồng nhất tìm kiếm 25 mục tiêu tĩnh ẩn trong khu đô thị 2.5D có tòa nhà che khuất sóng NLOS và chuyển tiếp đa chặng (multi-hop relay) về trạm mặt đất GCS đặt ở góc bản đồ, sao cho cực đại hóa số mục tiêu tươi (AoI thấp) giao về GCS trong khi tiết kiệm năng lượng bay cánh quay và bảo đảm an toàn bay.**

## 2. Toàn bộ Thực thể, Thuộc tính & Cấu trúc Bản đồ

| Thực thể | Số lượng | Thuộc tính trạng thái & thông số vật lý | Vai trò trong kịch bản |
| :--- | :--- | :--- | :--- |
| **Bản đồ đô thị 2.5D** | `1` | Kích thước `800 x 800 m`, chia lưới `20 x 20` ô (`40 m/ô`, 400 ô), `dt = 1.0 s`, horizon `T = 550 s`. | Không gian nhiệm vụ cân đối với tầm truyền LOS ~503 m; đường chéo ~1.131 m ép buộc multi-hop. |
| **UAV (Tác tử đồng nhất)** | `N = 6` | Vị trí `(x, y, z=50 m)`, vận tốc `(vx, vy)` (`v_max = 20 m/s`), gia tốc `(ax, ay)` (`a_max = 2 m/s^2`), pin `150 kJ` (Zeng 2019: bay treo `168 W` > bay hành trình `~132 W`), bán kính an toàn `15 m`, bán kính cảm biến `R_s = 60 m`, hàng đợi báo cáo FIFO tối đa 32 gói (`TTL = 120 s`), công suất phát cố định `P_tx = 0.1 W` (20 dBm). | Tự phân hóa vai trò động giữa trinh sát sâu trong vùng khuất sóng và làm cầu nối relay về GCS. |
| **Trạm mặt đất (GCS)** | `1` | Đặt cố định tại góc bản đồ `(40, 40, 10 m)`, chứa bản đồ niềm tin toàn cục & bảng khử trùng lặp `gcs_delivered[target_id]`. | Điểm đích cuối nhận báo cáo mục tiêu; đặt ở góc làm góc đối diện cách `~1.075 m > 2 * R_c`, buộc phải đi 2–3 hop. |
| **Tòa nhà đô thị (Obstacles)** | `22` | Hình trụ `(cx, cy, r, h)` với `r in [25, 45] m`, `h in [60, 120] m` (tất cả đều cao hơn trần bay `z = 50 m`), khoảng hở giữa tòa nhà `>= 35 m`. | Đồng thời là vật cản va chạm bay, vật cản che khuất cảm biến quang học và nguồn tạo bóng râm vô tuyến NLOS (`+20 dB`). |
| **Mục tiêu tĩnh ẩn (Targets)** | `25` | Tọa độ `(tx, ty)` nằm ngoài trụ tòa nhà và ngoài vùng phóng `160 m` quanh GCS; tối đa 1 mục tiêu/ô. | Đối tượng cần phát hiện qua tích lũy bằng chứng DS (`m_F >= 0.90, m_U <= 0.15`) và báo cáo về GCS. |

## 3. Cảm biến Dempster-Shafer (DS) & Độ tươi AoI
- **Cập nhật niềm tin DS bất biến thứ tự thời gian** (Zhang et al., *IEEE TIV 2024*, Định lý 1): Mỗi ô lưu bộ đếm phát hiện dương/âm `(N+, N-)`; suy ra trực tiếp `(m_F, m_E, m_U)` bằng công thức đóng `wp.float32`, loại bỏ hoàn toàn bão báo động giả của Bayesian `prior = 0.5`.
- **Xác nhận & Đóng gói báo cáo:** Khi ô đạt `m_F >= 0.90` và `m_U <= 0.15`, UAV tạo 1 gói báo cáo chứa `(target_id, timestamp t_s, hops=0, visited_mask=(1<<u))`.
- **Suy giảm mũ theo AoI tại GCS** (Xue et al., *Electronics 2026*): Khi báo cáo đến GCS lần đầu ở thời điểm `t`, giá trị tươi là `psi(t, t_s) = exp(-0.02 * (t - t_s))`.

## 4. Không gian Hành động & Quan sát (Dec-POMDP gọn, không `tx_power`, không PQL)
- **Hành động lai mỗi UAV:** `a_i = (a_x, a_y, d_i)` với `(a_x, a_y) in [-1, 1]^2` và `d_i in {0: Silent, 1..5: Peer_k, 6: GCS}` đi kèm **Adaptive Action Masking** (tự động khóa các đích ngoài tầm LOS danh định `480 m`, hàng đợi rỗng, hoặc UAV đã có mặt trong `visited_mask` để triệt tiêu vòng lặp relay).
- **Quan sát cục bộ `o_i` (~96 chiều) + One-Hot Agent ID (6 chiều):** Vị trí/vận tốc chuẩn hóa, phần trăm pin, trạng thái hàng đợi (số gói, tuổi AoI lớn nhất), bản đồ tóm tắt độ bất định DS cục bộ, vector tương đối tới các tòa nhà/UAV lân cận và trạng thái liên kết LOS/NLOS tới láng giềng & GCS.

## 5. Các Hiện tượng Mạng xảy ra trong kịch bản (PHY -> MAC -> Network -> App)
1. **Tầng PHY (`phy_sinr_fp32.py` — 2.4 GHz, 22 MHz, 2 Mbps, `P_tx = 0.1 W`, `NF = 7 dB`, `SINR >= 6 dB`):**
   - *Link Budget chuẩn đô thị:* `MAPL = 107.6 dB` với `alpha_LOS = 2.5` cho tầm phủ tự nhiên **`d_LOS ≈ 503 m`** và khi bị tòa nhà chắn (`+20 dB`) chỉ còn **`d_NLOS ≈ 80 m`** — ép buộc relay multi-hop một cách tự nhiên.
   - *NLOS Cliff Effect:* UAV bay ngang mép tòa nhà làm suy hao nhảy vọt 20 dB gây rớt gói tức thời.
   - *Nhiễu đồng kênh O(N^2) & Capture Effect:* Các UAV phát trong cùng sub-slot gây nhiễu cộng gộp tại máy thu; gói có tín hiệu áp đảo vẫn giải mã được nếu `SINR >= 6 dB`.
   - *Half-Duplex Deafness:* UAV relay đang phát lên GCS trong sub-slot `m` sẽ bị "điếc" trước gói từ UAV trinh sát gửi tới trong cùng sub-slot `m`.
2. **Tầng MAC (`mac_slotted_fp32.py` — Cách B: M=4 sub-slots cố định khi train + Full CSMA/CA UavNetSim khi eval):**
   - *Hidden & Exposed Terminals:* Ngưỡng cảm biến sóng mang `CCA = -82 dBm` cao hơn độ nhạy giải mã `-87.6 dBm` (5,6 dB) cộng với tòa nhà che khuất tạo hiện tượng nút ẩn/nút lộ thực tế.
   - *Retry Exhaustion & Head-of-Line Blocking:* Gói đầu hàng đợi lỗi vượt quá 5 lần retry sẽ bị hủy; trong lúc retry nó chặn các gói mới phía sau làm tăng AoI dây chuyền.
3. **Tầng Network & Application:**
   - *Funneling Effect quanh GCS:* Lưu lượng dồn vào 1–2 UAV gần GCS làm tăng xác suất va chạm cục bộ.
   - *Data Muling (Store-Carry-Forward):* Khi bị bao vây NLOS, UAV giữ gói trong bộ đệm và bay ra mép hành lang LOS trước khi truyền.
   - *Lão hóa hàng đợi & TTL Expiry:* Gói tồn quá `120 s` tự hủy và chịu phạt `w_5`.

## 6. Hàm thưởng & 5 Hàng rào chống Reward Hack
`R_t = 0.25 * Delta_U_global + 10.0 * Sum_new_GCS(exp(-0.02 * age)) - 0.03 * E_prop_kJ - 20.0 * N_safety_viol - 1.0 * N_drop_or_ttl`
1. **Global First-Arrival Deduplication:** Mỗi `target_id` chỉ nhận thưởng `w_2 = 10.0` duy nhất lần đầu tới GCS.
2. **Global Marginal Uncertainty:** `w_1 = 0.25` tính trên bản đồ DS hợp nhất toàn đội, cấm ăn trùng giữa các UAV.
3. **Zero Per-Hop Reward + Visited Bitmask:** Không thưởng trung gian khi truyền UAV-UAV, triệt tiêu ping-pong.
4. **Non-Terminal Safety Penalty (`w_4 = 20.0`):** Phạt liên tục + chiếu vận tốc an toàn (không ngắt episode sớm để tránh suicide exploit).
5. **Zeng (2019) Rotary-Wing Power:** Bay treo tốn `168 W` > bay hành trình `~132 W`, triệt tiêu hành vi đứng yên gần GCS ăn thưởng.

## 7. Cấu hình Vanilla MATD3 & Vanilla MASAC (1 Seed T4 -> 5 Seeds Paper)
- Cả hai dùng **Plain MLP `[256, 256]` ReLU**, **KHÔNG BatchNorm/LayerNorm/output_norm/PQL**, **Shared Actor + 6-dim One-Hot Agent ID**, **Shared Twin Centralized Critic**, `batch_size = 1024`, `replay_capacity = 300,000` (FP16) chiếm chỉ **1.03 GiB VRAM** -> nằm **100% trên VRAM GPU 1** của Kaggle T4.
- **Dual-Fidelity Eval (`dual_fidelity_eval.py`):** Sau khi train siêu tốc bằng Slotted Warp FP32 (`CUDA Graph`), đánh giá chéo trên cùng 100 seed độc lập bằng **Full CSMA/CA UavNetSim** để báo cáo bảng **Sim-to-Eval Gap (Delta)** cho bài báo Q1/A*.
