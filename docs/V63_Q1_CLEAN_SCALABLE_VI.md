> **ARCHIVED / KHÔNG CÒN CHÍNH XÁC (v6.3):** Các khẳng định `CCA=-70 dBm khớp UavNetSim`, `249.1 m LoS`, `zero-shot từ một checkpoint` và `Warp GPU scalable đã chạy` dưới đây chưa được chứng minh hoặc sai. Bản đính chính và gate thực tế: [V64_COMBINED_INDEPENDENT_AUDIT_VI.md](V64_COMBINED_INDEPENDENT_AUDIT_VI.md).

# Bản v6.3 Q1-Clean & Scalable (Tham số hóa N-UAV / Grid / Map + Cắt bỏ Bloat)

Ngày: 25/11/2026

## 1. Tham số hóa hoàn toàn cho thực nghiệm Scalability (Training & Zero-Shot Testing)

Tất cả hằng số từng bị hardcode (`6 UAV`, `20×20`, `1000 m`, `7 action`) nay đổi trực tiếp qua `research/zhang_xue/configs/scenario_7radio.yaml` hoặc `make_v63_q1_config(...)`:

- `num_uavs` (mặc định `6`, hỗ trợ `1..30` UAV, tự suy `gcs_id = num_uavs`, `radio_n = num_uavs + 1`, dàn đội hình khởi tạo tự động giữ khoảng cách an toàn $\ge 65\text{ m}$).
- `grid_w` & `map_size_m` (mặc định `20` & `1000.0 m`; hỗ trợ mở rộng `40` & `2000.0 m`, `60` & `3000.0 m`, gói tin 8-byte tự động chuyển chế độ nén 5-bit origin / 12-bit cell / 15-bit timestamp khi $N > 8$ hoặc lưới $> 512$ ô).
- `comm_action_mode: compact_3` (`0 = NO_SEND`, `1 = SEND_GCS`, `2 = SHARE_PEERS` gửi tới đồng đội sống gần nhất có dữ liệu mới): giữ kích thước action rời rạc **cố định bằng 3** bất kể `num_uavs = 4, 6, 8, 10, 12` → cho phép **train trên 6 UAV và test zero-shot trên số lượng UAV khác** (`encode_actor_input_scalable` + `decode_policy_action_2d`).
- `make_shapes_v63(num_uavs, grid_w)` trong `zhang_xue/gpu/layout_v62.py` tham số hóa dung lượng Warp GPU theo $(N_{\text{UAV}}, \text{grid\_w})$ (tiếp tục dùng Warp GPU, chưa cần phụ thuộc UavNetSim SimPy gốc).

## 2. Các thành phần thừa đã cắt bỏ / căn chỉnh chuẩn Q1 (khớp UavNetSim-v1)

- **Bỏ đầu ra công suất phát RF (`continuous_dim = 2`, chỉ còn `v_x, v_y`):** Loại bỏ chiều hành động chết gây nhiễu entropy MASAC/Critic MATD3; cố định `tx_power_w = 0.1 W` khớp Bảng III-B của [UavNetSim-v1 (arXiv:2507.09852)](https://arxiv.org/html/2507.09852v1) (tầm LoS ≈ 249,1 m, kích hoạt đa chặng tự nhiên trên bản đồ 1 km).
- **Sửa lỗi vật lý `CCA = -70.0 dBm` (`1e-10 W`):** Cao hơn sàn nhiễu nhiệt `-74 dBm` (`4e-11 W`) đúng 4 dB (khắc phục lỗi cũ `-82 dBm` nhỏ hơn nhiễu nhiệt 8 dB).
- **Bỏ `sensor_likelihood_ratio` và `enable_accel_energy` khỏi cấu hình Q1 mặc định:** Dùng thuần `paper_fixed (0.80/0.80)` của [Zhang et al. (IEEE TIV 2024)](https://www.semanticscholar.org/paper/Enhancing-Multi-UAV-Reconnaissance-and-Search-DDPG-Zhang-Lin/c3962200ccb4224ae66a28a26f9bd424f4e1a7f9) + AoI EWMA `UQ8.8` end-to-end của [Xue et al. (Electronics 2026)](https://www.mdpi.com/2079-9292/15/9/1875) + pin steady-level Eq. 12 của [Zeng et al. (IEEE TWC 2019)](https://www.researchgate.net/publication/324328894_Energy_Minimization_for_Wireless_Communication_With_Rotary-Wing_UAV).
- **Bỏ nhiễu phạt `r_airtime` (`lambda_b = 0.0`)** và căn chỉnh động học hợp lý cho bản đồ 1 km (`v_max_mps = 15.0`, `safety_horizon_s = 900`, `e_max_j = 180_000 J`).

## 3. Kết quả kiểm định

- **Governance Policy (`check_project_policy.py`):** 0 vi phạm trên toàn bộ cây mã nguồn ($\le 500$ dòng/tệp).
- **Kiểm thử hồi quy CPU (`unittest`):** **72 PASS, 0 FAIL, 1 SKIP** (thiếu `warp-lang`), đã kiểm chứng mở rộng `num_uavs = 4, 6, 8, 10` và lưới `40×40` / bản đồ `2.000 m`.
