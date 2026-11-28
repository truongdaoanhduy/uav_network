# V81 — BÁO CÁO KIỂM TOÁN TOÀN DIỆN V80 (24 VẤN ĐỀ + ĐỐI CHIẾU Q1/A*) & LÝ DO TỪNG BẢN SỬA MÃ NGUỒN

**Ngày:** 2026-11-26 (UTC+07). **Nguồn đối chiếu:** `UAV_V80_VAST_GITHUB_FULL_CODE_LOG_AUDIT_20261126.zip` (GitHub commit `0bdef599e5ef586b7973ae4220a1d53106773f32`, 402 passed, 6/6 CUDA PASS, 24 vấn đề trong `ISSUES_CODE_LOG_SCIENCE.csv`).

## 1. Kiểm kê các thành phần đang có trong mã nguồn V80 & xung đột cốt lõi

1. **Bản đồ, địa hình & NFZ (`scenario_v77_nfz_multi_1km.yaml`, `mission_v79_reset_obs.py`)**:
   - Đã có: Bản đồ 1000 m × 1000 m, tòa nhà, vùng cấm bay (NFZ) đa hình trụ, giới hạn biên và lưới niềm tin 20×20 (400 ô).
   - Xung đột / chưa đúng: (a) Khởi tạo đội bay theo `65.0 * float(u)` cho khoảng cách láng giềng `sqrt(65² + 30²) = 71,59 m`, biên dự phòng trên ngưỡng `50 m` chỉ còn `21,59 m < 22,5 m` quãng đường phanh cực đại (`v_max² / (2 a_max)`); (b) Actor chỉ nhận 14 kênh cục bộ + belief gộp, không nhận rõ vị trí 5 UAV bạn (từ gói HELLO) và bản đồ khoảng cách NFZ cục bộ (`#10`, `#11`); (c) 3/5 YAML ablation bị từ chối vì tràn UQ8.8 (`eta=0` với `H=900 > 255,996`), sai `battery_termination_mode`, hoặc `w_p2p > 0` chưa có trên GPU (`#12`, `#23`).
2. **Kịch bản UAV & Safety Shield (`mission_v80_shield.py`, `mission_v80.py`)**:
   - Đã có: Shield 66 ứng viên khả thi + 3-sweep coordinate descent, giữ `safe_d = 50 m`, `v_max = 15 m/s`, `a_max = 5 m/s²`, fail-closed không ghi đè `pos/vel` khi lỗi (`#08`).
   - Lỗi ẩn mới phát hiện (P1): Ở Phase 2 (`mission_v80_shield.py:230-258`), kiểm tra va chạm cặp gộp chung vòng lặp với hình học (`if battery[u] <= 0: continue`), khiến cặp `(u, peer)` bị bỏ qua khi `u` hết pin nhưng `peer > u` vẫn đang bay — tạo ra bất đối xứng vật lý.
   - Giới hạn khoa học (`#09`): 66 ứng viên rời rạc chưa có chứng minh tập khả thi (viability/CBF-QP) theo Wang, Ames & Egerstedt (IEEE T-RO 2017) và Alshiekh et al. (AAAI 2018).
3. **Mạng UavNetSim & kênh vô tuyến (`runtime_v67.py`, `scenario_v77_nfz_multi_1km.yaml`)**:
   - Đã có: Kênh vô tuyến suy hao xuyên tòa nhà tham chiếu ITU-R P.2109-2, MAC CSMA/CA nén chạy song song trên NVIDIA Warp, chốt chống tràn bộ đệm sự kiện `require_no_warp_overflow` mỗi bước trước `auto_reset_gpu()` (`#07`), khóa cứng `dt_s == 1.0` (`#06`).
   - Xung đột / chưa đúng: (a) Chưa có tương đương từng sự kiện rời rạc với UavNetSim-v1 SimPy gốc (Zhou et al., ICCC 2025) (`#18`) — chỉ là **surrogate MAC/PHY theo khe thời gian**; (b) `mac_noise_power_w = 4.389577e-13 W` lệch `19,596 dB` so với `4e-11 W` của bài báo UavNetSim (`#16`); (c) `rf_per_building_loss_db = 12.0` là heuristic hình trụ, chưa hiệu chuẩn shadowing (`#17`); (d) `lambda_b = 0.0` và `w_p2p = 0.0` nên không có phạt nghẽn airtime (`#15`).
4. **Reward & phân bổ tín dụng (`mission_v79_kernels.py`)**:
   - Đã có: Tách riêng sổ cái đội cục bộ (`w_loc`) và xác nhận về GCS, không rò rỉ oracle vào Actor.
   - Chưa kiểm chứng (`#14`, `#20`): Cần ablation giữa thưởng cục bộ vs thưởng chỉ khi giao về GCS kết hợp Potential-Based Reward Shaping (Ng, Harada & Russell, ICML 1999) và Age of Information (Yates et al., IEEE JSAC 2021). Ở 300 bước hiện tại cả MATD3 và MASAC đều đạt `0/15` mục tiêu.
5. **Huấn luyện song song trên NVIDIA Warp & ổn định MARL (`run_v80_real_rollout.py`, `models.py`, `matd3/learner.py`, `masac/learner.py`)**:
   - Đã có: Biên dịch nhanh `2,57s` nhờ `enable_backward=False, max_unroll=2` (`#02`), chạy được `E4096/B8192` 2 bước cập nhật.
   - Lỗi & xung đột nghiêm trọng bị bỏ sót trong CSV:
     - **Bệnh lý MATD3 (`38` chết / `35` đâm biên vs MASAC `3` chết / `1` đâm biên)**: `pretanh_abs_mean` tăng `0,091 -> 4,0678` (max `6,0064`), `|a| > 0,99` chạm `100,0%` từ bước 279, `tanh_deriv = 0,00318`, `actor_grad_norm = 0,00142` (yếu hơn MASAC **57×**), Dead ReLU ở `TwinCritic` đạt **`82,42%–83,3%`** (Sokar et al., ICML 2023; Lyle et al., ICML 2023). Kết quả là 26 world MATD3 đồng loạt đâm biên ở bước `116–119`.
     - **4 nguyên nhân gốc trong code chưa được CSV liệt kê**: (1) `TwinCritic` thiếu `LayerNorm` (Ball et al., ICML 2023; Nauman et al., ICML 2024); (2) `actor_output_regularization = 0.0` (thiếu phạt `pre_tanh^2` của MADDPG — Lowe et al., NeurIPS 2017); (3) `run_v80_real_rollout.py` không có warmup `learning_starts` và ép `replay_capacity = max(batch*4, envs*8) = 512` ở `E64` (ghi đè toàn bộ buffer mỗi 8 bước); (4) đồng bộ host D2H + ghi JSONL mỗi bước gây nghẽn khi lên `E4096` (`#24`), và GPU0 bị chiếm nên chưa chứng minh tăng tốc 2-GPU (`#21`).

## 2. Thẩm định toàn bộ 24 vấn đề trong `ISSUES_CODE_LOG_SCIENCE.csv`

| ID | Hạng mục trong CSV | Trạng thái V80 | Thẩm định (Đúng / Chưa đủ / Hướng xử lý V81) |
| --- | --- | --- | --- |
| `#01`–`#08` | 8 bản sửa Vast.ai (dtype, compile 2.57s, 14 args, key, clamp order, `dt_s=1`, overflow mỗi bước, fail-closed `pos/vel`) | `FIXED_*` | **ĐÚNG & ĐỦ**: Đã xác minh qua diff và test RED/GREEN. V81 giữ nguyên 100% 8 bản sửa này. |
| `#09` | Shield 66 ứng viên chưa có chứng minh hình thức | `OPEN_SAFETY` | **ĐÚNG, NHƯNG THIẾU 1 LỖI CODE**: V80 bỏ sót lỗi bất đối xứng `dead-UAV` ở Phase 2 -> **V81 đã sửa** trong `mission_v80_shield.py`. Phần chứng minh CBF-QP vẫn cần giữ trạng thái OPEN. |
| `#10` | Khoảng cách spawn `71,59 m` quá sát ngưỡng phanh `50 + 22,5 m` | `OPEN_INITIALIZATION` | **ĐÚNG**: Cần giữ làm ablation `72 / 100 / 120 m` trên GPU sau khi MATD3 hết bão hòa. |
| `#11` | Actor thiếu quan sát 5 peer từ HELLO và bản đồ NFZ cục bộ | `OPEN_ACTOR` | **ĐÚNG**: Không được bơm vị trí oracle; phải lấy qua gói HELLO có trễ mạng khi thiết kế bản V82. |
| `#12`, `#13` | `ds_aoi_eta=0` tràn UQ8.8; ghép niềm tin DS chưa đối chiếu giải tích | `OPEN_BELIEF*` | **ĐÚNG**: Giữ nguyên UQ8.8 cho V81 để không vỡ 402 test; cần mở rộng bộ đếm UQ16/float khi làm ablation `eta=0`. |
| `#14`, `#15` | Tín dụng cục bộ vs GCS; chưa bật phạt airtime `lambda_b` | `NEEDS_ABLATION` / `CONFIG_DIFF` | **ĐÚNG**: Đã có `scenario_v79_airtime_penalty_1km.yaml` được GPU hỗ trợ để chạy cặp so sánh trực tiếp. |
| `#16`, `#17`, `#18` | Lệch `19,596 dB` noise PHY, suy hao 12 dB/tòa nhà, chưa parity SimPy UavNetSim | `CONFIG_DIFF` / `OPEN_*` | **ĐÚNG**: Phải ghi rõ đây là surrogate MAC/PHY. V81 thêm `scenario_v81_nfz_multi_highnoise_1km.yaml` (`4.0e-11 W`, `all_depleted`) để mở khóa ngay ablation noise khớp bài báo UavNetSim. |
| `#19` | Chênh lệch scale loss Critic giữa MATD3 (`MSE1+MSE2`) và MASAC (`0.5*(MSE1+MSE2)*6`) | `NOT_PROVEN_BUG` | **CHƯA ĐỦ**: Đúng là không phải bug crash, nhưng làm gradient critic MATD3 yếu hơn 3× so với MASAC khi so sánh công bằng -> **V81 thêm `sync_critic_loss_scale`** (opt-in) để chuẩn hóa hai thuật toán về cùng thang đo. |
| `#20`, `#22` | Chưa hội tụ 15 mục tiêu; reset 900 bước mới chứng minh với hành động 0 | `OPEN_LONG_TRAIN` / `PARTIAL` | **ĐÚNG**: Không thể chạy 900 bước trước khi chữa bệnh chết biên ở MATD3 -> V81 giải quyết điều kiện tiên quyết này trước. |
| `#21`, `#24` | GPU0 bận (chưa có 2-GPU); D2H sync mỗi bước chậm ở `E4096` | `OPEN_2GPU` / `OPEN_PERFORMANCE` | **ĐÚNG**: Chạy các seed độc lập trên từng GPU khi có 2 GPU thay vì phụ thuộc P2P; giảm tần suất ghi JSONL ở vòng huấn luyện dài. |
| `#23` | 3 YAML ablation bị từ chối (`eta0`, `highnoise`, `commreward`) | `OPEN_ABLATION_SUPPORT` | **ĐÚNG**: V81 khắc phục ngay kịch bản `highnoise` qua `scenario_v81_nfz_multi_highnoise_1km.yaml`; `eta0` và `w_p2p` cần nâng cấp kernel ở V82. |

## 3. Lý do chi tiết cho từng tệp đã sửa/thêm trong V81 (và nguồn gốc khoa học)

1. **`uav_network/algorithms/core/models.py`** (`244` dòng):
   - *Sửa gì*: Thêm `layer_norm=False` vào `TwinCritic`, thêm bộ chọn `activation` (`"relu"` / `"silu"`), thêm `DeterministicActor.pre_tanh_penalty(observations)`.
   - *Tại sao phải sửa*: Log Run 38 chứng minh `q1_relu_0` bị **82,4%–83,3% Dead ReLU** và `pre_tanh` tăng lên `4,068` (bão hòa `tanh` 100%). Thay `ReLU` bằng `SiLU` + `LayerNorm` ngăn chết nơ-ron và ổn định biên độ hoạt hóa; phạt `mean(pre_tanh²)` giữ đạo hàm `tanh` không sụp về 0 để Actor còn khả năng rẽ tránh tường biên.
   - *Nguồn*: Ball et al. (ICML 2023, RLPD), Sokar et al. (ICML 2023), Nauman et al. (ICML 2024), Lowe et al. (NeurIPS 2017, MADDPG).
2. **`uav_network/algorithms/configuration/schema.py`** (`164` dòng):
   - *Sửa gì*: Thêm các trường opt-in (`critic_layer_norm`, `activation`, `critic_huber_delta`, `sync_critic_loss_scale`, `pre_tanh_reg`) với mặc định giữ nguyên hành vi V80 và kiểm tra hợp lệ fail-closed.
   - *Tại sao*: Bảo toàn 100% tương thích ngược với checkpoint V72–V80 và toàn bộ 402 bài test cũ.
3. **`uav_network/algorithms/matd3/learner.py`** (`452` dòng) & **`uav_network/algorithms/masac/learner.py`** (`463` dòng):
   - *Sửa gì*: Kết nối `critic_layer_norm`, `activation`, `F.huber_loss` (`critic_huber_delta > 0`), `sync_critic_loss_scale`, và cộng `pre_tanh_reg * actor.pre_tanh_penalty(obs)` vào `actor_loss` của MATD3.
   - *Tại sao*: Huber loss chặn cú sốc gradient khi TD-error nhảy vọt do phạt safety `-100` (`critic_loss` đỉnh `1678`); đồng bộ scale critic loss loại bỏ lệch 3× giữa MATD3 và MASAC.
4. **`research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py`** (`301` dòng):
   - *Sửa gì*: Tách vòng lặp kiểm tra hình học (`battery[u] > 0`) khỏi vòng lặp cặp `(u, peer)` ở Phase 2; chỉ bỏ qua cặp khi **cả hai** `battery[u] <= 0.0` và `battery[peer] <= 0.0`.
   - *Tại sao*: Loại bỏ lỗ hổng va chạm bất đối xứng khi UAV sống bay qua vị trí UAV đã cạn pin.
5. **`configs/learners/v81_matd3_masac_stable.yaml`**, **`research/zhang_xue/configs/scenario_v81_nfz_multi_highnoise_1km.yaml`**, **`scripts/gpu/run_v81_real_rollout.py`** (`227` dòng), **`tests/test_v81_stability_and_shield_contracts.py`** (`79` dòng):
   - *Tại sao*: `run_v80_real_rollout.py` ép `replay_capacity = 512` ở `E64` và cập nhật ngay từ bước 2 không có warmup, làm MATD3 overfit vào một nhúm mẫu đầu tiên. `run_v81_real_rollout.py` bổ sung `--learner-config`, `--learning-starts 4096`, `--replay-capacity 65536` mà không đụng tới hợp đồng của runner V80.

## 4. Trạng thái kiểm chứng trung thực (CPU Sandbox vs Vast.ai CUDA)

- **Đã PASS trên CPU sandbox**: `verify_source_manifests.py --check` (`294` root + `122` research checksums), `check_project_policy.py --scope full` (`296` tệp, `0 violations`, tất cả `<= 500` dòng), `compileall`, `tests/test_v81_stability_and_shield_contracts.py` (`5/5 PASS`), và `34/34` tệp `unittest` tương thích CPU (`0 FAIL`).
- **Chưa chạy được trong sandbox (bắt buộc chạy trên Vast.ai CUDA)**: `31` tệp test cần `torch`/`warp-lang`/`pytest` và rollout GPU CUDA thực tế cho MATD3/MASAC V81.
