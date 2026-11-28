# V115 — Kiểm Toán Độc Lập Nghiên Cứu Action (Q1/A*), Sửa C38 & C42 và Lộ Trình Khóa Kịch Bản

**Trạng thái:** `NOT SCENARIO_FROZEN` (chưa huấn luyện RL; **30/30 tệp `uav_network/algorithms/` giữ nguyên byte-identical**).  
**Bảo toàn bằng chứng trước đó:** Toàn bộ `V114_ACTION_PAPER_CONFLICT_AUDIT/` được lưu tại `outputs/v114_action_audit/`; hai YAML active (`scenario_v103_ds_only_1km.yaml` `3fcfe434...fc5`, `scenario_v105_ds_rf_stress_1km.yaml` `3ee084e7...b76`) giữ nguyên 100% để bảo toàn toàn bộ bằng chứng Vast.ai V109/V112/V113 (`593 passed`, C14 `4/4 PASS` ở `4096×900`).

---

## 1. Các lỗi thật P0 được sửa trực tiếp bằng code trong V115

1. **Sửa `C38` (P0 — Lệch ABI vận tốc giữa CPU và Warp GPU):**
   - **Nguyên nhân:** CPU `env.step()` nhận vận tốc theo `m/s`, trong khi Warp GPU `_candidate_v80` nhận vector chuẩn hóa `[-1, 1]` rồi chiếu lên đĩa đơn vị và nhân `v_max_mps`.
   - **Giải pháp V115:** Bổ sung `scale_normalized_xy_to_mps(norm_xy, v_max_mps)` và `step_normalized_xy_2d(env, norm_xy, destination, route_mask)` vào `research/zhang_xue/src/zhang_xue/learners/actor_adapter.py`, tạo ABI chuẩn hóa `[-1, 1]` khớp 1-1 với Warp mà không phá bất kỳ test CPU cũ nào.
2. **Sửa `C42` (P0 — Mặc định runner nguy hiểm dễ chạy nhầm hybrid & kịch bản cũ):**
   - **Nguyên nhân:** `scripts/gpu/run_v81_real_rollout.py` vẫn giữ default `--baseline-mode hybrid` và `scenario_v77_nfz_multi_1km.yaml` (bị `test_v88_active_scope_lock.py` khóa chuỗi).
   - **Giải pháp V115:** Bổ sung launcher fail-closed `scripts/gpu/run_v115_canonical_ds_baseline.py` khóa cứng mặc định `scenario_v103_ds_only_1km.yaml`, `v85_canonical_matd3_masac_baseline.yaml`, `--baseline-mode canonical_continuous`, `--batch-size 256` và từ chối fail-closed nếu YAML không phải `dempster_shafer` hoặc bật `enable_tx_power_control`.

---

## 2. Kiểm chứng độc lập câu hỏi: Paper Q1/A* nào cho UAV học action ngoài `(x, y)`?

Kiểm tra chéo 14 nguồn của V114 cho thấy: không có DOI ảo, nhưng 6/7 bài IEEE Q1 trong V114 mới đọc abstract, P01 (Zhang 2024, IEEE T-IV) đang bị Clarivate gắn cờ *On Hold* từ 7/2024, P02/P06 là MDPI Q2, P12 (`arXiv:2608.04590`) tên thật là `JUROR` (không phải `JUDOR`).

Kết quả tra cứu trực tiếp các bài **Q1 IEEE Transactions** đã xác thực:

| Action ngoài `(x, y)` | Bài Q1 đã xác thực (DOI/URL) | Vấn đề giải quyết | Tương thích canonical `MATD3`/`MASAC`? |
| :--- | :--- | :--- | :--- |
| **Độ cao bay liên tục $z$ ($v_z$)** | IEEE TMC 2026 (`10.1109/TMC...`, `11363000`); Li et al., IEEE IoTJ 2019 (`10.1109/jiot.2019.2955732`) | Trade-off vật lý 2 chiều: bay cao mở FoV & LoS tốt hơn nhưng giảm $P_d(z)$ & tăng pathloss; bay thấp ngược lại. | **CÓ (100% nguyên bản)**: `act_dim = 3`, `target_entropy = -3.0`. |
| **Lập lịch truyền / chọn kênh / relay** | Alam & Moh, IEEE TMC 2024 (`10535707`); Lu et al., IEEE TWC 2026 (`10.1109/twc.2026.3658601`); Yang et al., IEEE TVT 2025 (`10.1109/tvt.2025.3621495`); Emami et al., IEEE TVT 2024 (`10.1109/tvt.2024.3394235`) | Tránh nhiễu đồng kênh 3D, cân bằng tải relay/khe thời gian khi băng thông chật. | **KHÔNG** (biến rời rạc $\to$ phải đổi sang HQMIX / MF-HPPO / hybrid). |
| **Communication Gating** | Gao et al., IEEE IoTJ 2024 (`10.1109/jiot.2024.3362988`); Drones 2026 (Constrained QMIX + phạt Lagrange) | Chỉ có ích khi phạt chi phí phát sóng rất mạnh (communication budget) hoặc kênh nghẽn nặng. | **KHÔNG** (biến rời rạc). |
| **Tách tầng (RL bay + bộ lập lịch nhẹ)** | Fu et al., IEEE IoTJ 2025 (`10.1109/jiot.2025.3636204`, TL-RATS) | Chứng minh nhét scheduling rời rạc vào 1 policy RL phẳng gây mất ổn định; tách RL quỹ đạo + lập lịch nhẹ thắng cả fully-joint RL. | **CÓ** (đúng kiến trúc `canonical_continuous` hiện tại). |
| **MEC Offloading / Gimbal** | Luo et al., IEEE JSAC 2023; AIAA JAIS 2023 (`10.2514/1.I011284`) | Chỉ có nghĩa khi có mô hình tải tính toán biên hoặc camera xoay nghiêng. | Offloading: không; Gimbal liên tục: có. |

---

## 3. Tại sao `HOLD / SEND_NEW / SEND_PRIORITY` là action vô nghĩa trên V113 hiện tại?

1. **`HOLD` bị `always-SEND` thống trị tuyệt đối:** `lambda_b = 0.0`, TX `0.1 W` vài ms ($\sim 10^{-4}\text{ J}$) nhỏ hơn công suất bay (~200 W) $10^5$ lần, `QUEUE_CAP_PKTS = 64` không bao giờ tràn (`overflow = 0`), và E2E ACK tự giữ dirty cells nếu gói rớt RF.
2. **`SEND_PRIORITY` vs `SEND_NEW` bị triệt tiêu:** `MAX_ENTRIES_PER_PKT = 62` chứa gần như toàn bộ dirty cells trong 1 gói duy nhất và `ds_aoi_eta = 0.0`.
3. **Phá chuẩn `MATD3`/`MASAC` gốc (`C46`):** `canonical_continuous.py` đòi đạo hàm liên tục $\nabla_a Q_1$ (MATD3) và squashed Gaussian với single-$\alpha$ `target_entropy = -d` (MASAC).

---

## 4. Lộ trình chốt dứt khoát

- **Phase 1 (Khóa kịch bản 2D pure-GCS):** Giữ action liên tục 2D `(x, y)` + `event_driven_comm_action`, chốt `PEER=false, MULE=false` (`v103` Base + `v111_rf_stress_no_peer`), khóa `SCENARIO_FROZEN`.
- **Phase 2 (Baseline chuẩn gốc):** Xử lý `S25` (bỏ trùng `state` + lưu `obs` FP16 trong replay buffer để giảm 4× VRAM trên RTX 3060 12 GiB) rồi chạy `run_v115_canonical_ds_baseline.py --algorithm matd3|masac`.
- **Phase 3 (Mở rộng action có ý nghĩa nếu cần cho bài báo):** Chọn **độ cao bay liên tục $v_z$ (`act_dim = 3`)** kèm mô hình vật lý $P_d(z)$, $\text{FoV}(z)$, LoS($z$) để giữ nguyên 100% `MATD3`/`MASAC` continuous gốc.
