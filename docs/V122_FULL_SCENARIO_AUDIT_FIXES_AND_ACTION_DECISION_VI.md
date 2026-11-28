# V122 — Kiểm Toán Kịch Bản DS-Only, Sửa Lỗi C83–C84, Thẩm Định Action Space & Hiệu Chuẩn Đa Góc Nhìn (28/11/2026)

**Trạng thái khóa kịch bản:** `SCENARIO_FROZEN = false` *(sẵn sàng khóa `true` ngay sau 1 lượt chạy lại `pytest` trên Vast.ai RTX 3060 cho 3 tệp nguồn V122 vừa vá; toàn bộ 30/30 tệp `uav_network/algorithms/` và 2 YAML active vẫn 100% byte-identical).*

## 1. Thẩm định các kết luận của V121: điểm đúng, điểm chưa đủ và 2 lỗi mới phát hiện

| Hạng mục | Kết luận V121 | Thẩm định & Bằng chứng V122 |
| :--- | :--- | :--- |
| **CUDA 4096×900 không reset & Shield tie (`C73/C78`)** | 2/2 CUDA PASS, 0 lỗi an toàn, shield max diff `3.86e-6 m/s` | **ĐÚNG** về ổn định thực thi và shield trên prestate chung; vẫn không đồng nghĩa parity toàn trạng thái CPU–GPU. |
| **Lỗi mới `C83`: Công thức gia tốc Zeng 2019 Appendix A (`flight_safety_energy.py:74-76`)** | V121 đo `enable_accel_energy=True` giảm pin `74.12% -> 70.50%` nhưng không phát hiện lỗi đại số | **SAI ĐẠI SỐ ĐÃ SỬA (RED→GREEN):** V121 dùng `kappa**1.5 * sqrt(1/(hypot(1,x)+x))`, thiếu `kappa^2` trong căn so với Zeng et al. (IEEE TWC 2019, `10.1109/TWC.2019.2902559`, Appendix A Eq. 69): $P_{\text{ind}} = P_i \kappa \sqrt{\kappa^2/(\sqrt{\kappa^2+x^2}+x)}$. Sai lệch thực tế `2.35 W` tại $v=10\text{ m/s}, a=5\text{ m/s}^2$; sau sửa, pin tối thiểu 300 bước là `70.39%` (vs `74.12%` steady-level). |
| **Lỗi mới `C84`: Lệch thưởng `r_blocked_motion` CPU vs Warp khi terminate an toàn (`transition.py:259`)** | Chưa phát hiện trong V121 | **LỖI PARITY ĐÃ SỬA (RED→GREEN):** Khi `safety_violation=True` và `safety_failure_terminate=True`, Warp `_move_v80` gán `intervention[e] = 0.0`, còn CPU vẫn trừ `-0.1333` (`r_blocked_motion`). Đã gán `shield_intervention.fill(0.0)` ở CPU để khớp 100% Warp. |
| **DS dưới tương quan thời gian (`C04/C23/C41/C66`)** | Phát hiện `FP-ever` tăng `0.333 -> 13.0` ở $\rho=0.7$, `zhang_max` và `confirm_th=0.99` đều không giải quyết được | **ĐÃ ĐƯA RA & KIỂM CHỨNG GIẢI PHÁP WARP-READY (`ds_correlation_mitigation.py`):** Cài đặt module `O(1)` mỗi ô (`int16 last_obs_step` 4.8 KB/world + `uint8 uav_pos_mask` 400 B/world) với 2 cơ chế chuẩn thống kê: (a) **Revisit Cooldown** $\tau=9\text{ s}$ ($\rho^9 \le 0.05$) giảm FP-ever ở $\rho=0.7$ từ `13.000 -> 1.667` (**−87.2%**) và $\rho=0.5$ từ `8.333 -> 0.000`; (b) **AR(1) Effective Sample Weight** $w=(\frac{1-\rho^{\Delta t}}{1+\rho^{\Delta t}})$ giảm FP-ever từ `13.000 -> 4.333` (**−66.7%**). |
| **Suy hao tòa nhà RF `12 dB` (`C13/C32`)** | Ghi nhận `12 dB` là heuristic chưa hiệu chuẩn | **ĐÃ KIỂM CHỨNG ĐỘ NHẠY THEO AL-HOURANI (IEEE WCL 2014):** Quét `12 dB` (sparse single-obstacle) vs `20 dB` (Al-Hourani Urban) vs `23 dB` (Dense Urban) trên 3 seeds × 300 bước cho `gcs_tp_ever` và `delivered_cov` **trùng khớp 100%** ở cả Base (`11.667 TP / 79.25% cov`, PDR chỉ lệch `0.920 -> 0.906`) lẫn RF-stress (`5.333 TP / 32.50% cov / PDR 0.126`). |
| **Khung nén `70+8k B` vs UavNetSim `1082 B` (`C07/C33/C65`)** | Chưa đo tải kênh thực tế ở payload `1082 B` | **ĐÃ ĐỐI CHIẾU TRỰC TIẾP REPO `UavNetSim-v1` (commit `c2703f2`, arXiv:2507.09852):** Tổng airtime 6 UAV trên Base 300 s là `3,750 ms` (`1.25%` kênh); kể cả quy đổi toàn bộ DATA sang `1082 B` (`4.328 ms`) cũng chỉ `7,147 ms` (`2.38%` kênh, chưa bão hòa). Khóa scope chuẩn Q1/A*: *"UavNetSim-inspired GPU IEEE 802.11b DCF surrogate"*, không tuyên bố tương đương sự kiện SimPy. |

## 2. Thẩm định Action Space (Bắt buộc cố định độ cao `z = 50 m`, không thêm chiều vô nghĩa)

| Action ứng viên | Tình trạng trong kịch bản 6 UAV (`1×1 km`, `z=50 m`) | Tương thích MATD3/MASAC gốc | Quyết định |
| :--- | :--- | :--- | :--- |
| **2D XY liên tục (`act_dim = 2`)** | Giải quyết đồng thời phủ lưới 400 ô, né 5 tòa nhà/2 blackout, giữ LoS/data-mule về GCS và tối ưu đường cong công suất bay Zeng 2019 | **100% nguyên bản** | **KHÓA LÀM BASELINE PHASE 1 & 2** |
| **TX Power (`tx_power_w`)** | **Suy biến về $P_{\max}$ (vô nghĩa):** $P_{\text{tx}} \le 0.5\text{ W}$ chiếm `< 0.3%` so với bay `140–183 W`; kênh 6 nút chỉ tải `1.25%` nên giảm công suất không tạo spatial reuse mà còn gây hidden terminal (Kawadia & Kumar, IEEE WCM 2005). Warp `step_v67` cũng chặn `enable_tx_power_control=True`. | Liên tục nhưng thừa/suy biến | **LOẠI (giữ cố định `0.1 W`)** |
| **Routing Weights (`w_d,w_l,w_q,w_e`) / Direct Next-Hop** | **Over-parameterized & chết gradient:** Trên Base, `99.4%` bước có đường 1-hop thẳng về GCS; trên RF-stress chỉ `15.1%` bước mất GCS với `1–2` ứng viên relay. Luật chuyển tiếp có điều kiện tất định đã ăn trọn `+40.2% TP / +15.0 pp cov` mà không tăng action từ `12 -> 36` chiều cho Centralized Critic. | Weights: nhiễu cao; Next-Hop: rời rạc, phá MATD3/MASAC | **LOẠI khỏi action học (dùng luật tất định)** |
| **Transmit / Hold (VoI/AoI)** | **Bị áp đảo tuyệt đối (strictly dominated):** Khi `lambda_b = 0` và kênh rảnh (`1.25%`), Hold không có lợi ích biên dương, chỉ làm tăng trễ xác nhận `19×` (Base) và giảm `−20% TP` (RF-stress). | Rời rạc / gượng ép | **LOẠI** |
| **Tốc độ gắn $P_d(\|\mathbf{v}\|)$** | **Đánh đổi Pareto thật ngay trên `act_dim = 2`:** Bay nhanh phủ lưới vs bay chậm tăng $P_d$ giảm FP DS mà không tăng số chiều action. | **100% nguyên bản (`act_dim = 2`)** | **ỨNG VIÊN MỞ RỘNG PHASE 3 (miễn phí chiều)** |
| **Hướng quét Gimbal ($\theta_g$)** | **Ứng viên #1 nếu muốn thêm đúng 1 chiều liên tục (`act_dim = 3`) mà vẫn giữ `z=50 m` cố định:** Yêu cầu đổi cảm biến nadir tròn sang **off-nadir directional FoV** bị 5 tòa nhà 70–80 m che khuất 3D để tách vị trí giữ sóng RF và hướng nhìn mục tiêu. | **100% nguyên bản (`act_dim = 3`)** | **ỨNG VIÊN #1 CHO PHASE 3 (cần đổi model FoV)** |

## 3. Phân định nguồn gốc khoa học & Tham số đang tắt (Chính sách `AGENTS.md`)

1. **Công thức gốc Q1/A\*:**
   - DS binary frame: Shafer (1976); Zhang et al., *IEEE TIV* 2024 (`10.1109/TIV.2024.3352581`).
   - Blackout / AoI / delayed sync: Xue et al., *Electronics* 2026 (`10.3390/electronics15091875`).
   - PHY/MAC timing 2.4 GHz / 2 Mbps: Zhou et al., *UavNetSim-v1*, *IEEE/CIC ICCC* 2025 (`arXiv:2507.09852`, GitHub `Zihao-Felix-Zhou/UavNetSim-v1` commit `c2703f2`).
   - Rotary-wing steady-level + Appendix A thrust-tilt $\kappa$: Zeng, Xu & Zhang, *IEEE TWC* 2019 (`10.1109/TWC.2019.2902559`, Eq. 12 & Eq. 69).
   - Urban LoS/NLoS benchmark: Al-Hourani et al., *IEEE WCL* 2014 (`10.1109/LWC.2014.2342736`).
   - MARL baselines (Phase 2): MATD3 (Ackermann 2019 / Fujimoto 2018) & MASAC (Haarnoja 2018).
2. **Mở rộng có mục đích của dự án:**
   - `DS SUM` trên bộ đếm raw `uint16` + module giảm thiểu tương quan AR(1) / Revisit Cooldown (`ds_correlation_mitigation.py`).
   - Khung dữ liệu nén `70 + 8k B` + HELLO V89 `110 B` + path-vector 7-bit trong Warp GPU DCF surrogate.
   - Feasible 66-candidate acceleration-disk shield + 3-sweep coordinate descent (không gán nhãn CBF).
3. **Heuristic kỹ thuật & Tham số đang tắt ở canonical `scenario_v103_ds_only_1km.yaml`:**
   - `enable_tx_power_control: false` (`tx_power_w = 0.1 W` cố định).
   - `enable_peer_dissemination: false`, `enable_data_mule_forwarding: false` (pure-GCS).
   - `w_p2p: 0.0`, `lambda_b: 0.0`, `continuous_aoi_penalty_w: 0.0`, `ds_aoi_eta: 0.0`.
   - `enable_accel_energy: false` (giữ Zeng Eq. 12 steady-level làm canonical; báo cáo ablation `True` đã sửa `C83`).
   - `rf_per_building_loss_db: 12.0` (single-obstacle shadowing; đã chứng minh bất biến nhiệm vụ khi nâng lên `20/23 dB` Al-Hourani).

## 4. Kết quả kiểm chứng V122 & Lệnh chốt cuối trên Vast.ai RTX 3060 trước Phase 2

- **RED → GREEN V122 (`tests/test_v122_scenario_fixes_and_gates.py`):** 3/4 FAIL trước vá (`C83`, `C84`, `ds_correlation_mitigation`) → **4/4 PASS** sau vá.
- **CPU Unittest hồi quy:** Toàn bộ các test CPU không phụ thuộc `torch`/`warp` đều **PASS** (`0` lỗi logic; 30/30 file `uav_network/algorithms/` giữ nguyên fingerprint `96d57cd9...`; 2 YAML active giữ nguyên SHA256).
- **Manifest & Policy:** `verify_source_manifests.py --check` **PASS**, `check_project_policy.py --scope full` **0 vi phạm**.
- **2 lệnh cuối trên Vast.ai RTX 3060 để lật `SCENARIO_FROZEN = true` và khởi động Phase 2:**
  ```bash
  export PYTHONPATH=research/zhang_xue/src:.
  python -m pytest -q tests/ research/zhang_xue/tests/
  python scripts/gpu/run_v115_canonical_ds_baseline.py --algorithm matd3 --allow-experimental-physics --dry-run --output outputs/phase2_smoke_matd3.json
  ```
