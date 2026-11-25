# Báo cáo Kiểm định Đa góc nhìn v5.6 & Bản nâng cấp v5.7 (Q1/A*, NVIDIA Warp GPU & QA)

Ngày: 25/11/2026
Phạm vi: Kiểm định toàn diện mã nguồn `v5.6` và tài liệu bàn giao `v5.5` theo 3 góc nhìn độc lập (Phản biện tạp chí Q1/A*, Kỹ sư NVIDIA Warp GPU & Mạng 7-radio, Kỹ sư QA Logic/Conflict), sau đó triển khai và kiểm thử bản nâng cấp **v5.7**.

---

## 1. Phán quyết ĐÚNG / ĐÚNG MỘT PHẦN / SAI đối với các giải pháp của v5.6

| Thành phần trong v5.6 | Phán quyết | Lý do khoa học / kỹ thuật | Hành động trong v5.7 |
| :--- | :---: | :--- | :--- |
| **Tách vai trò 6 Actor + 1 GCS, không rò rỉ `true_targets` vào Actor** | **ĐÚNG** | Bảo đảm chuẩn Dec-POMDP và không tính airtime GCS vào pin UAV. | **Giữ nguyên**. |
| **Lõi DS log-space hiệu số nguyên (`counts_to_canonical_ds_stable`) & đối chứng `zhang_max` vs `sum`** | **ĐÚNG** | Ổn định số tới `n=65535`, khớp [Zhang et al., IEEE TIV 2024](https://www.semanticscholar.org/paper/Enhancing-Multi-UAV-Reconnaissance-and-Search-DDPG-Zhang-Lin/c3962200ccb4224ae66a28a26f9bd424f4e1a7f9) Eq. 16 và [Xue et al., Electronics 2026](https://www.mdpi.com/2079-9292/15/9/1875) Eq. 6–7. | **Giữ nguyên**, bổ sung `max_mass` calibration cho `sensor_likelihood_ratio` (tránh 1 false alarm `0.9647 >= 0.90` kích hoạt FP latch ngay lập tức). |
| **Chiết khấu AoI trên snapshot cộng dồn (`ledger_rx_age_s`)** | **ĐÚNG MỘT PHẦN** | Gây hiện tượng **trẻ hóa giả tạo** (1 mẫu mới xóa tuổi của 5 mẫu cũ) và tụt niềm tin nghịch lý khi gói mới đến trễ; chưa khớp lão hóa từng mẫu trong [Xue et al., Electronics 2026](https://www.mdpi.com/2079-9292/15/9/1875) Eq. 9–10. | Bổ sung `ewma_effective_counts_update` & `counts_to_canonical_ds_effective` giữ trạng thái $O(1)$ nhưng khớp toán học chính xác với suy giảm lũy thừa từng mẫu đo. |
| **Actor Observation 514d (`20×20 → 10×10` mean pooling)** | **ĐÚNG MỘT PHẦN** | Mean pooling pha loãng 4× đỉnh mục tiêu đơn ô (`0.96 → 0.24`); `obs[u,11]` (`dirty_gcs`) bị cắt trần ngầm ở `62/400 = 0.155`. | Bổ sung `obs_pool_mode='semantic'` (`max` cho $m_T$, `min` cho $m_F$, `mean` cho $m_U$, `max` cho AoI) và sửa `dirty_gcs` đếm trên toàn bộ 400 ô (B11). |
| **Hard Veto đặt cứng $v = 0$ khi va chạm** | **SAI (P0)** | Vi phạm `a_max = 5 m/s²` gấp 5 lần ($25 \to 0$ trong 1s), phá vỡ động học phanh ($d_{\text{brake}} = 62.5\text{ m} > 50\text{ m}$) và làm gãy gradient Critic ([Fox et al., IEEE RAM 1997](https://publications.ri.cmu.edu/storage/publications/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf); [Dalal et al., arXiv:1801.08757](https://arxiv.org/html/1801.08757v1); [Cheng et al., AAAI 2019](https://www.researchgate.net/publication/335678506_End-to-End_Safe_Reinforcement_Learning_through_Barrier_Functions_for_Safety-Critical_Continuous_Control_Tasks)). | Triển khai `kinematic_safety_shield` (`safety_mode='kinematic_shield'`) ép nghiêm ngặt $\|\mathbf{v}_{\text{safe}} - \mathbf{v}_{\text{prev}}\|/\Delta t \le a_{\max}$ và phạt can thiệp mượt. |
| **Cảm biến 2D bỏ qua che khuất 3D qua tòa nhà 70–80 m** | **SAI (P0)** | UAV bay `z=50m` nhìn xuyên qua tòa nhà cao `70–80m` trong khi sóng 2.4 GHz bị chính tòa nhà đó chắn ([arXiv:2303.03197](https://arxiv.org/pdf/2303.03197)). | Bật `enable_3d_sensing_occlusion=True` tái sử dụng `segment_hits_cylinder_3d_fp32` và loại các ô nằm trong chân tòa nhà. |
| **Mô hình năng lượng Zeng 2019 & Pin `180 kJ` vs Horizon `1500 s`** | **ĐÚNG MỘT PHẦN** | Khớp [Zeng et al., IEEE TWC 2019](https://www.researchgate.net/publication/324328894_Energy_Minimization_for_Wireless_Communication_With_Rotary-Wing_UAV) Eq. 12 nhưng thiếu hạng tử gia tốc, và `180 kJ` chỉ bay tối đa `1276.5 s` nên `truncated` ở `1500 s` là dead code. | Bổ sung `zeng_propulsion_power_with_accel_w` ($\kappa = \sqrt{1+(\|a\|/g)^2}$ + $\Delta KE$) và preset `make_v57_config(e_max_j=270_000.0)` (75 Wh). |
| **Hàm thưởng (`r_peer_information`, `r_loc_team`, `r_gcs_aoi`, `r_fp_hold`)** | **ĐÚNG MỘT PHẦN** | `r_peer_information` cho phép farm tới `+3.0/bước` bằng ô trống (125 bước bằng cả 15 TP); `r_loc_team` bỏ qua `zhang_max`; `r_gcs_aoi` bão hòa ở `60 s`; `gcs_fp_latch` không thể sửa sai. | Sửa `r_loc_team` tôn trọng `zhang_max`; thêm `peer_reward_mode='info_gain'`, `aoi_reward_mode='exponential'`, `enable_fp_unlatch=True`. |
| **Mạng 7-radio CPU & Warp CSMA/CA** | **SAI ở các lỗi P0** | Lỗi khóa thu tín hiệu dưới sàn nhiễu (`p_rx > 0`, B1), Warp lấy `max` thay vì `sum` nhiễu (B2), HOL blocking (B3), UAV hết pin vẫn phát HELLO/relay (B4), Warp không gửi lại `E2E_ACK` khi DATA trùng (B5), quên `enable_backward=False` làm compile 68.86s, quét 96 pending mỗi $\mu\text{s}$ event làm timeout E=32, HELLO đọc live DV thay vì snapshot. | **Đã sửa toàn bộ trong v5.7** trên cả CPU (`uav_v54_7radio_csmaca_sim.py`) và Warp (`v56_warp_*.py` + kernel PHY 3D NLoS mới `v57_warp_phy_kernels.py`). |

---

## 2. Nguyên nhân gốc rễ nút thắt NVIDIA Warp (Compile 69s, Timeout E=32 & Lộ trình 4096 envs trên 2× T4)

1. **Compile `v56_warp_event_kernels.py` tốn 68.86s:** Do quên khai báo `wp.set_module_options({"enable_backward": False})` và `@wp.kernel(enable_backward=False)` trong `v56_warp_event_kernels.py` và `v56_warp_injection_kernels.py`, buộc NVIDIA Warp sinh toàn bộ mã reverse-mode autodiff (adjoint) cho vòng `while` đơn khối ([NVIDIA Warp Docs](https://nvidia.github.io/warp/)). Ngoài ra `v56_warp_7radio_mac.py` còn giữ 4 kernel trùng lặp chết (B14). **Đã sửa xong trong v5.7.**
2. **E=32 timeout 75s trên Warp CPU:** Theo [NVIDIA Warp FAQ](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/faq.rst), Warp CPU chạy `dim=E` **tuần tự trên 1 luồng**; cộng thêm việc quét 96 phần tử `pending_expiry` bên trong **mỗi** sự kiện vi giây (tới 393,216 lần đọc/env/bước) và không `break` trong các vòng 62-entry. **v5.7 đã hoist vòng pending ra ngoài `while` và thêm `break`.**
3. **Chiến lược chạy 4096 envs trên 2× NVIDIA Tesla T4 (2048 envs / GPU 16 GiB):**
   - Kiểm toán chính xác SoA v5.6 = `563,200 B/env` (`1.074 GiB` / 2048 envs), trong đó 95.64% nằm ở 10 mảng payload/ledger/ACK.
   - Với `StateDerivedReplayBuffer` (bỏ lưu trùng `state`, capacity `300k/GPU` ≈ `3.56 GiB`) + nén SoA v5.7 (`315,710 B/env` ≈ `0.602 GiB/2048 envs`) + PyTorch/CUDA overhead (`~3.0 GiB`), tổng VRAM ước tính **~7.16 GiB / 16 GiB T4** (an toàn, xem `v57_gpu_memory_budget()` trong `src/v57_warp_phy_kernels.py`).
   - **Khuyến nghị kiến trúc kế tiếp cho Gate 4 trên T4:** Chuyển dần trục `E` vào trong cùng `(..., E)` hoặc dùng Warp tile/coalesced SoA để tránh đọc rải rác trên Turing SM (`sm_75`, [GPU Kernels 101 on Tesla T4](https://nandigamharikrishna.substack.com/p/gpu-kernels-101-what-a-gpu-actually)) và chạy `scripts/v56_cuda_preflight.py --device cuda:0 --envs 32 --allow-experimental` rồi tăng dần `256 → 1024 → 2048`.

---

## 3. Kết quả kiểm thử v5.7

- Bộ test hồi quy đầy đủ (`test_v54*`, `test_v55_memory`, `test_v55_observation_ds_reward`, `test_v56_warp_gate`, `test_v57_audit_and_fixes`): **47 PASS, 0 FAIL, 1 SKIP** (do container không cài `warp-lang`; xem `reports/v57_regression_unittest.log`).
- `test_v55_memory.py` (trước đây lỗi `ModuleNotFoundError: torch`) nay chạy **PASS** nhờ sửa lazy-import `torch` trong `src/v55_state_derived_replay.py`.
- Đã loại bỏ lỗi assertion tautology trong `tests/test_v55_observation_ds_reward.py`.
