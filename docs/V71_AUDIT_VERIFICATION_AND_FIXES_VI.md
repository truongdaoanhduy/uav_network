# Báo cáo Kiểm định Độc lập v7.0 & Các bản vá Kỹ thuật v7.1

**Phạm vi trung thực**: Toàn bộ kiểm định & regression test trong bản v7.1 được thực thi trên **CPU sandbox** (`Python 3.11`, `NumPy 2.1.3`, `SciPy 1.15.2`; không có `torch`, `warp-lang`, CUDA GPU). Trạng thái CUDA/GPU T4×2 và huấn luyện 4096 env song song **vẫn giữ nguyên trạng thái `BLOCKED_NO_CUDA_HOST`**, tuyệt đối không khai báo GPU PASS khi chưa có trace chạy thực trên NVIDIA T4.

## 1. Đánh giá độc lập 8 bản sửa của v7.0

| # | Mục v7.0 tuyên bố | Phán quyết | Lý do kỹ thuật & Xử lý bổ sung trong v7.1 |
|---|---|---|---|
| 1 | Warp RX packet `tx_node` (`V70-01`) | **ĐÚNG** *(chưa chạy CUDA)* | Truyền `tx_node` trước khi xóa `rx_lock` là đúng. Vẫn cần chạy thật trên CUDA. |
| 2 | PHY/CCA default (`V70-02`) | **ĐÚNG** | Đổi `rx_lock_on_cca=False` khớp ngưỡng giải mã SNR của CPU `channel.py:83`. |
| 3 | E2E ACK pending slot (`V70-03`) | **ĐÚNG MỘT PHẦN** | Đã chặn ghi đè slot pending, nhưng `compute_dirty_dest_v67` không bỏ qua đích đang bị khóa pending → v7.1 đã sửa để `SHARE_PEERS` chọn peer khả dụng khác. |
| 4 | Warp event loop overflow (`V70-04`) | **ĐÚNG MỘT PHẦN** | Có 3 sót: (a) off-by-one `ev >= max_ev` khi hoàn tất đúng sự kiện thứ `max_ev`, (b) ghi đè `s.now[e] = end_t` ngay cả khi overflow, (c) 2 lệnh `.numpy()` gây D2H stall mỗi step → v7.1 sửa cờ `completed`, không ghi hỏng `now` khi overflow, thêm `check_host_guards`. |
| 5 | Warp reset & numpy import (`V70-05/06`) | **ĐÚNG MỘT PHẦN** | Đã thêm `import numpy as np` và xóa state NumPy ref, nhưng thiếu check `done.shape == (self.envs,)` trong `runtime_v67.reset` → v7.1 đã bổ sung. |
| 6 | Input validation (`V70-07`) | **ĐÚNG MỘT PHẦN** | Từ chối `bool` cho `detected` và `np.integer` cho `t_s`, không chặn `grid_w < 1`, `gain` âm/NaN, và truyền zero-sized `buildings=[]` trực tiếp vào Warp → v7.1 đã sửa toàn bộ. |
| 7 | GPU memory check (`V70-08`) | **ĐÚNG MỘT PHẦN** | `assert_network_allocation_fits` chưa được gọi trong `WarpCompressedRadioNetworkV67.__init__`; `network_memory_report.py` đánh dấu `15.854 GiB` là `possible=True` trên T4 16 GiB → v7.1 gọi guard ngay trong `__init__` (CUDA) và dùng mốc thực tế `15.0 GiB` + `fits_t4_with_3gib_reserve`. |
| 8 | CUDA smoke 6 UAV | **ĐÚNG MỘT PHẦN** *(chưa chạy CUDA)* | Chỉ assert `overflow == 0`, không assert `stats_peer`, `stats_gcs`, `stats_mac_ack`, `stats_e2e` → v7.1 đã thêm đủ 4 assert giao gói thực tế. |

## 2. Các lỗi bổ sung phát hiện và đã vá trực tiếp trong v7.1

### A. Kernel NVIDIA Warp (`research/zhang_xue/src/zhang_xue/gpu/kernels/`)
1. **Data-Mule store-carry-forward bị xóa gói trong `events_v67.py`**: Trước v7.1, khi `nh < 0`, kernel gọi `_qpop_pos_v67` xóa luôn gói khỏi hàng đợi. v7.1 thay bằng `_select_routable_slot_v67` (dọn gói hết TTL, ưu tiên `PKT_E2E_ACK`, giữ nguyên gói chưa có route).
2. **Thiếu Exponential Backoff & off-by-one `MAX_RETX` (`mac_v67.py`, `events_v67.py`)**: Bổ sung `cw = ((CW_MIN + 1) << retries) - 1`, sửa `ret >= MAX_RETX`, và xóa route qua `next_hop` hỏng khi drop do hết retry.
3. **Thiếu Half-Duplex RX corruption & thu hồi route HELLO (`mac_v67.py`)**: Đánh dấu `rx_corrupt = 1` khi radio bắt đầu phát trong lúc đang khóa thu; thu hồi route đứt trên `HELLO_DV`.
4. **Tràn bit dấu `int32` (`inject_v67.py`)**: Ép kiểu `wp.uint32(...)` trước mọi phép dịch bit `<<` trong `_pack_w0_v67` và `w1`.
5. **Data race ghi mảng `gain` (`phy_v67.py`)**: Mỗi thread `(e, i)` chỉ ghi hàng `gain[e, i, :]` của riêng mình, bỏ ghi chéo `gain[e, j, i]`.

### B. Mạng UavNetSim CPU (`communication/`)
6. **Aggregate CCA (`channel.py`)**: Tính năng lượng kênh bận theo tổng công suất `sum(P_tx * gain) + noise_power_w >= cca_th_w` thay vì kiểm từng máy phát đơn lẻ.
7. **Thu hồi route hỏng (`delivery.py`, `scheduler.py`)**: Xóa ngay route đi qua `prev_hop` khi biến mất khỏi `HELLO_DV` và xóa route qua `active_next_hop` khi `drop_retx`. Truyền `ds_fusion_mode` vào `fuse_ewma_provenance_uq88`.
8. **Chống tràn FIFO trùng lặp & chống đói Data-Mule (`ledger.py`)**: Thêm *queue coalescing* trong `enqueue_app_data`, xoay vòng `origin` (round-robin) và ưu tiên ô mới nhất trước khi cắt `MAX_ENTRIES_PER_PKT = 62`.

### C. Môi trường DS–AoI, Reward, Safety Shield & MARL (`environment/`, `mobility/`, `learners/`)
9. **Lệch đồng hồ 1 giây khi hợp nhất DS tại GCS (`observations.py`, `transition.py`)**: Truyền `now_s=float(t_step_s)` vào `_fused_cell(self.gcs_id, cid, now_s=...)`.
10. **Phạt lặp FP, thiếu metric Q1 & EWMA `r_loc_team` âm (`core.py`, `transition.py`, `configuration.py`)**: Thêm `gcs_fp_ever_latched` (chống phạt lặp `-r_fp` khi bật `enable_fp_unlatch`), clamp `r_loc_team >= 0`, bổ sung `precision`, `recall`, `f1`, `false_confirm_events`, `false_confirm_rate`, cùng 2 cờ nghiên cứu `reward_scale_mode` (`raw`/`per_target_norm`) và `continuous_aoi_penalty_w`.
11. **Oracle Leak trong `compact_3` (`observations.py`, `transition.py`)**: Thêm cờ `compact3_decentralized_peer_select` chọn peer thuần cục bộ theo bảng định tuyến HELLO thay vì đọc tọa độ/pin toàn cục.
12. **Safety shield & công suất RF PA (`flight_safety_energy.py`, `transition.py`, `actor_adapter.py`)**: Bỏ điều kiện miễn trừ phanh trong vùng `<= 1 m` quanh tòa nhà; truyền `tx_power_w=float(c.tx_power_w)` khi cố định TX để PA power đúng `1.0 W` ở `0.10 W` TX (thay vì `3.0 W`); bỏ hardcode `ACTOR_N=6` trong `decode_tx_power_w`.

## 3. Lộ trình tiếp theo trên GPU CUDA (2× Tesla T4 16 GiB)
1. Chạy `scripts/gpu/warp_packet_cuda_gate.py --envs 4 --uavs 6 --grid 20 --device cuda:0` trên máy có `torch + warp-lang + CUDA`, xác nhận 4 assert giao gói thực tế + kiểm tra `device_wide_delta_bytes`.
2. Đối chiếu packet-event trace 1-1 giữa `WarpCompressedRadioNetworkV67` và CPU `SevenRadioCsmaCaNetwork` (và kiểm định phân phối PDR/delay/retry với [UavNetSim-v1](https://github.com/Zihao-Felix-Zhou/UavNetSim-v1)).
3. Đưa toàn bộ vòng lặp môi trường (kinematic shield, multi-cell FOV sensing, DS-AoI EWMA, reward, observation, boolean mask auto-reset) lên `@wp.kernel` bọc trong `wp.ScopedCapture` + `wp.to_torch()` zero-copy và shard 2×2048 envs trên 2× Tesla T4 bằng `torch.distributed` (NCCL).
