# Báo cáo Kiểm định Phản biện v6.8 & Hợp nhất Toàn diện v6.9

Bản **v6.9** kế thừa toàn bộ nền tảng kiểm định QA có log thực tế (`outputs/qa_evidence/`), 3 bản vá `B04/B05/B06` và sửa đổi `channel.py` (RX lock theo `SINR >= 6 dB` khớp [UavNetSim](https://github.com/Zihao-Felix-Zhou/UavNetSim)) của v6.8, đồng thời khắc phục các lỗ hổng logic còn sót trong v6.8 và tích hợp bộ kernel Warp GPU thực thi đầy đủ.

## 1. Các lỗi v6.8 còn sót và đã sửa dứt điểm trong v6.9

1. **B17 (P0 — Đã sửa): `r_loc_team` bão hòa ở lần quan sát đầu tiên (`environment/transition.py`)**: v6.8 vẫn truyền số nguyên UQ8.8 (`256`) vào `counts_to_canonical_ds_stable()` khi `aoi_ewma_end_to_end=True`. v6.9 giải mã bằng `decode_uq88()` và dùng `counts_to_canonical_ds_effective()`.
2. **B18 (P1 — Đã sửa): Quan sát lệch thang đo (`environment/observations.py`, `configuration.py`)**: Sửa `pooled[4] *= factor * factor` cho lưới 40×40/60×60; kẹp `dirty_gcs <= 1.0`; cập nhật `obs[u, 12]` theo `ack_wm_origin_ts`; thêm cận trên `grid_w^2 <= 4096`; đặt `mac_queue_capacity=200` trong `make_uavnetsim_reference_config()`.
3. **B09 (P1 — Đã sửa): Bổ sung 5 metric chuẩn paper vào `step()` và `run_scenario.py`**: `local_tp_discovered`, `true_area_observed_pct`, `pdr_gcs` (với mẫu số riêng `data_enqueued_gcs`), `mean_queue_len`, và `discovery_auc`.
4. **B01 & B02 (P0 — Đã triển khai mã nguồn Warp, chờ benchmark CUDA T4)**: Tích hợp 9 mô-đun Warp v6.7 (`layout_v67.py`, `state_v67.py`, `runtime_v67.py`, `parity_v67_ref.py`, `kernels/*_v67.py`) với `WarpCompressedRadioNetworkV67` (đủ `reset`, `update_phy_from_positions`, `sense_batch`, `inject_compact3`, `step`, `snapshot`) và `ack_origin_gen` (`int16`, **201.600 B/env**, tiết kiệm **75%** so với 806.400 B/env; tổng SoA **480.686 B/env** = **0,917 GiB** ở 2.048 env).
5. **B16 (P2 — Đã chuẩn hóa)**: Cập nhật `docs/V69_BUG_STATUS.json`, `outputs/qa_evidence/memory_estimates.json`, và tái tạo log QA thực tế cho bản v6.9.

## 2. Trạng thái Gate trung thực
- **CPU / Policy / Manifest / NumPy SoA Parity**: **PASS** (log thực tế nằm trong `outputs/qa_evidence/logs/`).
- **GPU Gate (`q1_gpu_preflight.py`)**: Giữ trung thực trạng thái **`BLOCKED_NO_CUDA_HOST`** (exit code `3`) khi chạy trên máy không có CUDA/Warp — chỉ mở gate sau khi chạy đối chiếu trace trên GPU NVIDIA T4.
