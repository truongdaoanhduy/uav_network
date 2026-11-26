# V75 — KIỂM TOÁN ĐA GÓC NHÌN (Q1/A*, WARP GPU, CODEBASE) & BẢN VÁ ĐÃ KIỂM CHỨNG

**Ngày:** 2026-11-26 (UTC+07). **Nguồn đối chiếu:**
1. Log công khai Kaggle `therwed/notebookb4853d42db?scriptVersionId=374162716` (2× Tesla T4, PyTorch `2.11.0+cu130`, `warp-lang==1.18.0`, tổng thời gian `124.7s`, 5/5 job exit `0`, trạng thái `DIAGNOSTICS_PASS_NOT_FULL_RL`, `full_gpu_rl_training: false`).
2. Toàn bộ mã nguồn và bản vá `uav_v74_full_audit_source_tests_logs_20261126.zip` (`source_v74`, `v73_to_v74.patch`).
3. Đối chiếu độc lập bởi 3 góc nhìn: (i) Tài liệu khoa học Q1/A* (Zhang 2024, Xue 2026, Zhou 2025 UavNetSim-v1, Ng et al. ICML 1999, Yates 2021 IEEE JSAC), (ii) Kiến trúc GPU NVIDIA Warp & PyTorch CUDA Interop, và (iii) Kiểm toán mã nguồn & bản vá v74.

---

## 1. Đánh giá các phát hiện & bản sửa lỗi của v74: Đúng, Thiếu, hay Sai?

| Hạng mục v74 | Phán quyết | Phân tích lý do kỹ thuật & khoa học |
|---|---|---|
| **C01/C02**: Truyền PHY (`tx_power_w`, `cca`, `noise`, `snr`) & `decentralized_peer_select` vào Warp | **ĐÚNG nhưng THIẾU** | Sửa đúng lỗi bỏ qua YAML ở `capacity_probe.py`, nhưng bỏ sót 6 hợp đồng: (1) `ds_fusion_mode='zhang_max'` vẫn truyền `aoi_eta=0.02` thay vì `0.0` (lệch CPU); (2) chưa fail-closed cho `comm_action_mode != 'compact_3'`, `enable_tx_power_control=True`, `aoi_ewma_end_to_end=False`, `network_substeps != 1`, `coalesce_duplicate_queue=True`; (3) `packet_workload.py` vẫn dùng fixture LoS trống (không buildings/blackout). |
| **C03**: Chặn fail-closed 6 tham số MAC timing vì Warp compile-time | **ĐÚNG hướng phòng thủ, CHƯA TỐI ƯU** | Chặn chạy sai âm thầm là đúng; tuy nhiên 6 hằng số MAC chỉ xuất hiện ở 6 dòng trong `events_v67.py` & `mac_v67.py`. Đưa vào `SimDimsV67` (`wp.int32`) an toàn 100%, giúp Warp hỗ trợ động thay vì khóa cứng. |
| **C04**: Giữ `Q_V67=32` và chỉ ghi chú lệch với CPU `64` | **CHƯA ĐẦY ĐỦ** | Toàn bộ 5 kernel Warp đã dùng động `dims.q_cap`! Nâng `q_cap: 32 → 64` chỉ tốn `+116,256 B/env` (`+0.443 GiB` ở `E=4096`, tổng chỉ `2.279 GiB` cho `20×20`). Giới hạn cứng duy nhất là `q_cap <= 127` vì `active_slot` là `wp.int8`. |
| **B01–B03**: DS SUM + AoI EWMA vs Zhang (2024) & Xue (2026) | **ĐÚNG hướng, THIẾU định lượng** | Chứng minh toán học cho thấy count-power aging $1-(1-m_f)^w$ **luôn over-confident nghiêm ngặt** so với chiết khấu BPA trực tiếp $w m_f$ của Xue (2026): chênh tối đa $+0.1555$ tại $\Delta t^* \approx 41.70\text{ s}$ và tỷ số đạt $2.0118\times$ khi $w \to 0^+$. |
| **R02**: Reward khám phá `r_loc_team` dưới EWMA aging | **PHÁT HIỆN LỖI P0 (v74 chưa sửa)** | `transition.py:266-267` dùng `max(0.0, team_u_prev[cid] - u_now); team_u_prev[cid] = u_now` tạo **vòng lặp cày thưởng vô hạn (oscillatory reward farming)** khi `ds_aoi_eta=0.02`, vi phạm Định lý 1 Ng, Harada & Russell (ICML 1999). |
| **R01**: `w_p2p=0, lambda_b=0` trong scenario chính | **HỢP LÝ cho baseline, CẦN ABLATION** | Tránh peer-chatter reward hacking ở baseline nhiệm vụ, nhưng thiếu động lực tiết kiệm airtime và khó gán công 2-hop Data-Mule; bắt buộc có bảng ablation 5 nhóm cho paper Q1/A*. |
| **N01**: Tham số PHY "chuẩn UavNetSim" | **CẢNH BÁO SAI LỆCH 19.60 dB** | Table I của Zhou et al. (2025, arXiv:2507.09852) dùng nhiễu $N = 4 \times 10^{-11}\text{ W}$ ($-73.98\text{ dBm}$), còn YAML v74 dùng nhiễu nhiệt $4.3896 \times 10^{-13}\text{ W}$ ($-93.58\text{ dBm}$, $\text{NF}=7\text{ dB}$) — chênh $19.60\text{ dB}$ ($91.12\times$, tầm liên lạc FSPL ~9.5×). Phải ghi minh bạch trong paper. |
| **W01–W03**: Chẩn đoán Warp 4096 env / batch 8192 | **ĐÚNG, bổ sung nút thắt Replay** | `TensorReplayBuffer` đang lưu trùng cả `obs` lẫn `state = obs.reshape(-1)` dạng FP32 (tốn `13.82 GiB` ở 300k → OOM cùng learner). Chuyển sang state-derived FP16 giảm còn `3.46 GiB`. |

---

## 2. Các bản sửa lỗi đã triển khai trực tiếp trong v75

1. **V75-1 (`environment/transition.py:266-269`) — Chặn Reward Farming**: Đổi `team_u_prev[cid]` thành **monotone minimum-uncertainty watermark** (`if u_now < self.team_u_prev[cid]: ...`), triệt tiêu hoàn toàn việc bay lặp qua ô cũ để ăn thưởng vô hạn khi `ds_aoi_eta > 0`.
2. **V75-2 (`uav_network/experiments/radio_config.py`, `capacity_probe.py`) — Siết hợp đồng fail-closed**:
   - Sửa `ds_fusion_mode == 'zhang_max'` truyền `aoi_eta = 0.0` (khớp CPU).
   - Truyền tường minh `rx_lock_on_cca: False`.
   - Chặn fail-closed với `comm_action_mode != 'compact_3'`, `enable_tx_power_control=True`, `aoi_ewma_end_to_end=False`, `network_substeps != 1`, `coalesce_duplicate_queue=True`.
   - Thêm kiểm tra `scenario.enable_tx_power_control` trong `capacity_probe.py`.
3. **V75-3 (`gpu/layout_v67.py`, `gpu/runtime_v67.py`) — Tham số hóa `q_cap` động (`1..127`)**:
   - Hỗ trợ `q_cap` (mặc định `32` bảo toàn tương thích ngược, cho phép nâng lên `64` khớp CPU), có guard `1 <= q_cap <= 127` bảo vệ `active_slot: wp.int8` khỏi tràn âm.
   - Cập nhật tính VRAM `backing_bytes_per_environment_v67` và `assert_network_allocation_fits` theo `q_cap`.
4. **V75-4 (`gpu/state_v67.py`, `gpu/kernels/mac_v67.py`, `gpu/kernels/events_v67.py`, `gpu/runtime_v67.py`) — 6 tham số MAC động trong `SimDimsV67`**:
   - Đưa `difs_us`, `sifs_us`, `slot_us`, `cw_min`, `max_retx`, `ack_timeout_us` vào `SimDimsV67`, validate qua `CsmaProtocolConfig`.
   - Giữ mặc định `allow_dynamic_mac_timing=False` trong `radio_config_manifest` để 7/7 test v74 vẫn PASS; bật `allow_dynamic_mac_timing=True` để dùng 6 tham số MAC động.
5. **V75-5 (`configuration/loader.py`) — Sửa lỗi crash `bba_reliability_cap: null`**: Lấy tập trường cho phép `None` từ `EnvConfigV54()` thay vì ép kiểu `float` từ `make_v57_config()`.
6. **V75-6 (`scripts/gpu/packet_workload.py`, `tests/test_v75_audit_fixes.py`)**: Bổ sung tham số tùy chọn `buildings`, `blackout_centers`, `blackout_r`, `cell_size_m` cho `run_packet_workload()` và viết bộ kiểm thử `test_v75_audit_fixes.py`.

---

## 3. Lộ trình chuyển sang Full GPU Vectorized MARL bằng NVIDIA Warp (2× Tesla T4)

- **VRAM trên 2× Tesla T4 (16 GiB/GPU)**:
  - `20×20, q_cap=64, E=4096`: Radio SoA `2.279 GiB` + Env phụ trợ `0.063 GiB` = **`2.342 GiB`** (rất dư trên `cuda:0`).
  - `40×40, q_cap=64, E=4096`: **`6.256 GiB`** (vừa thoải mái trên `cuda:0`, ghép Chiến lược A: `cuda:0` Env+Radio+Actor mirror, `cuda:1` State-derived FP16 Replay + Learner).
  - `60×60, q_cap=64, E=4096`: **`12.779 GiB`** → sát ngưỡng OOM trên 1× T4; bắt buộc dùng Chiến lược B (`2×2048` env + DDP trên 2 GPU, ~`6.39 GiB` env/GPU).
- **5 Warp kernel cần port để xóa `DIAGNOSTICS_PASS_NOT_FULL_RL`**:
  1. `motion_shield_energy_kernel` (`dim=E`): `kinematic_safety_shield`, va chạm cặp, năng lượng Zeng 2019, cập nhật `radio_alive` và `compute_channel_gain_v67` tại chỗ.
  2. `sense_occluded_ewma_kernel` (`dim=(E, N)`): FoV + che khuất trụ 3D + cập nhật EWMA UQ8.8 `ledger_pos/neg/gen` không cần host guard.
  3. `project_semantic_obs_kernel` (`dim=(E, N, 100)`) + `build_telemetry_mask_kernel` (`dim=(E, N)`): semantic pooling 10×10 và mask ngay trên GPU.
  4. `gcs_latch_aoi_reward_kernel` (`dim=E`): GCS TP/FP latch, monotone watermark `r_loc_team`, map-wide AoI, `terminated`/`truncated`.
  5. `partial_reset_env_and_radio_kernel`: tự động reset theo `done_mask` sau khi ghi replay, xóa sạch hàng đợi, 3 vòng dedup, `now=0`, `ledger_gen=-1`, `ack_origin_gen=-1`.
- **0 host sync**: Dùng `wp.from_torch(..., return_ctype=True)` + `wp.stream_from_torch` + `wp.ScopedStream` + `@torch.no_grad()`, bỏ `.numpy()` khỏi hot-loop (chỉ kiểm tra sticky `overflow` mỗi 128–256 bước) và bọc chuỗi kernel bằng CUDA Graph sau khi đạt parity với CPU.

---

## 4. Trạng thái kiểm chứng trung thực (Sandbox CPU)

- Đã chạy và xác minh trực tiếp trong sandbox CPU: toàn bộ bài test `unittest` (bao gồm `test_v75_audit_fixes.py`), bộ kiểm thử `test_v74_radio_config_contract.py` (qua runner chuẩn), `check_project_policy.py --scope full` (`0 violations`), và `verify_source_manifests.py --check` (`PASS`).
- **Giới hạn trung thực**: Sandbox hiện tại không có GPU/CUDA, `torch`, hay `warp-lang`; do đó các kernel Warp (`events_v67.py`, `mac_v67.py`, `runtime_v67.py`) được kiểm định tĩnh + hợp đồng Python trên CPU và cần chạy xác minh trên Kaggle T4×2 (`warp-lang==1.18.0`, `torch>=2.11`) để đóng dấu CUDA G2+.
