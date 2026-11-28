# BÁO CÁO KIỂM TOÁN ĐA GÓC NHÌN & TRIỂN KHAI V103 — CHUYỂN KỊCH BẢN SANG DEMPSTER–SHAFER (DS-ONLY)

- **Phiên bản gốc**: `uav_network_V102` (Bayesian log-odds active)
- **Phiên bản mới**: `uav_network_V103` (**DS-only active**, `NOT SCENARIO_FROZEN` cho đến khi PASS đầy đủ cổng kiểm định trên Vast.ai `cuda:0`)
- **Nguyên tắc SCENARIO-FIRST**: Giữ nguyên **30/30 file** trong `uav_network/algorithms/` byte-identical (chưa sửa/tune `MATD3` và `MASAC`; giai đoạn baseline thực hiện sau khi khóa kịch bản).
- **Trung thực môi trường kiểm chứng**: Toàn bộ unit test và kiểm tra manifest/policy ở bản này chạy trên **CPU sandbox** (không có `torch`/`warp`/`cuda:0`). Các lệnh kiểm định trên **Vast.ai RTX 3060 `cuda:0`** được cung cấp sẵn ở Mục 6.

---

## 1. Tổng hợp các lỗi & xung đột nghiêm trọng phát hiện trong V102 (Đa góc nhìn)

### 1.1. Lỗi P0 trên CPU: Sai lệch ngữ nghĩa `raw_u16_count` vs `UQ8.8` (lệch hệ số 256×)
- **Nguyên nhân gốc**: Trong `scenario_v95_ds_shared_coverage_1km.yaml`, cả `static_evidence_mode: raw_u16_count`, `ds_aoi_eta: 0.0` và `aoi_ewma_end_to_end: true` đều được bật. Trong khi Warp kiểm tra cờ `static_raw_count` để lưu và đọc số nguyên `1, 2, 3...`, toàn bộ pipeline CPU lại chỉ kiểm tra `aoi_ewma_end_to_end` và bỏ qua `static_evidence_mode`:
  1. `communication/ledger.py:9-18` (`record_local_observation`): nhân `256` (UQ8.8) khi ghi vào `ledger_pos/neg` dù đang ở chế độ `raw_u16_count` (gây bão hòa sớm tại 255 lần thăm và lệch 256× trên dây so với Warp).
  2. `environment/observations.py:25-33` (`_fused_cell`): gọi `fuse_ewma_provenance_uq88` chia `256.0` trên dữ liệu `raw_u16_count`, khiến 1 quan sát positive cho $m(T) = 0.63\%$ thay vì $80.00\%$, 2 positive cho $1.25\%$ thay vì $96.00\%$.
  3. `environment/transition.py:253-276`: gọi `fuse_ewma_provenance_uq88` làm sai lệch team exploration reward `r_loc_team` và `local_tp_latch`.
  4. `communication/delivery.py:101-108`: gọi `fuse_ewma_provenance_uq88` trước `deng_entropy`, làm `step_peer_info_gain` sụp về gần 0.
- **Giải pháp V103**: Thêm thuộc tính `EnvConfigV54.use_uq88_ewma = bool(aoi_ewma_end_to_end and static_evidence_mode == "ewma_uq88")`, truyền `static_evidence_mode` xuống `SevenRadioCsmaCaNetwork`, và tạo hàm `fuse_raw_provenance_counts` dùng thống nhất cho cả 4 đường xử lý CPU.

### 1.2. Sai lệch Observation Schema giữa CPU (`514`) và Warp GPU (`549`)
- **Nguyên nhân**: Warp bổ sung 35 chiều telemetry non-oracle từ gói HELLO của 5 peer gần nhất (`mission_v89_hello_peer.py`, tổng 549 chiều khi `obs_schema_v2=true`), nhưng CPU `core.py:17` hardcode `obs_dim = 514` và bỏ qua `obs_schema_v2`.
- **Giải pháp V103**: Đồng bộ `obs_dim = 549` trên CPU khi `obs_schema_v2=true`; đính kèm `peer_pose` vào `PKT_HELLO_DV` trong `scheduler.py`, lưu `hello_rx_pose`/`hello_rx_at_us` tại `delivery.py`, và trích xuất đúng 35 chiều tại `observations._hello_peer_features` (với sentinel `[0,0,0,0,0,1.0,1.0]` cho peer vắng/hết hạn nhằm tránh giả lập va chạm 0 m).

### 1.3. Lỗ hổng Reward Hacking trên `peer_info_gain` và `team_u_prev`
- **Nguyên nhân**:
  1. `peer_h_min[rx, cid]` chỉ được cập nhật khi nhận gói từ peer (`delivery.py`), không cập nhật khi UAV tự cảm biến (`ledger.py`). Nếu UAV tự quét một ô đến khi entropy về 0 rồi nhận 1 gói cũ từ peer, hệ thống vẫn thưởng `peer_info_gain ≈ 1.0`.
  2. Nhánh `else` trong `transition.py:295` thiếu guard monotone `if u_now < self.team_u_prev[cid]:`.
- **Giải pháp V103**: Cập nhật `peer_h_min` ngay trong `record_local_observation` và áp dụng monotone watermark `if u_now < self.team_u_prev[cid]` thống nhất cho mọi nhánh trong `transition.py`.

### 1.4. Các lỗi & rủi ro trên NVIDIA Warp GPU (4096 envs song song)
- **Luồng Bayes vẫn tồn tại trong Warp**: `mission_v79_kernels.py`, `mission_v79_reset_obs.py`, `mission_v79.py`, `mission_v80.py` vẫn import `_bayes_from_counts_v91` và chạy nhánh Bayes.
  - *Giải pháp V103*: Gỡ bỏ hoàn toàn nhánh Bayes khỏi runtime Warp V79/V80 (`from_scenario()` fail-closed nếu `belief_inference_mode != "dempster_shafer"`).
- **Triệt tiêu chữ số FP32 ở `_ds_v79` khi $n_+, n_-$ lớn**: Công thức trực tiếp $(1-m_f)^{n_+}$ underflow về 0 từ $n \ge 65$ trên FP32 gây `0/0 = NaN`; ngay cả log-shift `a - shift` vẫn có sai số $\sim 1.34\times 10^{-5}$ tại $(5000, 4998)$.
  - *Giải pháp V103*: Bổ sung nhánh hiệu chính xác `(p - n) * logmf` khi `wp.abs(logmf - logme) <= 1.0e-7` vào cả `_ds_v79` (Warp) và `counts_to_canonical_ds_effective` (CPU), đạt độ chính xác sát machine epsilon FP32/FP64.
- **FP32 ULP nuốt năng lượng RF PA**: Ở `battery ≈ 180,000 J`, ULP của FP32 là `0.015625 J`. Phép trừ tách rời `battery - flight_j - pa_dc_w * air_s` làm mất hoàn toàn mọi bước có `air_s < 7.8 ms`.
  - *Giải pháp V103*: Gom thành `battery[e, u] - (flight_j + pa_dc_w * air_s)` tại cả `mission_v80_shield.py:453` và `mission_v79_kernels.py:230`.
- **Đụng độ RNG seed tuyến tính ở 4096 envs**: Công thức `seed + e * 4093 + ...` dễ trùng giữa các cặp `(e, ep_id, u, t)`.
  - *Giải pháp V103*: Bổ sung hàm trộn bit 32-bit `_mix_seed_v103` trong `mission_v79_kernels.py` và `mission_v79_reset_obs.py`.
- **`dirty_dest` cũ trước `resolve_compact3_v67`**: `mission_v79.py` và `mission_v80.py` không gọi `compute_dirty_dest_v67` trước `resolve_compact3_v67`.
  - *Giải pháp V103*: Gọi tường minh `compute_dirty_dest_v67` ngay trước `resolve_compact3_v67` trong `step()`.

### 1.5. Thiếu giám sát Conflict $K$, Deng Entropy và TP Lifecycle
- **Nguyên nhân**: Khi chuyển DS-only, $m(U)$ giảm ngay cả khi có bằng chứng mâu thuẫn (`1 pos + 1 neg` cho $m(U)=0.1111$ nhưng $K=0.64$). Ngoài ra `gcs_tp_latch` khóa vĩnh viễn làm lệch `precision/f1` nếu belief tụt xuống dưới ngưỡng.
- **Giải pháp V103**: Bổ sung `ds_cumulative_conflict` ($K_{\text{cum}} = (-\operatorname{expm1}(a))(-\operatorname{expm1}(b)) \in [0,1)$, $O(1)$ không underflow) và log đầy đủ trong `info`: `gcs_mean_conflict_k`, `gcs_max_conflict_k`, `gcs_high_conflict_cells`, `gcs_mean_uncertainty_u`, `gcs_mean_deng_entropy`, `gcs_tp_active_confirmed`, `gcs_tp_belief_drops`, `gcs_fp_ever_confirmed`, `gcs_fp_unlatched_count`.

---

## 2. Kiểm toán tầng mạng (UavNetSim-v1), `tx_power` và `PEER`

### 2.1. Đối chiếu với UavNetSim-v1 (`Zihao-Felix-Zhou/UavNetSim-v1`, arXiv:2507.09852v1)
- **Phạm vi trung thực**: Tầng mạng trong dự án là **Validated GPU DCF Surrogate** (không phải tương đương từng gói với SimPy upstream).
- **Các khác biệt chính cần khai báo minh bạch trong bài báo**:
  - Noise floor: dự án dùng thermal noise $-93.58\text{ dBm}$ ($4.39\times 10^{-13}\text{ W}$, $22\text{ MHz}, \text{NF}=7\text{ dB}$), thấp hơn mức $4\times 10^{-11}\text{ W}$ của UavNetSim-v1 ($-19.6\text{ dB}$), giúp tầm LoS đạt $2377.9\text{ m}$ trên bản đồ $1\text{ km}$.
  - Vật cản RF: `multi_obstacle` suy hao $12\text{ dB/tòa nhà}$ + 2 vùng blackout Manhattan (lựa chọn kịch bản nghiên cứu, không phải hiệu chuẩn vật liệu thực đo).
  - Routing: Path-Vector DV qua `HELLO_DV` + E2E ACK watermark non-oracle (thay cho Greedy/DSDV/OPAR của UavNetSim-v1).

### 2.2. Tại sao bỏ hành động `tx_power` và giữ cố định `0.1 W`
1. **Link budget**: Ở mức thấp nhất `0.05 W`, tầm LoS đã là $1681.4\text{ m} > 1414.2\text{ m}$ (đường chéo bản đồ). Trong blackout/opaque ($g=0$), tăng `tx_power` lên `0.5 W` vẫn cho $P_{\text{rx}}=0$ nhưng lại mở rộng bán kính CCA lên $2798.2\text{ m}$ gây đóng băng MAC toàn bản đồ.
2. **Năng lượng (Zeng et al., IEEE TWC 2019)**: Công suất bay + avionics là $141\text{–}183.5\text{ W}$, trong khi PA DC tại `0.1 W` là `1.0 W` với duty cycle $1\%\text{–}5\%$, chỉ chiếm **$0.005\%\text{–}0.036\%$** tổng năng lượng ($< 0.5\%$). Với `lambda_b = 0.0`, `tx_power` không tạo gradient hữu ích nào cho MARL.

### 2.3. Tại sao `PEER=OFF` trong cấu hình baseline V103 (`scenario_v103_ds_only_1km.yaml`)
- Trong benchmark V100, `PEER=ON` thua `PEER=OFF` do: (i) `resolve_compact3_v67` chọn peer theo min-hop mù không kiểm tra peer có đường về GCS hay không; (ii) pending-ACK lockout khi gửi GCS làm xả gói sang peer gây unicast gossip storm; (iii) **định tuyến đa chặng về GCS qua `HELLO_DV` vẫn hoạt động đầy đủ ngay cả khi `enable_peer_dissemination: false`**.
- Do đó, kịch bản baseline V103 đặt `enable_peer_dissemination: false`, `w_p2p: 0.0` để làm sạch baseline Phase 2 (canonical MATD3/MASAC); nhánh `PEER=ON` với `hop_to_gcs` quảng bá qua HELLO để dành cho Phase 3 (Hybrid Action Space / ablation).

---

## 3. Đối chiếu khoa học Q1/A* cho quyết định thiết kế V103

| Thành phần | Giá trị V103 | Cơ sở khoa học / Phân loại |
|---|---|---|
| DS Belief Map ($\Theta=\{T,F\}$) | Init $(0,0,1)$, closed-form log-space, origin-indexed counts | **Q1**: Zhang et al. (*IEEE T-IV* 2024); Shafer (1976). |
| Global Belief Decay $\gamma$ | **Tắt (`ds_aoi_eta: 0.0`)** cho mục tiêu tĩnh | **Q1**: Hu et al. (*IEEE TCST* 2013); Zhang et al. (*IEEE T-IV* 2024). Không dùng $\gamma=0.995$ của Xue (2026) vì sau 900 bước $0.995^{900}=0.011$ gây mất trí nhớ nhân tạo. |
| BBA ($m_f, m_e$) | `paper_fixed` ($0.80, 0.80$) mặc định; hỗ trợ `sensor_likelihood_ratio` + reliability cap | **Q1 + Design choice**: Zhang (2024) dùng $(0.8, 0.8)$; hiệu chuẩn từ $(P_d, P_f)$ theo Denoeux (*IJAR* 2014) với discounting $\alpha=0.75$ cho $(0.7235, 0.6340)$ là phương án ablation chuẩn. |
| Confirmation Rule | $m(T) \ge 0.90$ và $m(U) \le 0.20$ tại GCS (ngoài blackout) | **Q1 + Design choice**: Ngưỡng kép $(\sigma_1, \sigma_\phi)$ theo Xue (2026); $0.90$ yêu cầu $\Delta n \ge 2$ hit. |
| Năng lượng UAV | Zeng 2019 rotary-wing + avionics 15 W + PA DC tuyến tính (`0.1 W` cố định) | **Q1**: Zeng, Xu & Zhang (*IEEE TWC* 2019). |
| Action Space Phase 2 | Continuous 2D XY + fixed local comm policy cho canonical MATD3 / MASAC | **A***: Lowe et al. (NeurIPS 2017), Fujimoto et al. (ICML 2018), Haarnoja et al. (ICML 2018), Ackermann et al. (2019). |

---

## 4. Danh sách file đã sửa / thêm trong V103

- **Sửa (15 file)**:
  1. `research/zhang_xue/src/zhang_xue/belief/dempster_shafer.py`
  2. `research/zhang_xue/src/zhang_xue/environment/configuration.py`
  3. `research/zhang_xue/src/zhang_xue/environment/core.py`
  4. `research/zhang_xue/src/zhang_xue/environment/observations.py`
  5. `research/zhang_xue/src/zhang_xue/environment/transition.py`
  6. `research/zhang_xue/src/zhang_xue/communication/common.py`
  7. `research/zhang_xue/src/zhang_xue/communication/lifecycle.py`
  8. `research/zhang_xue/src/zhang_xue/communication/scheduler.py`
  9. `research/zhang_xue/src/zhang_xue/communication/ledger.py`
  10. `research/zhang_xue/src/zhang_xue/communication/delivery.py`
  11. `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py`
  12. `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_reset_obs.py`
  13. `research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py`
  14. `research/zhang_xue/src/zhang_xue/gpu/mission_v79.py`
  15. `research/zhang_xue/src/zhang_xue/gpu/mission_v80.py`
  16. `research/zhang_xue/configs/scenario_v95_ds_shared_coverage_1km.yaml`
  17. `tests/test_v101_energy_conservation.py`
- **Thêm mới (3 file)**:
  1. `research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml`
  2. `tests/test_v103_ds_only_scenario.py`
  3. `docs/V103_DS_ONLY_AUDIT_AND_FIXES_VI.md`
- **Giữ nguyên 100% byte-identical**: Toàn bộ **30/30 file** trong `uav_network/algorithms/`.

---

## 5. Kết quả kiểm chứng đã chạy trên CPU Sandbox

- `python3 scripts/maintenance/verify_source_manifests.py --check`: **PASS**
- `python3 scripts/maintenance/check_project_policy.py --scope full`: **PASS** (`0` vi phạm, mọi file `< 500` dòng)
- `unittest` (`test_v79_audit_and_parity_contracts`, `test_v101_energy_conservation`, `test_v103_ds_only_scenario`): **PASS (15/15)**
- Kiểm tra SHA256 `uav_network/algorithms/` (30/30 file): **PASS (byte-identical với V102)**
- **Lưu ý trung thực**: Các test yêu cầu `torch`/`warp`/`pytest` và chạy GPU `cuda:0` **chưa chạy** trong CPU sandbox (do không cài `torch`/`warp`/`pytest` và không có GPU).

---

## 6. Lệnh kiểm định bắt buộc trên Vast.ai RTX 3060 (`cuda:0`) trước khi chốt `SCENARIO_FROZEN`

```bash
unzip -q UAV_V103_DS_ONLY_SCENARIO_20261127.zip -d /tmp/v103_check && cd /tmp/v103_check/uav_network_V103
python3 scripts/maintenance/verify_source_manifests.py --check
python3 scripts/maintenance/check_project_policy.py --scope full
PYTHONPATH=.:research/zhang_xue/src pytest -q tests/test_v101_energy_conservation.py tests/test_v103_ds_only_scenario.py tests/test_v89_spawn_geometry.py tests/test_v80_feasible_shield_parity.py
python3 scripts/gpu/v98_simpy_vs_warp_mac_parity.py --device cuda:0 --des-us 24000000 --out /tmp/v103_s12_24s_parity_cuda0.json
python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py --config research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml --device cuda:0 --envs 4096 --steps 900
```
