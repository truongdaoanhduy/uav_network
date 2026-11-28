# V120 — Nguyên Nhân Gốc & Phương Pháp Giải Quyết Toàn Bộ 72 Xung Đột (`C01–C72`)

- **Nền tảng kiểm toán:** Kế thừa nguyên vẹn mã nguồn, kernel Warp đã vá ABI (`0.5085 m/s → 3.59e-7 m/s`), 2 tệp YAML active (`3fcfe434...fc5`, `3ee084e7...b76`), và **30/30 tệp trong `uav_network/algorithms/` giữ nguyên 100% byte-identical** từ V119.
- **Trạng thái khóa kịch bản:** `SCENARIO_FROZEN = false` (fail-closed đúng quy trình cho tới khi bạn duyệt các mục **Khóa phạm vi bài báo** và chạy 2 gate GPU cuối trên Vast.ai RTX 3060).
- **Thực thi thực tế trong V120 (CPU):**
  - `tests/test_v120_conflict_resolution_gates.py`: **5/5 PASS** (`C15, C16, C27, C39, C40`).
  - `scripts/maintenance/v120_ds_correlated_mission_calibration.py`: **45 lượt chạy ghép cặp** (`3 seeds × 300 steps`) cho `C04, C05, C23, C35, C41, C66`.
  - `scripts/maintenance/v120_cpu_rf_substeps_energy_calibration.py`: **45 lượt chạy ghép cặp** + tính cận trên 5 seed `[44..48]` cho `C08, C12, C13, C24, C28`.

---

## 1. Phân loại toàn bộ 72 xung đột (`C01–C72`) từ V119 → V120

| Nhóm trạng thái tại V119 | Số lượng | Danh sách ID | Hành động & kết quả tại V120 |
|---|---:|---|---|
| **Đã FIXED / CLOSED** | 15 | `C01, C02, C03, C06, C17, C37, C42, C48, C50, C51, C52, C53, C68, C69, C70` | Giữ nguyên, không hồi quy; `C16/C69` đã khóa hash kiểm tra tự động. |
| **Quyết định thiết kế / Đã ghi nhận** | 12 | `C10, C15, C16, C21, C22, C24, C30, C43, C45, C47, C49, C54` | Bổ sung đo đạc thực tế `C24` (`comm=0` tốn `1512 ms` HELLO), test tự động `C15/C16`. |
| **Đã loại bỏ cùng nhánh V117** | 9 | `C56–C64` | Không đưa trở lại canonical; `C60` (stale HELLO/route TTL) vẫn được ghi nhận ở lớp mạng gốc. |
| **Đang MỞ / PARTIAL (36 mục)** | **36** | **13 P0:** `C04, C07, C14, C18, C20, C31, C33, C34, C38, C55, C65, C66, C67`<br>**23 P1/P2:** `C05, C08, C09, C11, C12, C13, C19, C23, C25, C26, C27, C28, C29, C32, C35, C36, C39, C40, C41, C44, C46, C71, C72` | **16 mục** đã có số liệu hiệu chuẩn / unit test PASS mới trong V120; **6 mục** đóng bằng khóa phạm vi bài báo + pure-GCS; **6 mục** cần 2 lệnh GPU cuối trên Vast.ai; phần còn lại là ablation Phase 2 sau freeze. |

---

## 2. Kết quả hiệu chuẩn & kiểm chứng thực thi trong V120

### 2.1. Nhóm DS Belief, Sensing, Threshold & Mass (`C04, C05, C23, C35, C41, C66`)
Nguồn dữ liệu: `outputs/v120_conflict_resolution/c04_c05_c23_c35_c41_c66_ds_calibration.json` (3 seeds `[44, 45, 46]`, 300 steps, 45 runs):

| Thí nghiệm | Cấu hình | `rho` | `mean_gcs_tp_ever` | `mean_gcs_fp_ever` | `mean_gcs_tp_belief_drops` | Kết luận giải quyết |
|---|---|---:|---:|---:|---:|---|
| `C04/C41/C66` | `fusion_sum` | 0.0 | 1.667 | **0.000** | 1.333 | i.i.d. không phát sinh báo động giả. |
| `C04/C41/C66` | `fusion_sum` | 0.5 | 1.667 | **1.333** | 0.000 | Tương quan Markov làm tăng mạnh FP-ever. |
| `C04/C41/C66` | `fusion_sum` | 0.7 | 1.667 | **2.333** | 0.000 | Báo động giả cao dù cùng quỹ đạo. |
| `C04/C41/C66` | `fusion_zhang_max` | 0.7 | 1.667 | **2.333** | 0.000 | `zhang_max` **không** triệt tiêu FP do 1 UAV vẫn lặp quan sát trên ô của chính nó. |
| `C05/C35` | `confirm_th = 0.90` | 0.0 / 0.7 | 1.667 / 1.667 | 0.000 / 2.333 | 1.333 / 0.000 | Giữ `0.90` canonical. |
| `C05/C35` | `confirm_th = 0.95` | 0.0 / 0.7 | 1.667 / 1.667 | 0.000 / 2.333 | 1.333 / 0.000 | **Tương đương 100% với `0.90`** về mặt đại số. |
| `C05/C35` | `confirm_th = 0.99` | 0.0 / 0.7 | 1.667 / 1.667 | 0.000 / 2.000 | 0.333 / 0.000 | Chỉ giảm nhẹ FP ở $\rho=0.7$ trong khi V119 giảm −2,67 pp delivery. |
| `C23` | `paper_fixed (0.80/0.80)` | 0.0 | 1.667 | **0.000** | 1.333 | Trùng khớp `sensor_lr (cap=0.80)`. |
| `C23` | `sensor_lr (cap=0.99)` | 0.0 | 1.667 | **1.667** | 0.000 | $m_f=0.9647 \ge 0.90 \Rightarrow$ **1 lần nhiễu giả đã chốt FP ngay**, chứng minh `cap=0.80` an toàn hơn hẳn. |

### 2.2. Nhóm Vật lý, Năng lượng, RF Building Loss, Substeps & Control Plane (`C08, C12, C13, C24, C28`)
Nguồn dữ liệu: `outputs/v120_conflict_resolution/c08_c12_c13_c24_c28_rf_energy_calibration.json` (45 runs + 5-seed upper bound):

- **`C08/C12` (Khả đạt năng lượng Zeng 2019, `e_max_j = 180,000 J`):**
  - Công suất tổng (propulsion + 15 W avionics): Hover = **`183.49 W`** (endurance `981.0 s`), bay `10 m/s` = **`141.03 W`** (`1276.3 s`), bay `15 m/s` = **`153.55 W`** (`1172.3 s`).
  - Cận trên đa UAV cho **15 mục tiêu** trên cả 5 seed `[44..48]`: `oracle_makespan_s = 96.56–106.62 s` (**100% khả đạt trong 900 s**), năng lượng lớn nhất mỗi UAV = **`14.0–15.7 kJ`** (**7.78–8.71%** pin 180 kJ).
  - `enable_accel_energy` `False` vs `True`: `mean_min_battery_pct` đều `69.63%` (chênh lệch `< 0.01%`).
- **`C13` (`rf_per_building_loss_db` ∈ `{0, 6, 12, 20}` dB trên pure-GCS tách biệt khỏi PEER/MULE):**
  - Base pure-GCS: `mean_pdr_gcs = 0.937`, `mean_gcs_tp = 1.667`; RF-noise pure-GCS (`4e-11 W`): `mean_pdr_gcs = 0.9344`, `mean_tx_airtime_ms` tăng từ `3046.4 ms` → `5249.56 ms` do retx. Ghi nhận rõ `12 dB/tòa nhà` là tham số thiết kế tổng hợp.
- **`C28` (`network_substeps` `1` vs `2`):**
  - Chênh lệch `mean_gcs_tp` = `0.000`, `mean_pdr_gcs` = `0.0000`, airtime lệch `< 0.08%` (`5249.56 ms` vs `5245.66 ms`). Giữ `network_substeps = 1` và công bố cận sai số rời rạc hóa.
- **`C24` (`comm=0` control-plane):**
  - Khi ép toàn bộ `comm = 0` (`NO_APP_SEND`): `data_enqueued = 0`, `gcs_tp_ever = 0`, nhưng mặt phẳng điều khiển HELLO vẫn phát **`1512.0 ms`** airtime (**49,6%** tổng airtime Base `3046.4 ms`). Đổi thuật ngữ hành động 0 thành `NO_APP_SEND`.

### 2.3. Bộ kiểm thử tự động V120 (`tests/test_v120_conflict_resolution_gates.py` — 5/5 PASS)
1. **`C15` (`test_c15_underlying_ds_grid_preserves_simplex`)**: Khẳng định lưới DS gốc $20\times 20$ luôn thỏa $m_T+m_F+m_U = 1.0 \pm 10^{-6}$; tensor gộp $5\times 10\times 10$ được gắn nhãn rõ là đặc trưng tóm tắt cho actor.
2. **`C16` (`test_c16_active_yamls_ds_only_and_pinned_hashes`)**: Khóa cứng `belief_inference_mode == "dempster_shafer"`, `enable_tx_power_control == false`, `tx_power_w == 0.1` và đúng SHA256 của 2 YAML active.
3. **`C27` (`test_c27_multi_trajectory_non_oracle_sentinel_taint`)**: Hoán đổi `true_targets` ở mọi ô ngoài FoV trên 3 seed × 5 bước, khẳng định `max_abs_diff(obs) == 0.0`, `max_abs_diff(state) == 0.0`, `mask` trùng khớp tuyệt đối.
4. **`C39` (`test_c39_gcs_tp_latch_vs_active_belief_drop_contract`)**: Tiêm +2 chứng cứ dương ($m_T=0.96 \Rightarrow$ `latch=1, active=1`) rồi tiêm +2 âm tính $\Rightarrow$ `active=0`, `gcs_tp_belief_drops >= 1`, `latch=1`. Khóa định nghĩa bài báo là *ever-confirmed at GCS*.
5. **`C40` (`test_c40_blackout_exit_route_gated_exact_once_sync`)**: Trong tâm blackout không có tuyến (`mask[0,1]==0`, `gcs_ledger==0`); sau khi thoát blackout, HELLO khôi phục tuyến và đồng bộ đúng 1 lần (`ledger==1, ack_wm==1`). Khóa mô tả là *route-gated packet synchronization*.

---

## 3. Lộ trình chốt `SCENARIO_FROZEN = true` (Chỉ còn 2 bước quyết định + 2 lệnh GPU)

### Bước A — Khóa phạm vi tuyên bố khoa học (không cần sửa code)
1. **`C07 / C33 / C65`**: Tuyên bố rõ lớp mạng là **`UavNetSim-inspired IEEE 802.11 CSMA/CA DCF surrogate`** với khung dữ liệu DS nén (`70+8k B`), **không** tuyên bố tương đương từng sự kiện byte-level với SimPy upstream.
2. **`C04 / C66 / C41`**: Tuyên bố rõ DS `SUM` tại GCS giả định quan sát độc lập theo bước thời gian (kế thừa Zhang 2024) và công bố bảng nhạy cảm $\rho \in \{0, 0.5, 0.7\}$ từ V120.
3. **`C34 / C20 / C09`**: Khóa kịch bản canonical là **pure-GCS (`PEER=false, MULE=false`)** dựa trên bằng chứng ghép cặp V119 (`PEER` làm giảm −32,67 pp delivery dưới RF-stress).
4. **`C11 / C26`**: Khóa giao diện baseline Phase 2: canonical `MATD3/MASAC` học hành động liên tục 2D XY kết hợp luật truyền thông GCS tất định cố định (không trộn hybrid vào baseline gốc).

### Bước B — 2 lệnh GPU cuối trên Vast.ai RTX 3060 (đóng `C14/C36/C38/C55/C67/C68/C72`)
```bash
# 1. Chạy lại toàn bộ pytest + bộ test V120 mới trên môi trường GPU
PYTHONPATH=research/zhang_xue/src:. pytest -q tests/test_v120_conflict_resolution_gates.py tests/test_v119_canonical_launcher_pin.py tests/test_v88_active_scope_lock.py

# 2. Chạy gate sticky-horizon 900 bước liên tục (không reset) để đóng C14/C36/C72
PYTHONPATH=research/zhang_xue/src:. python scripts/gpu/v99_endurance_sticky_cuda_gate.py \
  --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml \
  --worlds 4096 --steps 900 --output outputs/v120_gpu/sticky_900_base.json
```
Sau khi hoàn tất Bước A và Bước B, kịch bản DS-only sẵn sàng chuyển sang `SCENARIO_FROZEN = true` để bắt đầu Phase 2 training baseline `MATD3` và `MASAC` gốc.
