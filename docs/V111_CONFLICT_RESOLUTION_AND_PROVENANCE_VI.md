# Kiểm Toán Khoa Học Toàn Diện, Xử Lý Xung Đột (V109/V110 → V111) & Nguồn Gốc Paper

**Trạng thái khóa kịch bản:** `NOT SCENARIO_FROZEN` (chờ chạy thực nghiệm tách nhân tố C04/C05/C09/C14 và đồng bộ cổng freeze cuối).  
**Toàn vẹn thuật toán Phase 2:** **30/30 tệp** trong `uav_network/algorithms/` giữ nguyên **byte-identical** (chưa sửa MATD3/MASAC gốc).  
**Bằng chứng GPU V109 được bảo toàn nguyên vẹn:** `scenario_v103_ds_only_1km.yaml` (`3fcfe434...fc5`) và `scenario_v105_ds_rf_stress_1km.yaml` (`3ee084e7...b76`) giữ nguyên 100%, nên kết quả V109 trên Vast.ai RTX 3060 (`587 passed`, `compute-sanitizer 0 errors`, 2× endurance `4096×900` PASS với `0` safety/overflow/nonfinite) vẫn có hiệu lực đầy đủ. Toàn bộ 23 tài liệu kiểm toán reviewer V110 được bảo toàn tại `outputs/v110_reviewer_audit/`.

---

## 1. Các lỗi thật & xung đột đã được sửa trực tiếp trong mã nguồn V111

1. **Sửa `C19` (Lỗi lệch giao diện Observation `549/3` vs Adapter `514/7` & hợp nhất ABI CPU/GPU):**
   - **Nguyên nhân:** `research/zhang_xue/src/zhang_xue/learners/actor_adapter.py` trước đây khóa cứng `RAW_DIM=514`, gây lỗi khi nhận kịch bản active `obs_schema_v2=true` (`549` chiều + `compact_3` mask `3` chiều), trong khi `run_v81_real_rollout.py` và `episode_metrics_v87.py` tự viết lặp lại hàm `encoded_obs()` nội tuyến.
   - **Giải pháp V111 (không sửa 30/30 file `uav_network/algorithms/`):**
     - Mở rộng `encode_actor_input_scalable` trong `actor_adapter.py` hỗ trợ cả `514` (legacy) và `549` (active) $\to$ **`553` chiều** (`state_dim = 3318`).
     - Thêm `encode_actor_input_torch(raw_obs, route_mask)` làm nguồn chân lý duy nhất (Single Source of Truth) và nối vào `scripts/gpu/run_v81_real_rollout.py` cùng `scripts/gpu/episode_metrics_v87.py`.
     - **Đính chính quan trọng:** `encoded[..., 9]` (motion alive mask) bắt buộc đọc từ `obs[..., 4] > 0` (`soc > 0`), **không** đọc từ `obs[..., 5]` (`own_radio and own_alive`), vì `obs[5] = 0` khi UAV bay vào vùng blackout dù vẫn còn pin — nếu đọc `obs[5]` sẽ khiến UAV đứng chết trong vùng mất sóng.
2. **Sửa `C20` (Tách nhiễu yếu tố giữa Base và RF-Stress bằng 4 file YAML độc lập):**
   - **Nguyên nhân:** `scenario_v103_ds_only_1km.yaml` (Base) và `scenario_v105_ds_rf_stress_1km.yaml` (RF-Stress) đổi đồng thời 3 biến (`mac_noise_power_w`, `enable_peer_dissemination`, `enable_data_mule_forwarding`), làm mất khả năng quy kết nhân quả.
   - **Giải pháp V111:** Giữ nguyên `v103` và `v105`, bổ sung 4 file YAML tách đúng **1 biến độc lập/cặp**:
     - `scenario_v111_ds_rf_stress_no_peer_1km.yaml`: `4.0e-11 W`, `PEER=false`, `MULE=false` (so với `v103` $\to$ tách thuần **RF noise**).
     - `scenario_v111_ds_rf_stress_peer_only_1km.yaml`: `4.0e-11 W`, `PEER=true`, `MULE=false` (so với `rf_stress_no_peer` $\to$ tách thuần **PEER**; so với `v105` $\to$ tách thuần **Data Mule**).
     - `scenario_v111_ds_base_peer_only_1km.yaml`: `4.39e-13 W`, `PEER=true`, `MULE=false` (so với `v103` $\to$ đo overhead **PEER** ở Open-RF).
     - `scenario_v111_ds_ablation_th099_1km.yaml`: Base + `confirm_th: 0.99` (tách thuần yêu cầu $p - n \ge 3$ so với $p - n \ge 2$).
3. **Bổ sung `C27` (Kiểm chứng tự động chống rò rỉ Ground Truth — Non-Oracle Taint Test) & chứng minh đại số `0.90 ≡ 0.95`:**
   - `tests/test_v111_adapter_taint_and_factorial.py` kiểm chứng tự động: hoán đổi `true_targets` ngoài tầm nhìn cảm biến giữ nguyên `obs (549)`, `encoded_obs (553)`, `mask (3)` và `state (3318)` giống hệt bit-for-bit.
   - Chứng minh đại số trên tập đếm nguyên $(p, n)$ với $m_f=m_e=0{,}80$: khi $p-n=1$ thì $m(T) \le 0{,}8333$; khi $p-n=2$ thì $m(T) \ge 0{,}9600$. Vì vậy **`confirm_th = 0.90` và `0.95` tương đương 100%** (chỉ `0.99` mới đẩy lên $p-n \ge 3$).
4. **Sửa 2 vi phạm đóng gói governance của V109:**
   - Gỡ bỏ thư mục `HISTORICAL_ARCHIVES/*.zip` ra khỏi cây mã nguồn (nguyên nhân gây 2 lỗi trong `check_project_policy.py`) và chuyển `REVIEWER_AUDIT_20261128/` sang `outputs/v110_reviewer_audit/`, khôi phục trạng thái **0 violations** trên toàn repo.

---

## 2. Phân tách minh bạch nguồn gốc khoa học (3 cột: Paper chuẩn vs Điều chỉnh vs Tự thiết kế / Vô hiệu hóa)

**Lưu ý phân hạng nguồn (Venue Hygiene):** Chỉ **Zhang et al. (IEEE T-IV 2024)**, **Zeng et al. (IEEE TWC 2019)** và **TD3/SAC (ICML 2018)** thuộc chuẩn **Q1/A\***. **Xue et al. (2026)** đăng trên *MDPI Electronics*, còn **UavNetSim-v1** là mã nguồn mô phỏng SimPy tham khảo.

| Thành phần | Lấy đúng từ paper | Điều chỉnh có chủ đích (Lý do & Câu hỏi reviewer) | Tự thiết kế / Vô hiệu hóa (Đánh giá) |
|---|---|---|---|
| **DS Belief & BBA** | Khung $\Theta=\{T,F\}$, công thức đóng $m_f=m_e=0{,}80$ (**Zhang 2024 [Q1]**, Eq. 3–17, Table II–IV); trần $m(U)\le 0{,}20$ (**Xue 2026**, Eq. 11). | Tính trong log-space (`counts_to_canonical_ds_stable`) để chống tràn số FP32. **Reviewer sẽ hỏi:** $m_f=m_e=0{,}80$ chưa hiệu chuẩn theo `Pd=0.85, Pf=0.03`. | Cảm biến Bernoulli + che khuất tia 3D quanh 5 tòa nhà trụ (hợp lý cho mô hình 2.5D). Nhánh Bayesian legacy CPU cần cô lập là `ARCHIVE`. |
| **Tổng hợp đa UAV (Fusion)** | Tính giao hoán, kết hợp (**Zhang 2024**, Theorem 1; **Xue 2026**, Remark 1). | **`SUM` theo origin-ledger** thay vì `MAX` (**Zhang 2024**, Eq. 16) để không mất bằng chứng độc lập từ UAV thứ 2 khi gửi về GCS. **Reviewer sẽ hỏi:** nguy cơ đếm trùng/tương quan lặp ($\rho>0$). | Origin-indexed ledger + E2E ACK watermark (đóng góp kỹ thuật cốt lõi nối mạng gói tin với DS). |
| **AoI & Xác nhận mục tiêu** | Ràng buộc truyền thông trước khi xác nhận (**Xue 2026**, Eq. 18). | **`ds_aoi_eta = 0.0`**, bỏ global decay $\gamma=0{,}995$ (**Xue 2026**, Eq. 9–10) vì bài toán là **15 mục tiêu tĩnh**. **Không được claim** là tái lập Xue AoI-weighted DS. | Thưởng trễ xác nhận lần đầu `+6*exp(-latency/60)` (đặt tên đúng: *bounded first-confirmation latency bonus*). **`continuous_aoi_penalty_w = 0`** (tắt). |
| **Action & TX Power** | Continuous 2D XY (**Zhang 2024**, Eq. 1–2); `tx_power_w = 0.1 W` cố định là tham số radio (**UavNetSim-v1** `utils/config.py`); công suất bay $P(V)$ (**Zeng 2019 [Q1]**, Eq. 12). | Safety shield chiếu động học ($v_{\max}=15, a_{\max}=5$) thay cho phạt mềm; cần nhấn mạnh là *kinematic projection*, **không phải CBF**. | **TX power cố định 0.1 W KHÔNG phải lỗi.** Legacy 3rd TX head trong `actor_adapter.py` bị khóa fail-closed. **`enable_accel_energy = false`** (tắt). Action `0` nên gọi là **`NO_APP_SEND`** thay vì `SILENT` (vì HELLO/ACK vẫn chạy). |
| **Mạng PHY/MAC/Routing** | Tham số IEEE 802.11 DCF (`CCA -82 dBm`, DIFS/SIFS/slot, `CWmin=31`, `max_retries=5`) theo **UavNetSim-v1**. | **Warp GPU MAC surrogate** thay SimPy để chạy 4096 env song song. **Bắt buộc gọi là** *"UavNetSim-inspired GPU surrogate"*, chưa có event-level parity với SimPy upstream. | `rf_per_building_loss_db = 12.0 dB` (tham số stress tự chọn, cần sweep `0/6/12/20 dB`). `PEER/MULE` cần chứng minh nhân quả bằng bộ YAML V111. |
| **Hàm thưởng (Reward)** | Thưởng xác nhận mục tiêu + phạt an toàn (**Zhang 2024**, Eq. 28–33). | **`first_visit`** (+0.5/ô mới) thay `ds_ignorance` (Eq. 27) để dập triệt để hover-farming. | **`w_p2p = 0.0`**, **`lambda_b = 0.0`** đang **vô hiệu hóa** — tuyệt đối không đưa vào phương trình reward active trong bài báo nếu không bật lên để chạy ablation. |
| **Observation & CTDE** | CTDE + canonical **TD3/SAC (ICML 2018 [A\*])**. | `obs = 549` (non-oracle) $\to$ encode `553` (`state = 3318`). | `obs_pool_mode = 'semantic'` ($5\times 10\times 10$) không cộng bằng 1 $\to$ gọi là *pooled spatial belief features*, không gọi là DS BBA. |

---

## 3. Việc còn lại để chốt `SCENARIO_FROZEN` trước Phase 2 (`MATD3`/`MASAC` canonical)

1. Chạy bộ test V111 + full pytest trên Vast.ai `cuda:0` (`python3 -m pytest -q tests/ research/zhang_xue/tests/`).
2. Chạy thực nghiệm tách nhân tố **C20/C09** trên 4 cấu hình V111 (`v103`, `rf_stress_no_peer`, `rf_stress_peer_only`, `v105`) với $\ge 5$ paired seeds × 900 bước để xác nhận lợi ích nhân quả của `PEER` và `Data Mule`.
3. Chạy stress test **C14** (adverse/random boundary actions ngoài `gentle_patrol`) và **C04/C05** (so sánh mission-level `confirm_th=0.90` vs `0.99` dưới nhiễu tương quan).
4. Sau khi chốt `SCENARIO_FROZEN`, mở Phase 2 bằng đúng lệnh tường minh:
   `python3 scripts/gpu/run_v81_real_rollout.py --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml --baseline-mode canonical_continuous --config configs/learners/v85_canonical_matd3_masac_baseline.yaml` (sau khi xử lý bài toán VRAM replay buffer `S25`).
