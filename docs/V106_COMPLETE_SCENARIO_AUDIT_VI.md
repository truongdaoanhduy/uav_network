# BÁO CÁO KIỂM TOÁN & TRIỂN KHAI V106 — HOÀN THIỆN CÁC CỔNG S15–S24 (DS-ONLY, SCENARIO-FIRST)

- **Nền kế thừa**: `UAV_V105_FULL_SOURCE_WITH_S13_EVIDENCE.zip` (đã giữ nguyên toàn bộ bằng chứng `outputs/s13_vast/`).
- **Nguyên tắc giữ chặt**:
  - **DS-only active**: Không sử dụng Bayesian inference trong kịch bản active.
  - **TX power cố định `0.1 W`**: Không đưa `tx_power` vào action của RL baseline.
  - **SCENARIO-FIRST**: Giữ nguyên **30/30 file `uav_network/algorithms/` byte-identical**; `S25` (tối ưu VRAM replay buffer) để sang Phase 2 sau khi khóa kịch bản.
- **Trạng thái `SCENARIO_FROZEN`**: **CHƯA KHÓA (`NOT SCENARIO_FROZEN`)** cho đến khi chạy xong cổng **`S24` (full DS-only pytest + CUDA stress + `compute-sanitizer` trên Vast.ai `cuda:0`)**.

---

## 1. S15 — Phân loại & chuyển đổi toàn bộ 25 file tham chiếu Bayes cũ (không xóa bừa bãi)

- **Nhóm A (14 file chức năng bị trỏ nhầm vào YAML Bayes → đã chuyển sang `scenario_v103_ds_only_1km.yaml`)**:
  - Tests: `test_v93_invalid_actions.py`, `test_v93_one_step_viability.py`, `test_v93_peer_causality.py`, `test_v102_warp_energy_runtime.py` (kèm override `enable_peer_dissemination=true` cho test PEER mask và DS parity), `test_v92_scenario_freeze_audit.py`, `test_v95_s11_configuration.py`, `test_v96_scenario_freeze_gates.py`.
  - Scripts: `v91_discrete_action_utility_probe.py`, `v91_network_differential_audit.py`, `v91_oracle_feasibility_probe.py`, `v92_non_oracle_mission_probe.py`, **`v93_mission_audit.py` (đã sửa lỗi crash ẩn `AttributeError: p.bayes_prior_occupancy`)**, `v99_freeze_evidence_decision.py` (gỡ case `s11_bayes_shared_cov` khỏi `required_cases`).
- **Nhóm B (Bayes-rejection & historical boundary tests → giữ lại và cập nhật assertion đúng hợp đồng V106 DS-only)**:
  - `test_v91_bayesian_belief_contract.py` (xóa import hỏng `_fuse_belief_v91`; khẳng định Warp từ chối `v91`/`v95` Bayes fail-closed), `test_v96_s11_end_to_end_runtime.py` (khẳng định `v91`/`v95` Bayes fail-closed + chạy runtime DS S11), cùng `test_v90`, `test_v93_probe_metrics`, `test_v94`, `test_v103`, `test_v104`.
- **Nhóm C (Legacy Bayes-only probes → gắn nhãn `[LEGACY_BAYES_ONLY]`)**:
  - `scripts/gpu/v91_belief_reliability_audit.py`, `scripts/gpu/v98_scenario_ablation_suite.py`.

---

## 2. S16 & S18 — Monte Carlo Calibration (50,000 trials/cấu hình, $P_d=0.85, P_f=0.03$, ngưỡng `m_T >= 0.90, m_U <= 0.20`)

- **Cảm biến độc lập ($\rho=0.0$) — `paper_fixed (0.80, 0.80)`** (= `SLR cap 0.80`):
  - $N=1$: `TPR=0.000`, `FPR=0.000` (chống báo động giả 1-shot; đối chứng `SLR uncapped (0.9647, 0.8454)` có `FPR=0.0300` ngay ở $N=1$).
  - $N=2$: `TPR=0.7222`, `FPR=0.00098`, `Brier=0.05104`.
  - $N=5$: `TPR=0.8340`, `FPR=0.00000`, `Brier=0.01196`.
  - $N=10$: `TPR=0.9898`, `FPR=0.00000`, `Brier=0.00181`.
  - $N=20$: `TPR=0.9997`, `FPR=0.00000`, `Brier=0.00005`.
- **Cảm biến tương quan khi hover ($\rho=0.7$, Markov sticky)**:
  - `FPR` tăng vọt **21×** ở $N=2$ (`0.02054`) và **762×** ở $N=3$ (`0.01524`); ngay ở $N=10$ vẫn còn `1.17%` và Brier xấu đi **28×** (`0.05119`).
  - **Nghịch lý đồng thuận giả**: Mean $K_{\text{cum}}$ và mean disagreement đều **giảm** khi $\rho=0.7$ (vd. $N=5$: $K_{\text{cum}}$ giảm `0.2936 -> 0.1321`), chứng minh chỉ số conflict nội bộ không phát hiện được tương quan cảm biến — cần khai báo rõ giả định *conditionally independent sensor noise* trong bài báo.
- **Đối chứng `SLR + Denoeux (alpha=0.75 -> 0.7235, 0.6340)`**: Giảm Brier Score gấp **3.1×** tại $N=10, \rho=0$ (`0.00058` vs `0.00181`) mà không dính 1-hit FP.

---

## 3. S17 — Kiểm chứng toán học Reward khi evidence mâu thuẫn

Tại trạng thái mâu thuẫn cân bằng **`(10 pos, 10 neg)`** ($m_T=0.50, m_F=0.50, m_U \approx 0.00, K_{\text{cum}} \approx 1.00$):
- **`ds_ignorance` ($1 - m_U$)**: Thưởng tĩnh **`+1.0000`** (mù hoàn toàn trước mâu thuẫn).
- **`deng_entropy` ($1 - E_d / \log_2 3$)**: Chặn được bước `(1,0)->(1,1)` (`+0.0000`) và dồn thưởng mạnh vào bước xác nhận `(2,0)` (`+0.4626`), nhưng **vẫn rò rỉ `+0.3691`** tại `(10,10)` vì $E_d(0.5, 0.5, 0) = 1.0 < \log_2 3$, và có thể vượt `1.0` tại `(0.2, 0.2, 0.6)` ($\Phi=1.465$).
- **`pignistic_entropy` ($1 - H(\text{BetP})$)**: Thưởng tĩnh **`0.0000`** tại mọi mâu thuẫn cân bằng `(1,1), (5,5), (10,10)`, nhưng dưới watermark đơn điệu thì `(1,0)` vẫn khóa sẵn `+0.5310` và chặn thưởng bước sửa báo động giả `(1,2)`.
- **Quyết định V106**: Giữ mặc định `exploration_reward_mode: ds_ignorance` cho baseline (đúng công thức Zhang 2024, không thay đổi tùy tiện chỉ vì phức tạp hơn), giữ `deng_entropy` là cờ ablation opt-in, và nêu minh bạch giới hạn toán học này trong báo cáo.

---

## 4. S19, S20, S22, S23 — Kết quả kiểm chứng định lượng trên CPU

- **S19 (Action XY sensitivity, 3 seeds × 120 bước, fixed GCS comm)**:
  - `hover (0,0)`: `obs=5.50%`, `cov=5.50%`, `tp_act=2.00`, `reward=73.00`, `min_batt=87.77%`.
  - `random_xy`: `obs=12.83%`, `cov=12.67%`, `tp_act=2.33`, `reward=97.34`, `min_batt=88.92%`.
  - `non_oracle_explore_xy`: **`obs=70.75%` (12.9×)**, **`cov=67.50%` (12.3×)**, **`tp_act=8.67` (4.3×)**, **`reward=397.50` (5.45×)**, **`min_batt=89.83%`** (đúng Zeng 2019: cruise tiết kiệm pin hơn hover), `0` vi phạm an toàn.
- **S20 (Independent 2×2 Ablation `PEER × data_mule`, `noise=4.0e-11 W`, 3 seeds × 120 bước, quỹ đạo cố định `obs=36.75%`)**:
  - `(False, False)` và `(False, True)`: trùng khớp tuyệt đối (`tp=5.33, cov=31.50%, AoI=38.39s, reward=224.38`) → `data_mule` đơn lẻ có tác dụng `0.00`.
  - `(True, False)`: trao đổi `219.7` gói P2P nhưng không chuyển tiếp về GCS (`tp=5.33, reward=224.38, cov=32.25%`).
  - `(True, True)`: tương tác cộng hưởng thực sự kéo `tp=5.67` (`+0.34`), `cov=34.50%` (`+3.00 pp`), `AoI=36.14 s` (`-2.25 s`), `reward=234.13` (`+9.75`).
- **S22 (Dynamic Reachability qua `kinematic_safety_shield`)**: Cả **4/4 đích khó** (góc xa nhất `(975,975)` trong 60 bước; ô sát tòa nhà nhất `cid=106` cách mép `0.28 m` trong 12 bước; tâm Blackout #1 + quay ra GCS trong 26 bước; tâm Blackout #2 + quay ra GCS trong 58 bước) đều đạt mục tiêu, quan sát và giao GCS thành công với **`safety_violations == 0`**.
- **S23 (Runtime Energy Conservation, 60 bước / 360 UAV-steps đủ trạng thái)**: Sai số tổng 6 UAV giữa thực đo và tích phân công thức là **`2.11e-10 J`** (max mỗi UAV `1.69e-10 J << 1e-6 J`).

---

## 5. Phân biệt rõ nguồn khoa học vs điều chỉnh vs tham số tự thiết kế

| Thành phần | Phân loại | Cơ sở |
|---|---|---|
| Khung DS $\Theta=\{T,F\}$, $m_f=m_e=0.80$, reward $-\Delta m_U$ | **Từ paper** | Zhang et al. (*IEEE T-IV* 2024, Eq. 5–17, Table II–IV) |
| Gộp count `sum` theo origin provenance (thay cho `max` Eq. 16 của Zhang) | **Điều chỉnh từ paper** | Chống double-counting qua mạng đa chặng mà không mất thông tin khi 2 UAV độc lập cùng quan sát |
| Tắt global belief decay (`ds_aoi_eta=0.0`), đo AoI tách biệt | **Điều chỉnh từ paper** | Khác Xue et al. (*Electronics* 2026, Eq. 9–10) vì mục tiêu tĩnh trên horizon 900s ($0.995^{900}=0.011$) |
| Công suất bay cánh quạt rotary-wing | **Từ paper** | Zeng, Xu & Zhang (*IEEE TWC* 2019) |
| Tầng MAC CSMA/CA 802.11 DCF trên Warp GPU | **Surrogate (S21)** | Lấy cảm hứng từ UavNetSim-v1 (SimPy) & Bianchi (2000); đã kiểm chứng với cleanroom DES 24s, **không** tuyên bố packet-level parity với SimPy upstream |
| Hình học đô thị (`1km, 20×20, 5 trụ 70–80m, 12 dB/tòa, 2 blackout`), `confirm_th=0.90`, `r_tp=25`, `r_fp=12.5`, `tx_power=0.1W` fixed | **Tham số tự thiết kế (Design Choice)** | Phù hợp bài toán nghiên cứu, cần ghi rõ là scenario design choice trong bài báo |

---

## 6. Cổng cuối (`S24`) cần chạy trên Vast.ai RTX 3060 (`cuda:0`) trước khi chốt `SCENARIO_FROZEN`

```bash
unzip -q UAV_V106_DS_ONLY_SCENARIO_20261127.zip -d /tmp/v106_check && cd /tmp/v106_check/uav_network_V106
export PYTHONPATH=.:research/zhang_xue/src
python3 scripts/maintenance/verify_source_manifests.py --check && python3 scripts/maintenance/check_project_policy.py --scope full
# 1. Full DS-only pytest trên seluruh suite (sau khi dọn S15)
pytest -q tests/test_v91_bayesian_belief_contract.py tests/test_v92_scenario_freeze_audit.py tests/test_v93_invalid_actions.py tests/test_v93_one_step_viability.py tests/test_v93_peer_causality.py tests/test_v95_s11_configuration.py tests/test_v96_scenario_freeze_gates.py tests/test_v96_s11_end_to_end_runtime.py tests/test_v101_energy_conservation.py tests/test_v102_warp_energy_runtime.py tests/test_v103_ds_only_scenario.py tests/test_v104_scenario_audit.py tests/test_v105_complete_scenario_audit.py tests/test_v106_s15_s23_audit.py
# 2. CUDA Compute Sanitizer trên cả Base lẫn RF-Stress
compute-sanitizer --tool memcheck pytest -q tests/test_v89_spawn_geometry.py tests/test_v102_warp_energy_runtime.py -k cuda
# 3. CUDA Endurance 4096x900 cho cả Base và RF-Stress (4e-11 W, PEER=ON, data_mule=True)
python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml --device cuda:0 --envs 4096 --steps 900 --output /tmp/v106_base_endurance_cuda0.json
python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py --scenario research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml --device cuda:0 --envs 4096 --steps 900 --output /tmp/v106_rf_stress_endurance_cuda0.json
```
