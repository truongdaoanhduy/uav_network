# BÁO CÁO KIỂM TOÁN & TRIỂN KHAI V105 — HOÀN THIỆN CÁC CỔNG S06–S14 (DS-ONLY, SCENARIO-FIRST)

- **Nền kế thừa**: `UAV_V104_DS_ONLY_PARTIAL_AUDIT.zip` (đã giải quyết `S01–S05` với RED/GREEN).
- **Phạm vi V105**: Giải quyết dứt điểm và cung cấp bằng chứng định lượng cho các cổng mở **`S06, S07, S08, S09, S10, S11, S12, S14`** trên cả CPU lẫn mã nguồn NVIDIA Warp GPU; giữ nguyên **30/30 file `uav_network/algorithms/` byte-identical**.
- **Trạng thái `SCENARIO_FROZEN`**: **CHƯA KHÓA (`NOT SCENARIO_FROZEN`)** vì cổng **`S13` (Warp CUDA `4096 × 900` endurance & MAC parity trên Vast.ai RTX 3060 `cuda:0`)** chưa chạy thực tế trong CPU sandbox (không cài `torch`/`warp`).

---

## 1. Bảng tổng hợp kiểm chứng & giải pháp cho toàn bộ S06–S14

| ID | Vấn đề từ kiểm toán V104 | Kết quả đo thực tế / Chẩn đoán | Giải pháp đã triển khai trong V105 |
|---|---|---|---|
| **S06** | Semantic pooling (`max m_T, min m_F, mean m_U`) phá vỡ DS simplex $m_T+m_F+m_U=1$ | 1 target `(2,0)` + 3 ô unknown cho tổng $\Sigma = 0.96 + 0 + 0.76 = \mathbf{1.72} \in (0, 1.75]$. Mọi DS fusion/confirm/reward đều chạy trên lưới gốc $20\times 20$ (thỏa simplex tuyệt đối). | Giữ `semantic` pooling để bảo toàn đỉnh target thưa `15/400` cho actor (không làm loãng 4× như `mean`), ghi rõ contract **non-simplex actor feature** (không gọi là pooled BBA) ở `observations.py`, `mission_v79_reset_obs.py`, và khóa bằng unit test. |
| **S07** | Tích lũy quan sát lặp (correlated hover) gây overconfidence & $K_{\text{cum}}$ bão hòa $>0.99$ | `(5,5)` xung đột thật cho $K=0.9994$, nhưng hover trên ô rỗng `(3,97)` cũng cho $K=0.9920$ (không phân biệt được!). | Thêm `ds_disagreement_rate` $= \min(p,n)/(p+n) \in [0, 0.5]$ (`0.50` vs `0.03`) và log `gcs_mean_disagreement_rate`, `gcs_max_disagreement_rate`, `gcs_mean_eff_obs_count`, `gcs_max_eff_obs_count` (có đồng bộ decay `exp(-eta*dt)` khi dùng EWMA). |
| **S08** | Giảm ignorance $m_U$ vẫn thưởng nhầm khi có mâu thuẫn (`(1,0)->(1,1)` thưởng `+0.0889`) | Dưới `deng_entropy` chuẩn hóa $E_d/\log_2 3$ với monotone watermark `1.0`, bước `(1,0)->(1,1)` và `(2,1)->(2,2)` đều nhận **`0.0000`**, giữ nguyên ngân sách tối đa `1.0 * w_loc`. | Thêm `exploration_reward_mode: ds_ignorance \| deng_entropy` trên cả CPU (`configuration.py`, `transition.py`) và Warp (`mission_v79_kernels.py` `_deng_norm_v79`, `mission_v79.py`, `mission_v80.py`); mặc định giữ `ds_ignorance` cho backward parity. |
| **S09** | FP32 grouped subtraction chưa bảo toàn năng lượng RF PA qua 900 bước | Đo thực tế FP32 vs FP64: tại `air_s=0.001 s` ($v=10, 15\text{ m/s}$), **FP32 grouped vẫn nuốt 100.00% năng lượng RF PA (`0.90 J`)**, tổng sai số tới `2.75 J`! Dùng **Kahan residual accumulator** giảm sai số tổng xuống **`<= 0.0124 J`** (`< 1 ULP`) và sai lệch RF xuống **`<= 1.13%`**. | Thêm buffer `battery_comp` (Kahan compensated subtraction) vào `mission_v79.py`, `_reset_world_v79`, `_move_v79`, và `_move_v80` (`mission_v80_shield.py`), cập nhật `test_v101_energy_conservation.py`. |
| **S10** | Sai lệch ngữ nghĩa CPU vs Warp (spawn, obs, RNG, GPU TP active) | Phát hiện **4 lỗi thực tế**: (1) CPU spawn `65 m` cho `min_dist = 71.59 m < 95 m` braking margin (5/15 cặp vi phạm); (2) Warp `obs[12]` lệch **đúng 6.0×** khi `enable_data_mule=False`; (3) `_mix_seed_v103` dùng signed `int32` shift làm mất bit 31 và gây **323 va chạm** ở `T=100`; (4) GPU thiếu `gcs_tp_active` và `gcs_tp_drops`. | Sửa triệt để cả 4: (1) đồng bộ CPU spawn `120 m` (`123.69 m >= 95 m`); (2) sửa nhánh `dims.enable_data_mule == 0` cho `obs[12]` ở `mission_v79_reset_obs.py`; (3) đổi `_mix_seed_v103` sang `wp.uint32` logical shift + Knuth lẻ `2654435761` (`0` va chạm, bit 31 = `0.50`); (4) thêm `gcs_tp_active`, `gcs_tp_drops`, `enable_tp_unlatch` lên GPU (`mission_v79.py`, `mission_v79_kernels.py`, `mission_v80.py`). |
| **S11** | Kết luận `tx_power` vô nghĩa cần bằng chứng định lượng | Đo paired-seed (3 seeds × 120 bước): ở **V103-Base (`noise=4.39e-13 W`)**, tầng nhiệm vụ bão hòa (`tp_act=6.00`, `AoI=33.54s`), MAC `pdr_gcs` nhích `+4.3 pp` (`0.949→0.992`). Ở **noise UavNetSim (`4.0e-11 W`)**, tăng `0.05→0.50 W` làm `deliv_gcs` tăng **5.9×** (`76.3→452.3`), `tp_act` tăng **4.0×** (`1.33→5.33`). PA RF luôn `< 0.003%` tổng năng lượng. | Ghi nhận trung thực: cố định `tx_power_w = 0.1 W` là **lựa chọn phạm vi baseline (`V105-Base`)** dưới kênh RF thoáng (nếu `lambda_b=0`, học `tx_power` sẽ thoái hóa về công suất cực đại); bổ sung cấu hình stress-test `scenario_v105_ds_rf_stress_1km.yaml` (`4.0e-11 W`) cho khảo sát độ nhạy RF. |
| **S12** | `PEER=OFF` có thể làm suy yếu cooperative belief sharing | Đo paired-seed (3 seeds × 150 bước, heuristic non-oracle GCS-first): ở **V103-Base**, `PEER` kích hoạt **0.0 lần** (93.4% bước ngoài blackout đã có sẵn tuyến GCS). Ở **noise cao `4.0e-11 W`**, `PEER=ON + data_mule` tăng `deliv_gcs` **`+47.7%`**, `tp_act` **`+71.4%`** (`2.33→4.00`), giảm AoI **`-8.95 s`**, tăng reward **`+83.6%`** (`108.46→199.13`). | Giữ `PEER=OFF` trong `scenario_v103_ds_only_1km.yaml` cho Phase 2 baseline (canonical MATD3/MASAC liên tục 2D sạch), đồng thời cung cấp `scenario_v105_ds_rf_stress_1km.yaml` (`PEER=ON`, `data_mule=True`, `noise=4.0e-11 W`) làm benchmark mạng chia cắt chính thức. |
| **S14** | Kiểm chứng reachability, coverage, 3D occlusion & blackout toàn bản đồ | Đo trên lưới `20×20` (`5` tòa nhà, `2` blackout, `alt=50m`, `fov=50m`): **392/392 (100%)** ô legal đều reachable & observable (`0` ô bị 3D occlusion che hoàn toàn; BFS không gian bay an toàn đạt **100% liên thông** 1 thành phần). Cả 2 vùng blackout (`50` ô) đều cho phép sense + store-carry-forward và kết nối trực tiếp GCS (`18.37 dB` / `14.76 dB`) trong phạm vi `106.1 m` (`~7.1 s`) sau khi rời vùng. | Khóa kiểm chứng 392/392 ô observable vào `tests/test_v105_complete_scenario_audit.py`. |
| **S13** | Endurance `4096 × 900` & MAC parity trên Vast.ai `cuda:0` | **CHƯA CHẠY** trong CPU sandbox (thiếu `warp`/`torch`/`cuda:0`). | Cung cấp bộ lệnh CLI đã xác minh bên dưới để chạy trực tiếp trên Vast.ai RTX 3060 `cuda:0`. |

---

## 2. Lệnh chạy xác nhận cổng cuối (`S13`) trên Vast.ai RTX 3060 (`cuda:0`)

```bash
unzip -q UAV_V105_DS_ONLY_SCENARIO_20261127.zip -d /tmp/v105_check && cd /tmp/v105_check/uav_network_V105
export PYTHONPATH=.:research/zhang_xue/src
python3 scripts/maintenance/verify_source_manifests.py --check
python3 scripts/maintenance/check_project_policy.py --scope full
pytest -q tests/test_v105_complete_scenario_audit.py tests/test_v104_scenario_audit.py tests/test_v103_ds_only_scenario.py tests/test_v101_energy_conservation.py tests/test_v89_spawn_geometry.py
python3 scripts/gpu/v98_simpy_vs_warp_mac_parity.py --device cuda:0 --des-us 24000000 --output /tmp/v105_s12_24s_parity_cuda0.json
python3 scripts/gpu/v99_endurance_sticky_cuda_gate.py --scenario research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml --device cuda:0 --envs 4096 --steps 900 --output /tmp/v105_endurance_cuda0.json
```
