# V5.8 — Báo cáo kiểm định độc lập và bản vá có điều kiện (25/11/2026)

**Đầu vào:** nguyên gốc ZIP `uav_v57_audited_physics_mac_warp_20261125.zip` do người dùng cung cấp. **Không đồng nhất** với repo 3 km đang training nếu chưa được merge/đối chiếu commit. Mọi đánh giá dưới đây là trên **mã trong ZIP**. Các paper chỉ là cơ sở mô hình/thí nghiệm, KHÔNG chứng nhận một implementation là đúng.

**Kết quả tái lập:** baseline `python -m pytest -q`: **49 PASS, 2 SKIP** (khác log 47/1 của tác giả do lựa chọn test runner/tập test). Sau các vá v5.8 và test mới: xem `reports/v58_pytest.log`. `warp-lang` không cài trong container đóng gói; không có CUDA/Tesla T4. Chưa đo hiệu năng GPU, chưa chạy MATD3/MASAC end-to-end.

## 1. Phán quyết từng thành phần v5.7

| ID | Kết luận | Evidence từ code | Nguyên nhân / hành động |
|---|---|---|---|
| P0-SCENARIO | **SAI nếu dùng để thay môi trường nghiên cứu 3 km** | `EnvConfigV54`, `make_v57_config`, `GRID_W`, `CANONICAL_BUILDINGS_5` | ZIP 1 km / 15 targets / 20² cells / 5 cylinders / v=25 / a=5; nghiên cứu cũ 3 km / 35 targets / 40² cells / 25 obstacles / v=20 / a=2. Không mặc nhiên gán cùng tên benchmark. **Chưa đổi ZIP sang 3 km** vì cần thay cả hệ tọa độ, occupancy, sensing, radio, Warp array, replay và test. Script `scripts/v58_scenario_audit.py` chặn nhầm lẫn này. |
| P0-SAFETY | **SAI nếu khẳng định tránh va chạm được bảo đảm** | `kinematic_safety_shield` | Mẫu hai UAV cách 60 m, bay ngược chiều 25 m/s, a=5 m/s²: trả về 20/-20 m/s, quãng cách còn 20 m sau 1 s (<50 m). **Đã thêm** `trajectory_safety_violations` kiểm tra swept paths, terminal flag và penalty vào environment v5.8. Chưa có bảo đảm tránh va chạm vật lý khi trạng thái không khả thi, phải lập kế hoạch trước (viability/braking) hoặc mô hình va chạm terminal. Không cài lại CBF. |
| P0-HOL | **SAI đối với khẳng định v5.7 đã sửa Warp B3** | `v56_warp_event_kernels._step`: chỉ đọc `q_head` rồi `break` nếu không route | **Đã sửa source v5.8**: tìm gói priority E2E ACK rồi DATA routable ở toàn bộ 64 slot, chuyển kèm ID, seq, TTL, origin, payload. **Chưa biên dịch/execution chứng minh với Warp**. Cần kiểm tra thêm fairness, TTL và parity do cách reorder gói có thể khác CPU. |
| P0-TRAIN | **CHƯA ĐƯỢC TRIỂN KHAI** | `train_v55_packet_cpu.py`, `v56_warp_*.py`, `v57_warp_phy_kernels.py` | Code có Warp network kernel và CPU smoke trainer phụ thuộc repo `uav_network`, nhưng chưa nối motion + sensing + reward + batched Actor/Critic trên GPU thành end-to-end training. Không có checkpoint học thật MATD3/MASAC từ ZIP. Không chạy 4096 env. |
| P1-DS | **ĐÚNG có giới hạn** | `counts_to_canonical_ds_stable` | Log-space giảm underflow và kiểm tra zero conflict; phép MAX và provenance-safe SUM là **hai thuật toán khác nhau**. Tham số `max_mass=0.8` là *cap heuristic* chưa phải chứng minh hiệu chuẩn thống kê Pd/Pf; phải ablate. |
| P1-AOI | **HÀM ĐÚNG, TÍCH HỢP SAI/THIẾU** | `ewma_effective_counts_update` có hàm độc lập nhưng `record_local_observation`, `pack_entry_2x_uint32`, `_fused_cell` vẫn truyền counts nguyên + một timestamp cuối | Không tái dựng chính xác tuổi từng measurement từ aggregate 8-byte. Cần versioned effective-count snapshots hoặc truyền per-observation. **Chưa âm thầm thay wire format** vì điều đó làm mất parity MAC, ACK watermark, Warp layout. |
| P1-BELIEF | **ĐÚNG một phần** | `obs_pool_mode='semantic'`, 514d | Max m(T) giữ đỉnh nhưng min m(F), mean m(U) không nhất thiết cộng đúng 1 trong mỗi patch; đây là *features*, không phải BBA hợp lệ cho DS tiếp theo. Actor không thấy lân cận/nhà tường riêng trong 514d, dù world có chướng ngại. Thử CNN, local occupancy mask, neighborhood vector, per-channel ablations. |
| P1-OBST | **HỢP LÝ CHO TẦNG SENSING, CHƯA ĐỦ 3D** | `segment_hits_cylinder_3d_fp32`; `positions` z cố định 50; actions (vx,vy) | Ground sensing ray 3D được chặn qua nhà, nhưng UAV không điều khiển cao độ, obstacles là cylinders cố định chứ không phải 25 vật thể trong benchmark gốc. Cần kiểm tra line segment, FOV vật lý, độ cao/PD/PF theo altitude, validity sensor mask. |
| P1-ENERGY | **HỆ SỐ MỞ RỘNG CHƯA KIỂM CHỨNG** | `zeng_propulsion_power_with_accel_w`; pin 270 kJ | P(V) từ Zeng steady level. `kappa`, delta KE, `m=2kg`, efficiency=0.70 là extension do code giả định; không mặc định chính là phương trình Zeng đã hiệu chuẩn. Pin 75 Wh có thể vẫn không kéo dài 1500 s tùy hành trình, TX, avionics. Hết pin giữa dt hiện chỉ được xử lý sau khi đã chạy mạng 1 s. |
| P1-REWARD | **CẦN ĐỊNH NGHĨA LẠI “INFO GAIN”** | `step_peer_info_gain` tính `1 / prev_total` sau khi tăng counts | Đây không phải độ giảm DS entropy và chưa chứng minh chống reward farming. `w_loc_team` dựa uncertainty giảm; r_gcs_tp dựa truth (được phép cho reward nhưng không Actor); FP latch được unlatch; AoI exp thưởng lúc TP confirm, không phải dense gradient cho các bước không xác nhận. Nên log mọi thành phần + kiểm tra cumulative exploit. |
| P1-PHY | **MÔ HÌNH RF ĐƠN GIẢN, KHÔNG PHẢI UavNetSim NGUYÊN BẢN** | `link_path_gain` + CPU event scheduler | Có CCA/DIFS/backoff, ACK, HELLO DV, interference. GCS là radio7, nhưng path-vector 7-bit tùy biến không phải đầy đủ DSDV/OPAR/Q-routing; MAC/PHY không chạy SimPy. Dùng UavNetSim gốc như *reference testbench*, đối chiếu cùng kịch bản; không tuyên bố là port bitwise. |
| P1-MEM | **V5.7 SAI SỐ CẤP PHÁT** | `v57_gpu_memory_budget()` hardcode 315710, nhưng `WarpSevenRadioNetwork` cấp phát `SHAPES` | **Đã sửa** tính từ chính `SHAPES`: **635931 bytes/env**, **1.213 GiB/2048 env/GPU**, riêng network. Replay 300k ~3.533 GiB và PyTorch 3 GiB là giả định/budget. Nén 315710 chỉ là *ý tưởng chưa hiện thực*. Đánh dấu CUDA peak chưa xác minh. |
| P1-POS | **MÔ PHỎNG KÊNH TĨNH TRONG MỖI BƯỚC** | `step`: `run_until_us(t+1s)` trước, sau đó mới cập nhật positions | Ở v=25 m/s, topology có thể đổi ~25 m trong thời gian gói đang đi. UavNetSim gốc có cấu hình cập nhật vị trí định kỳ. Cần thời gian microstep hoặc nội suy vị trí khi xử lý sự kiện, đối chiếu độ nhạy `dt_net=1.0,0.1,0.02` s. |
| P1-TX | **CHƯA có power-control action** | `EnvConfigV54.tx_power_w=0.5` đưa vào `SevenRadioCsmaCaNetwork`, không được học bởi Actor | Điểm nghiên cứu mạng muốn tối ưu TX power chưa được thực hiện. Cần cho Actor chọn mức TX, ràng buộc SINR/airtime/energy thật, và so sánh budget/coverage; nếu muốn nghiên cứu cũ max 0.4 W thì 0.5 W đã đổi điều kiện PHY. |
| P1-PD | **THIẾU phụ thuộc cao độ** | `uav_alt_m=50`, `pd=0.85,pf=0.03` hằng số | Khác bảng Pd/Pf theo alt [50,100,150] của thiết kế trước. Kiểm tra độ nhạy, không gọi là tái lập kịch bản gốc. |
| P1-TARGET | **CÓ BUG VÒNG LẶP VÔ HẠN** | `while placed < num_targets` và chỉ kiểm tra `num_targets<=400`, nhưng buildings chiếm nhiều cell | **Đã sửa** bounded placement, kiểm tra `legal_cells`, tùy chọn exclusion target–GCS và min target spacing; báo `ValueError` khi không đạt. Default 0 bảo tồn benchmark v5.7. |
| P2-TRAIN-INPUT | **ĐÚNG CÓ ĐIỀU KIỆN** | `StateDerivedReplayBuffer` chỉ phục hồi `state=obs.reshape()` | Phù hợp với critic chỉ dùng joint-obs; **không tương đương** centralized critic có bản đồ vật cản, LOS, true state. Nếu muốn CTDE full-state, cần thiết kế state/replay mới, không tiếp tục quảng bá tiết kiệm state như không mất thông tin. |
| P2-CONFIG | **Chưa có contract khoa học đóng băng tham số** | thay đổi `pd/pf`, obstacles, battery, speed, reward giữa versions | Tránh “training cùng run name nhưng khác kịch bản”: lưu `config_hash` + `git_sha` + `scenario_id` + Warp/cuda versions theo mỗi run (W&B/HF). |

## 2. Các chỉnh sửa ĐÃ thực hiện trên file (v5.8 candidate)

1. `src/uav_v54_ds_and_battery.py`: sửa docstring sai về “guarantees all safety”; thêm `trajectory_safety_violations` kiểm tra swept wall/building/pairwise paths, kể cả radio hết pin còn nằm như vật cản.
2. `src/uav_v54_7node_env.py`: tính và báo `safety_violation`, `safety_violations`, `r_safety_failure`; terminate episode khi shield bất khả thi; tránh reset radio hai lần; target placement có kiểm tra khả thi và ràng buộc khoảng cách tùy chọn.
3. `src/v56_warp_7radio_mac.py`, `src/v56_warp_event_kernels.py`: thêm ưu tiên và tìm gói có route ở sau queue, trao đổi nguyên metadata + toàn bộ 62 payload entries. Đã kiểm tra static, **CẦN Warp runtime compile/test**.
4. `src/v57_warp_phy_kernels.py`: thay con số nén tưởng tượng bằng byte size thực của SHAPES; giữ chi phí overhead CUDA dưới tên *assumed* và cờ `cuda_peak_verified=0`.
5. `tests/test_v58_p0_regressions.py`: thêm adversarial shield, fail-closed, target placement impossible, exclusion target–GCS, queue source contract, VRAM, AoI wiring check.
6. `scripts/v58_scenario_audit.py`: xuất manifest/mismatch và **`--strict` trả exit code 2** để khóa training khi gate chưa đạt.
7. `stage_v58_on_v57.sh`: chỉ tạo nhánh mới nếu các preimage đúng SHA-256 từ ZIP v5.7, không commit/push hay ghi đè một nhánh gốc đang dirty.

## 3. Kịch bản thống nhất đề xuất và các vấn đề phải quyết định

- **Benchmark A — reproduction:** khóa hoàn toàn 1 km, 15 targets, 5 cylinders, 20x20, fixed 50m, P(V) + packet-network, 6 UAV + GCS như ZIP v5.7. So sánh với chính baseline v5.7, *không* báo là Zhang/Xue reproduced numbers.
- **Benchmark B — nghiên cứu chính:** 3 km, 35 targets, 25 obstacles, 40x40 map, v_max=20, a_max=2, horizon1500, battery277200 J, 6 UAV + GCS, target–GCS exclusion ~150 m và khoảng cách target–target ~250 m theo thiết kế trước; xác nhận bộ tham số trước khi chạy. Phải triển khai lại kinematics, FOV/altitude Pd/Pf, energy/radio calibration và khởi tạo có ràng buộc; dùng một scenario ID và seed fix. Không chỉ đổi các constant Python vì Warp arrays đang dùng `CELLS=400`, `GRID_W=20`.
- **DS prior:** vacuous mass `m(T)=m(F)=0,m(U)=1` KHÔNG có nghĩa prior Bayesian P(target)=0. Prior Bayes bằng 0 tuyệt đối sẽ không thể cập nhật thành xác suất dương bằng likelihood Bayes thông thường. Nên chọn prior thống kê nhỏ >0, hoặc giữ DS vacuous *không gọi đó là Bayesian 0*.
- **AoI cần phiên bản dữ liệu:** nguồn duy trì `effective_count_pos/neg` với decay tại mỗi measurement và `origin_cell_version` tăng đơn điệu; đóng gói snapshot + generation time + version. Receiver từ chối version cũ, suy giảm snapshot theo tuổi thật và thay thế (không double count). ACK watermarks theo monotonic version, không theo counts đang decay. Khi đổi 8B/entry, cập nhật frame duration/Q buffers/Warp/CPU parity.
- **Safety P0:** mô hình collision terminal là baseline hợp lệ; để *bảo đảm* không va chạm cần lập kế hoạch từ viable set, mô hình quỹ đạo liên tục/phanh, thử head-on/corner/cylinder/peer-inactive và chứng minh đủ điều kiện ban đầu. Nếu không có quỹ đạo an toàn thì không thể vừa chặn va chạm vừa bảo đảm a_max chỉ bằng v trong 1 s.
- **Reward P1:** peer reward chỉ khi ACK xác nhận thông tin tại receiver; dùng delta độ bất định/entropy trên map receiver trước/sau với cap, provenance chống trùng, thay vì `1/(count+...)`. Đánh giá ablation `TP, FP, AUC, coverage, delivery latency, PDR, queue overflow, joule/target`.

## 4. Gate kế tiếp, từng bước và không được nhảy gate

**G0 (đã làm):** extract ZIP, hash baseline, CPU pytest; kiểm tra giả thuyết nguồn; xác định scenario mismatch. **PASS**.

**G1 (đã làm một phần):** counterexample shield, terminal collision, bounded target placement, static queue & byte budget. CPU tests **PASS**, nhưng chưa đánh giá Monte Carlo nhiều seed/episode (unsafe rate, reward exploit). **PARTIAL**.

**G2 (CHƯA đạt):** Warp CPU runtime test trên code *v5.8* (1-hop, 2-hop, no route HOL, E2E duplicate, TTL, collision, per-world reset) và so sánh **từng sự kiện** với CPU reference cùng topology & packet generation. Thuật toán random khác nhau thì cần fixed deterministic backoff/tie-break schedule để parity có nghĩa. Không dùng test AST thay execution. **BLOCKED**.

**G3 (CHƯA đạt):** T4 CUDA GPU device (2 card độc lập). Chạy `scripts/v56_cuda_preflight.py --device cuda:0 --envs 32 --allow-experimental`, so đúng CUDA kernel với NumPy PHY, tiếp E=256,1024,2048, peak VRAM, overflow, compile/steady-state ms/step, NaN/Inf. Sau đó GPU1. *Không suy luận compile nhanh hơn chỉ vì thêm `enable_backward=False` nếu chưa đo*. **BLOCKED**.

**G4 (CHƯA đạt):** nối full movement + sensing + PHY + MAC + belief + reward trên device (không CPU step mỗi world); nghiệm thu throughput/seed reproducibility và triển khai trainer MATD3/MASAC + replay 300k per GPU, checkpoints/eval/recovery. E=2048/GPU không có nghĩa cộng VRAM 2 T4 thành một. **BLOCKED**.

**G5 (CHƯA đạt):** train nhiều seed, ablation, metrics task + network + safety, bảng Q1-paper methodological comparison, confounders, CI. **BLOCKED**.

## 5. Nguồn gốc và giới hạn trích dẫn

- [Zhang et al., IEEE Transactions on Intelligent Vehicles (2024), DOI 10.1109/TIV.2024.3352581](https://ieeexplore.ieee.org/document/10388472): cooperative coverage + static target localization, DS belief maps; không có bằng chứng paper gốc phê duyệt reward/packet code này.
- [Xue et al., *Electronics* 15(9):1875 (2026), DOI 10.3390/electronics15091875](https://www.mdpi.com/2079-9292/15/9/1875): DS và AoI trong cooperative search; dùng làm ý tưởng **không** coi trợ giúp đại số riêng là reproduction end-to-end.
- [Yates et al., Age of Information survey, arXiv:2007.08564](https://arxiv.org/abs/2007.08564): nền tảng phân biệt tuổi cập nhật và tuổi một tập mẫu.
- [Zeng et al., IEEE Transactions on Wireless Communications (2019), DOI 10.1109/TWC.2019.2905569](https://ieeexplore.ieee.org/document/8663615): P(V) for rotary-wing UAV; hạng tử kappa và KE trong ZIP được ghi rõ là extension scenario-specific.
- [UavNetSim — Zihao-Felix-Zhou (GitHub)](https://github.com/Zihao-Felix-Zhou/UavNetSim): SimPy, PHY/MAC/routing/mobility/energy; ACK bypass queue; unroutable DATA đi waiting_list; metrics PDR, E2E delay, NRL, throughput/hops. Port này chỉ là mô phỏng giản lược tương tự, chưa phải chạy original UavNetSim.
- [NVIDIA Warp module options](https://nvidia.github.io/warp/v1.17/user_guide/configuration.html): `enable_backward=False` giảm sinh mã reverse-mode, không tự bảo đảm hiệu năng GPU; cần đo compile, occupancy & coalescing.

**Định nghĩa hoàn thành:** không được thông báo “training production PASS”, “CPU–Warp parity PASS” hay “đã nén SoA” nếu chưa có log thực thi tương ứng. Tài liệu này nêu minh bạch những phần source đã sửa và các giới hạn còn lại.

### Script kiểm tra T4 tuần tự (không dùng khóa/không train)

`python scripts/v58_t4_scale_gate.py --allow-experimental --devices cuda:0 cuda:1 --sizes 32 256 1024 2048` — gọi CUDA preflight từng kích cỡ, dừng ngay khi có lỗi/timeout và ghi JSON. Một khi smoke đều xanh vẫn cần CPU–Warp event parity và trainer GPU tích hợp; nó không tự cấp phép training.
