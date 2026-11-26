# V77 — Kiểm toán khoa học và sửa NFZ cản sóng cho Multi-UAV MARL + UavNetSim-inspired network

**Ngày:** 26-11-2026. **Nguồn:** bản V76 đã kiểm toán, xuất phát từ ZIP V75 do người dùng cung cấp. **Nhánh làm việc:** `source_v77` độc lập, không cập nhật `main` và không đẩy GitHub. **Mục tiêu:** NO-FLY ZONE đồng thời là **vật cản RF thực sự** khi gắn vật liệu/toà nhà, nhưng không đánh đồng vùng cấm bay hợp pháp với vật chắn sóng. **Kết quả hiện tại:** CPU, NumPy và Warp CPU PHY qua các gate nêu trong bằng chứng; **chưa** chạy Kaggle account 2 / CUDA T4×2, **chưa** training MARL đầy đủ 4096 env trên GPU.

## I. Định nghĩa phân tầng: phân biệt 4 thực thể

- **NFZ (cấm bay):** tập không gian không được UAV đi vào. Có thể là vùng quy định pháp lý, không phải nguồn suy hao RF tự động.
- **Building RF obstacle:** các khối trụ với tâm XY, bán kính, chiều cao; dùng chung cho kiểm tra va chạm/không bay vào và kiểm tra đường nhìn giao khối **3D** khi tính gain. Đây là *NFZ kết hợp vật cản RF* trong phiên bản này.
- **Ideal radio blackout:** cắt gain nếu TX hoặc RX nằm trong vùng; có thể dùng làm kịch bản giả lập, **không** tương đương với một jammer thực sự.
- **Jammer:** nguồn gây nhiễu đến receiver phải mô hình `I_jam,j = sum_i P_jammer,i * G_ij` trong mẫu số SINR. Việc tia TX→RX đi ngang hình tròn jammer không đủ cơ sở để đặt gain=0. **Chưa triển khai**; được ghi rõ là future gate.

**Chuyển động hiện tại:** 2D mặt phẳng XY ở z=50m cố định. **Mạng:** xét 3D geometry các đường radio; có thể gọi 2.5D nhưng mô tả khoa học khuyến nghị: *fixed-altitude cooperative multi-UAV target search with height-aware obstruction and packet communication*.

## II. Inventory cấu hình chính (mẫu `scenario_v66_datamule_1km.yaml`; các tệp khác là ablation)

| Nhóm | Mã hiện tại | Hệ quả / ranh giới |
|---|---|---|
| Map | 1,000×1,000m, `grid_w=20` ô 50m, 6 UAV, 1 GCS | Không phải kịch bản 3km/35 target 3D ban đầu |
| Bay | `uav_alt_m=50`, `v_max_mps=15`, `a_max_mps2=5`, 900 bước ×1s | Altitude không phải RL action; safety shield 2D XY |
| Target | `num_targets=15`, FOV radius50m, `pd=.85`, `pf=.03` | Bản đồ mục tiêu 2D, cảm biến 3D occlusion tùy chọn |
| DS | `ds_mf=ds_me=.8`, confirm `.9`, fusion `sum`, `ds_aoi_eta=.02`, fixed calibration | Không phải Bayes-dynamic-mass, confirm=.99 theo ý tưởng cũ, không phải literal Zhang MAX/Xue 2026 |
| NFZ | 5 khối trụ building và 2 ideal blackout | Building có footprint, height; *blackout* không phải vật cản truyền xuyên hợp chuẩn |
| PHY | fixed `tx_power_w=.1`, background noise `4.38957709293155e-13` W, CCA −82dBm | Upstream UavNetSim Table I dùng 4e−11W: lệch **19.596dB**, điều kiện khác |
| Network | own seven-radio CPU MAC (CSMA/CA, SINR, ACK, peer forwarding, data mule) | Không chạy nguồn Python/SimPy upstream; không thể gọi 'original UavNetSim' |
| MAC | CPU queue64 / default Warp queue32, Warp opt-in q64; dynamic MAC from V75/V76 | Nếu không bật `--warp-q-cap 64 --require-queue-parity` không được so sánh queue parity |
| Reward | `w_p2p=0`, `lambda_b=0`, continuous AoI penalty mặc định=0 | Chưa có áp lực reward trực tiếp airtime/peer ở baseline; penalty hacking phải kiểm soát |
| Năng lượng | `enable_accel_energy=false`; vẫn giới hạn gia tốc | Năng lượng không chứa thành phần tăng/giảm tốc khi flag off; có nguy cơ so sánh sai |
| GPU | Warp radio SoA, capacity probe `envs=4096`, batch8192 | 4096 radio matrices **không** phải GPU end-to-end independent flight/sensing/belief/reward/reset MARL |

## III. Đánh giá 10 đề xuất trước khi thay đổi

| ID | Ý tưởng được đề xuất | Thẩm định khoa học | Quyết định / thực hiện |
|---|---|---|---|
| NFZ-01 | Building vừa cấm bay vừa che sóng | **Đúng**, khi vật cản có thực và cao đủ che ray 3D | Giữ shield; sửa PHY opt-in |
| NFZ-02 | Thấy nhiều building chỉ dùng `break` là bug | **Không hẳn.** `break` đúng với mô hình phân loại nhị phân, nhưng sai nếu công bố là multi-penetration | Giữ binary baseline + opt-in `multi_obstacle` |
| NFZ-03 | `n_building ×12dB` cùng alpha=3 | **Không đủ căn cứ**, rủi ro đếm hai lần NLoS | Dùng `FSPL d^-2 × 10^(-N L_b/10)` như **phenomenological ablation**, loss giả định, không gọi chuẩn ITU vật liệu |
| NFZ-04 | Giao vật cản → gain0 | **Có thể**, nhưng chỉ *ideal RF-opaque* scenario | Opt-in `rf_opaque` và độc lập baseline |
| JAM-01 | Tia đi xuyên blackout nên gain0 | **Sai với jammer**, nguồn nhiễu không hoạt động như mặt chắn | Không đổi endpoint-blackout; đề xuất thêm SINR jammer receiver noise sau |
| MAC-01 | Hidden terminal do building khiến hai UAV không nghe CCA nhưng cùng thấy GCS | **Đúng có điều kiện hình học** | Kiểm tra CCA và ép hai frame overlap để chứng minh `sinr_drops` tăng; không tuyên bố đã đo thống kê ngẫu nhiên |
| AIO-01 | FIFO → overwrite stale packet `(origin,dst)` | **Có nguy cơ sai** do DS cumulative evidence sequence, timestamps, ACK, peer duplicates | Không bật mặc định, yêu cầu chứng minh replacement preserving BBA/seq trước Warp parity |
| PHY-01 | Noise difference 19.596 dB là bug | **Không nhất thiết**; mức noise khác mô hình/khác BW/NF | Giữ nguyên baseline, tạo high-noise ablation 4e−11W để đối chiếu, không gọi phổ quát |
| REW-01 | Bật P2P/airtime/AoI penalty để học giao tiếp | **Có lý**, nhưng weight `0.1/0.5/0.01` là **giả thuyết thực nghiệm** | Chỉ thêm YAML ablation, không thay đổi baseline reward |
| GPU-01 | 4096 radio env = 4096 full MARL env | **Sai** | Giữ status explicit `NOT_FULL_RL`, yêu cầu port full transition trước công bố throughput RL |

## IV. Chi tiết triển khai V77

**Mã đã thay** (file provenance trong `patches/v76_to_v77.patch`):

1. `research/zhang_xue/src/zhang_xue/communication/common.py`: `link_path_gain` hỗ trợ `binary_nlos`, `multi_obstacle`, `rf_opaque`; xác minh số building cắt tia trong khối hình trụ 3D. `multi_obstacle`: 

   \[G_{ij} = (d_{ij}+1)^{-2}10^{-N_{ij} L_b/10}\]

   với `N_ij` = số cylinder bị cắt, `L_b` dB mỗi cylinder. Đây **không phải** dự đoán vật liệu có hiệu lực cho mọi tần số. `binary_nlos` giữ công thức cũ \((d+1)^{-3}10^{-12/10}\) khi bị cắt, còn không cắt thì \((d+1)^{-2}\).
2. `communication/lifecycle.py`: CPU `SevenRadioCsmaCaNetwork` nhận/tính đúng các tùy chọn; `environment/core.py` luân chuyển từ YAML; `environment/configuration.py` validates enum và loss hữu hạn ≥0.
3. `gpu/kernels/phy_v67.py`: NumPy reference + kernel Warp JIT nhận mode int và loss; bảo toàn độ cao và blackout; `gpu/runtime_v67.py` luân chuyển flags vào chính kernel.
4. `uav_network/experiments/radio_config.py`: capacity manifest báo rõ `rf_phy_applied_by_capacity_probe=false`, vì **easy packet fixture không chứa buildings**; tránh báo cáo sai.
5. `scripts/gpu/warp_phy_cuda_gate.py`: thêm số liệu multi and hard-block; `scripts/maintenance/nfz_radio_gate.py`: vật lý reference, CCA hidden terminal và gây xung đột gói có kiểm soát, optional Warp device.
6. `tests/test_v77_nfz_radio_models.py`: kiểm thử hình học 0/1/3 building, rooftop, blackout, scenario wiring, CPU/NumPy and optional real Warp CPU.
7. Thêm YAML `scenario_v77_nfz_multi_*`, `scenario_v77_nfz_opaque_1km.yaml`, và high-noise/communication-reward sweeps chỉ để **ablation**.

**Chú ý tính sát thực:** tiếp tuyến chỉ chạm mặt building trong bộ dò `segment_hits_cylinder_3d_fp32` vẫn tính là *blocking*; để nghiên cứu calibrated `path penetration length`, phải thêm độ sâu xuyên và epsilon tolerance. Mô hình `multi_obstacle` trên đây là model-sweep; chưa đo vật liệu/khoảng tần để suy ra `L_b`.

## V. Những conflict và rủi ro còn mở cần xử lý theo thứ tự

| Tầm quan trọng | Conflict/mã | Vì sao ảnh hưởng kết quả? | Đường xử lý / gate |
|---|---|---|---|
| CRITICAL | `capacity_probe.py` dùng **bốn** CPU transitions lặp để cấp replay GPU | Báo `4096 env RL trained` là sai và có rủi ro leakage/correlation | Port Warp flight+collision→sensor+DS belief→network→reward→reset; GPU trajectories khác nhau; G6 |
| HIGH | `rf_obstacle_model` đang chạy **CPU env** và **Warp PHY gate**, nhưng capacity packet fixture không đưa building vào | Test 4096 radio chưa kích hoạt NFZ trong toàn pipeline | G4 tạo workload route dynamic obstacle GPU và đo `SINR drops` trước G6 |
| HIGH | CPU queue=64, Warp default32; slot/DIFS/ACK configs thay đổi | PDR/AoI/airtime khác nhau ngoài policy | V76 opt-in q64/dynamic MAC, đồng bộ và histogram parity |
| HIGH | CCA hidden terminal và SINR collision có thể chưa kiểm tra dưới random backoff thực | Forced two TX overlap chỉ chứng minh logic SINR chứ không đo rate | G4 randomized seeds, traffic load, K slots, compare upstream heuristic |
| HIGH | Censor AoI đến terminal / unequal horizon | Short episode có AoI thấp giả tạo | Fix horizon-censored extrapolation V76, luôn log termination |
| HIGH | Noise +19.596 dB giữa mẫu tham chiếu UavNetSim và YAML | Cùng policy cho PDR khác dù code giống | Matching scenario paired tests, lý giải BW/NF |
| HIGH | DS `sum` vs Zhang `zhang_max`, delayed packet AoI power variant vs direct BPA; fixed mass .8 confirm .9 | Không thể gọi *faithful paper reproduction* | Declare novel variants, compare ablations, preregister fusion semantics |
| MEDIUM | No 3D altitude control, 3D sense/PHY geometry có | Nếu đặt tiêu đề 3D navigation reviewer phản biện | Dùng thuật ngữ fixed-altitude 2.5D |
| MEDIUM | `enable_accel_energy=false` dù có `amax` | Energy metric không phản ánh tăng tốc | Nêu mô hình đơn giản hóa; optional energy ablation, không phá baseline |
| MEDIUM | No queue overwrite parity, stale DS seq management | Overwrite làm mất dữ liệu và ACK | G5 coalescing chỉ với snapshot replacing all old info, test replay/duplicates/expiry |
| MEDIUM | `w_p2p=lambda_b=0` baseline | MARL không tối ưu những chỉ số đó trực tiếp | reward sweep, kiểm soát spam/no send, comparative energy fairness |
| MEDIUM | No GCS information exclusivity or oracle test in every path | Critic privileged may leak into actor via observation | Dec-POMDP actor/critic dataflow audit + randomized no-oracle tests |

## VI. Cổng test và bằng chứng (thực tế, không tự suy đoán)

- **G0** Frozen V76/V75 snapshot, source SHA, isolated folder, Git patch dry-run. Không sửa user main.
- **G1** CPU `link_path_gain` 0/1/3 obstacles, FOV above roof, blackout; config enum validation (PASS).
- **G2** NumPy reference numerical parity (PASS).
- **G3** Warp 1.18 **CPU thực thi** 3 chế độ, matrix vs NumPy ref, 1 env × 3 radios, `max_abs_diff≈2.22e-16`, compile (PASS). **Chưa** có CUDA test trên thiết bị thật.
- **G4** Deterministic controlled overlap hidden terminal / SINR drop (PASS logic). Random traffic, large-scale delivery/TTL, multi-hop route under NFZ (NOT DONE).
- **G5** Queue AoI-aware vs FIFO and ACK semantic proof (NOT DONE, intentionally no silent mutation).
- **G6** Nvidia Warp full env 4096 and online MARL training (NOT DONE, radio capacity only).
- **G7** Paired fixed-seed multi-policy/multi-noise ablation (NOT DONE, templates supplied).
- **G8** Multi-seed statistical confidence, calibration against measured channel and original UavNetSim (NOT DONE).

**V77 không khẳng định tất cả bug đã hết hoặc Q1/A* đã đạt.** Gate labels là căn cứ từ log, không phải hứa hẹn có kết quả future. Kaggle notebook trong ZIP chứa source code/sha, chạy `Run All` khi có GPU T4x2, không đọc `key.txt`. User-host sandbox không có CUDA driver, không thể thực thi CUDA/4096 training trên đó.

## VII. Các đối chứng bắt buộc cho bài báo Mạng & Truyền thông

- So sánh PDR (per hop **và** E2E packet), delivery 15 target tới GCS, delay percentile, retransmission, `sinr_drops`, queue drops, TTL expiration, airtime, AoI fixed-horizon, map discovery/coverage AUC, false confirm, battery per successful delivery. Định nghĩa metric phải nhất quán **mẫu số** (sent vs generated vs delivered).
- Baselines radio `NO_PEER`, `ALWAYS_GCS`, `ALWAYS_PEER`, `ROUND_ROBIN`, `LEARNED_COMPACT3`, no NFZ, single building, multi building, RF-opaque, high-noise, blackout. Paired same terrain/target/seed; mỗi seed chỉ thay đúng một trục ablation.
- Đánh giá **mean±CI across independent seeds** chứ không chọn run thắng duy nhất. Kiểm soát channel realism: obstacle count vs path-length penetration; RSSI/SINR calibration; jammer bandwidth/interference không thể suy ra từ vùng tô màu.
- Reward shaping Ng et al 1999 yêu cầu potential difference dạng đặc biệt; watermark monotone của dự án không tự động bảo đảm policy invariance.

## VIII. Nguồn khoa học và kỹ thuật, dùng đúng phạm vi

1. Zhou et al., *UavNetSim-v1: A Python-based Simulation Platform for UAV Communication Networks*, arXiv 2507.09852 (2025): https://arxiv.org/abs/2507.09852 — nguyên bản Python/SimPy, CSMA/CA, DSDV/OPAR/greedy, Table I sample `4e−11 W` noise. **Không** phải chính mã Warp của ta.
2. UavNetSim source: https://github.com/Zihao-Felix-Zhou/UavNetSim — upstream event-driven Python network software.
3. ITU-R P.2040, *Effects of building materials and structures on radiowave propagation*: https://www.itu.int/rec/R-REC-P.2040 — tham khảo vật liệu; **V77 hiện không implement đầy đủ P.2040**.
4. ETSI/3GPP TR 38.901 Rel19 V19.4.0 2026: https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.04.00_60/tr_138901v190400p.pdf — stochastic channel LOS/NLOS; dùng làm **kiểm tra phạm vi**, không khẳng định code chuẩn 3GPP.
5. Zhang et al., 2024 IEEE T-IV, target search belief probability maps: https://doi.org/10.1109/TIV.2024.3352581.
6. Xue et al., 2026 *Electronics* 15(9):1875, AoI-aware cooperative multi-UAV: https://doi.org/10.3390/electronics15091875.
7. Yates et al., *Age of Information: An Introduction and Survey*: https://arxiv.org/abs/2007.08564 — cách phân biệt thông tin mới/cũ, lưu ý censored samples.
8. Ng et al., *Policy Invariance under Reward Transformations*: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf.
9. NVIDIA Warp 1.18 documentation: https://nvidia.github.io/warp/ — CPU/CUDA JIT and supported devices.

**Q1/A* không phải nhãn mặc định cho từng paper.** Cần đối chiếu năm/phân ngành SCImago/CORE theo từng venue; paper arXiv chưa peer-reviewed không được ghi là bài Q1. Tham khảo full audit V76 trong ZIP `report/V76_SCIENTIFIC_CODE_AUDIT_AND_ACTION_PLAN_VI.md`.
