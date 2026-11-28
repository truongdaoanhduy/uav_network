# V123 — kiểm toán Action Space và kịch bản DS-only trên mã V122 (28/11/2026)

**KẾT LUẬN KỸ THUẬT/NGHIÊN CỨU:** Không thêm action mới vào canonical. `SCENARIO_FROZEN=false`. Full project V122 giữ nguyên 30 tập tin thuật toán MATD3/MASAC và hai YAML đang hoạt động. Bản V123 bổ sung **nhánh thăm dò CPU-only gimbal azimuth** có gate fail-closed cho Warp, **không phải** triển khai action 3D để training.

## 0. Phân biệt rõ 4 phạm vi bằng chứng

1. Mã nguồn V122 do người dùng tải lên: đã mở trực tiếp, SHA256 `2fa8599387f42362ba417320318c41bc903e66fcdd19043d3929ab7522763eee`. Nó không phải V94/V102.
2. Bài báo: dẫn **đúng DOI/URL và nội dung gốc có thể đọc**, không lấy ý tưởng trong paper làm bằng chứng đã kiểm chứng trên code. Không tự gán tất cả công trình là A* hoặc Q1 không có dữ liệu phân ngành/năm.
3. Thử nghiệm V122 **trên GPU Vast RTX3060**: 644 PASS / 6 skipped / 153 subtests; Base và **RF-only** 4096×900 mỗi lượt stationary, không reset, không thấy vi phạm monitor. KHÔNG phải GPU thí nghiệm với gimbal hay full RF v105 peer.
4. Thử nghiệm V123 mới **trên CPU của môi trường phân tích**: 12 targeted PASS, 3 seeds×60×3 camera arms, trace parity V122-V123 2 seeds×20. Nhiễu cảm biến không keyed CRN và pilot chưa training RL; không dùng để công bố tốt hơn deterministic.

## 1. Kịch bản chuẩn thực sự hiện hành và action hiện có

| Hạng mục | Giá trị kiểm tra trực tiếp trong YAML và source |
|---|---|
| Map, môi trường | 1000×1000m, grid 20×20 50m, 5 building (70–80m), 2 blackout |
| UAV | 6, fixed z=50m, v_max=15m/s, a_max=5m/s², safe-distance=50m |
| Mục tiêu | 15 ô mục tiêu tĩnh, horizon=900 |
| Cảm biến chuẩn | nadir circular FoV radius=50m, Pd=0.85, Pf=0.03 (project parameters, NOT calibration) |
| Evidence | Dempster–Shafer `dempster_shafer`, `raw_u16_count`, DS mass `mf=me=0.8`, confirm 0.9, ignorance ceiling0.2, prior vacuous |
| CPU MAC/PHY | `SevenRadioCsmaCaNetwork` 7 node (6UAV + GCS), DIFS50us, SIFS10us, slot20us, CWmin31, ACK; GCS antenna elevation is not GCS flight |
| GPU MAC | UavNetSim-**inspired** Warp surrogate, not an event-by-event clone of upstream SimPy |
| Communication canonical | 0.1W TX fixed, PEER dissemination off, autonomous route table supports HELLO, app sends to GCS; λ_b=0 |
| `act_dim` learned canonical | **2**: velocity XY continuous. `comm_action_mode=compact_3` is backend APP interface, not automatically third learned action: runner must enforce deterministic `comm=1` when sending to GCS. |
| Reward | first-visit coverage, TP at GCS, FP penalties, AoI, energy and safety. Beware first-visit reward ≠ target confirmation and app PDR ≠ target delivery. |

**Có điều chưa chuẩn:** CPU `step` có `act_comm_disc` và `tx_power_w` tùy chọn, nhưng Warp `GPUWorldV79Params.from_scenario` từ chối per-UAV TX; cấu hình canonical tắt nó. Không thể mô tả learned TX routing khi thực tế chúng được hardcode.

## 2. Action không di chuyển: chỉ thêm nếu đạt cơ chế + tác động + cải thiện

| Action | Làm thay đổi phương trình nào | Lý do không thêm vào canonical ngay | Quyết định |
|---|---|---|---|
| TX power (continuous) | SINR, carrier sensing range, TX energy | Warp từ chối power-control; tải kênh base ~1.25% theo V122; tăng Pmax có thể làm tăng interference/hidden-terminal, **không chứng minh luôn suy biến** từ energy fraction nhỏ | **Không thêm**; đòi test PDR, delay, spatial reuse tại tải cao trên kernel parity |
| Routing weights (continuous) | argmax chọn next-hop dựa trên HELLO/SINR/queue/energy | Với 6UAV nhiều trạng thái có 1hop; nhiều weight cho cùng một next-hop, gradient yếu; phải so sánh adaptive deterministic baseline | **Không thêm** |
| Direct next-hop (categorical) | thay relay/GCS | Rời rạc; MATD3/MASAC continuous chuẩn không nhận được gradient qua round(); ứng viên rất ít | **Không thêm**; nếu sau này có mixed action phải đánh giá algorithm riêng |
| Transmit/Hold | queue, AoI, airtime | Channel Base chưa nghẽn, hold tăng trễ; chưa có lý do để RL tối ưu. Không được suy ra strictly-dominated trong mọi kịch bản | **Không thêm** |
| Frequency/channel | interference, frequency occupancy | Không có multiple independent bands có cơ chế interference/spectrum assignment thật trong Warp hiện tại | **Không thêm** |
| Packet priority / relaying role | AoI, target data delay, queue class | Gói DS frame aggregate và không có deadline/priority differentiated trong canonical; thêm action sẽ vô hiệu nếu MAC queue ignores it | **Không thêm** |
| Altitude z | FoV, LoS, Pd, energy | Fixed 50m, cần coupled sensing & 3D obstacle/safety/energy calibration, phá scope XY | **Không thêm** |
| Zoom/FoV (continuous) | GSD ~ `2h*tan(FOV/2)/Npx`, footprint width ~`2h*tan(FOV/2)` | Cơ sở về ground sampling distance; **chưa có** sensor resolution/Npx, Pd(GSD) và Pf(GSD) đo thực trong code | **Ứng viên nghiên cứu**, KHÔNG chính thức |
| Gimbal azimuth (continuous) | ray `d·axis / ||d|| >= cos beta`, occlusion; action slew | Có effect thật nhưng 9 pilot runs còn thua nadir; không có Pd/Pf theo góc nghiêng hoặc CUDA parity | **Code CPU pilot riêng, từ chối canonical** |
| Pd(speed)/dwell | detection/confidence, coverage vs tốc độ | Không cần chiều action, nhưng phải có measurement exposure/blur Pd(v), correlated false alarms | **Ứng viên mô hình Phase3**, không code tùy tiện |

### Cơ sở toán cho thăm dò camera

Tại UAV `(x,y,h)` và azimuth `psi`, trục nhìn nghiêng góc `alpha` khỏi thẳng xuống là `a=(sinα cosψ, sinα sinψ, -cosα)`. Vector từ camera tới ô đất `d=(cx-x,cy-y,-h)`. Ô được nhìn **nếu**

`(d · a) / ||d|| >= cos β`, với `β` nửa góc của nón quang học, cộng điều kiện chưa bị nhà chắn. Bán kính ứng viên an toàn cho raster=`h*tan(α+β)` khi `α+β<90°`; azimuth bước kế tiếp là heading mục tiêu đã giới hạn `|Δψ|≤ωmaxΔt`. Điều này là **geometry ray-plane/pinhole** có thể kiểm thử, KHÔNG chứng minh camera phát hiện với cùng Pd/Pf khi off-nadir. Pilot tùy chọn `α=β=25°`, `ωmax=45°/s` đều là lựa chọn mới **chưa hiệu chuẩn**, không hề lấy ba con số này từ Q1/A*.

Zwick et al. 2023 xây dựng sensor-performance model phụ thuộc GSD/góc nhìn; Rios et al. 2025 mô tả zoom để duy trì GSD thay vì đổi altitude. Các công trình này **ủng hộ việc cần ghép detector/footprint vật lý** chứ không cấp phép đặt `Pd(angle)=0.85` tùy tiện.

### Cùng trajectory 3 seeds×60 steps: Kết quả chạy THẬT V123

| Policy camera | TP tại GCS trung bình | FP-ever trung bình | Coverage delivered % | Mean reward/step | Pin min % |
|---|---:|---:|---:|---:|---:|
| Canonical nadir V122 | **3.667** | 0.333 | **32.00** | **2.8867** | 94.8036 |
| Fixed East +25° tilt | 1.667 | 0.000 | 15.75 | 1.3861 | 94.8036 |
| Sweep East/West mỗi 15 step | 0.667 | 0.000 | 18.75 | 0.96944 | 94.8036 |

**Kết luận đúng phạm vi:** Azimuth hữu hiệu hình học nhưng 2 controller naïve kém nadir. Không suy từ pilot rằng mọi RL policy camera đều kém hoặc rằng một action mới có ích. Bằng chứng không đủ để nâng `act_dim=3`; giữ `act_dim=2`.

## 3. C85–C94 — Xung đột và nguyên nhân chưa đóng (đầy đủ 94 mục JSON/CSV)

- **C85 P1 DS tương quan:** module `ds_correlation_mitigation.py` chỉ được gọi trong hiệu chuẩn offline có monkeypatch và test; `transition.py`/Warp chưa nối. Dù báo cáo V122 offline `ρ=0.7` FP-ever giảm `13.0 -> 1.667` khi cooldown, **canonical vẫn dùng raw count i.i.d.**; triển khai production cần cả CPU và Warp + conservation/dedup đúng nguồn.
- **C86 P1 FREEZE gate:** 644/644 pytest runnable không thay thế xác minh khoa học. Chưa chốt kịch bản chỉ bằng pytest CUDA.
- **C87 P1 SENSOR:** Pd/Pf off-nadir không hiệu chuẩn; action gimbal pilot research-only.
- **C88 P2 Transactionality:** đã vá để không ghi trạng thái camera khi vận tốc/comm đầu vào không hợp lệ, có GREEN test.
- **C89 P3 Legacy Bayes:** còn field `bayes_prior_occupancy` không ảnh hưởng DS + `mission_v91_bayesian.py` cũ trong archive; kiểm tra `belief_inference_mode='bayesian_logodds'` bị từ chối. **Không phải đang train Bayesian.** Xóa file cũ chỉ sau khi đánh giá backward compatibility.
- **C90 P1 Parity GPU–CPU:** cần paired per-step traces moving E4×300 của position, DS mass, reward, queues, MAC/PDR, energy. 4096×900 **stationary** GPU gate chỉ đo tính ổn định.
- **C91 P2 Action degeneracy:** 2 policy gimbal đơn giản giảm TP/coverage; không thêm vào canonical.
- **C92 P1 SimPy vs Warp:** MAC/PHY surrogate nén DS packet, cập nhật link theo snapshot; không thể tuyên bố tương đương UavNetSim-v1 event-by-event.
- **C93 P2 RF label:** 2 lượt Vast là **Base** và **Base+RF noise 4e-11 W**, không được ghi là đã chạy v105 PEER=ON stress.
- **C94 P2 RCT/CRN:** môi trường sensor noise khác số lần rút RNG giữa arms; để đánh giá hiệu ứng nhân quả cần keyed randomness theo `(seed,u,cell,t)` và paired CI, >=5 seed, để tránh chỉ đánh giá may rủi.

Các vấn đề kế thừa C01–C84 vẫn nằm trong registry gốc và registry mới, không tự động gắn nhãn đã đóng; đặc biệt C13/C32 RF-loss 12dB là heuristic, C04/C23/C41/C66 DS temporal correlation, C07/C33/C65 network surrogate.

## 4. Cấu trúc đầu ra, chứng minh thay đổi code và log

- Full project trong `uav_network_V123/`; original V122 ZIP trong `input/V122_ORIGINAL_UNMODIFIED.zip`.
- Code diff `outputs/v123_action_audit/V122_to_V123_source.diff`, SHA file changes, registry 94 JSON/CSV.
- CPU experimental code `sensing/gimbal_cone.py`, flags trong `environment/configuration.py`, actor observation 551dim **chỉ khi bật**, `step(gimbal_azimuth_cmd=...)`, unit tests; Warp fail-closed.
- `V123_GIMBAL_PILOT_3SEEDS_60STEPS.json`, per-step **JSONL 540 records** và log.
- `V122_CANONICAL_CPU_TRACE.json`, `V123_CANONICAL_CPU_TRACE.json` SHA identical across 42 records (2 seeds, reset+20 steps).
- Vast V122 GPU tests: summary JSON và **đường dẫn gốc** trong `V122_VAST_GPU_VERIFIED_SUMMARY.json`. Bản ChatGPT ZIP này **không chứa raw Vast GPU stdout/JSONL** nếu chưa được chuyển dữ liệu từ GPU: không tuyên bố có. Raw logs tại `/workspace/v123_validation/gpu_logs` trên Vast.

## 5. Cổng để khóa Phase1 và không phá baseline

1. Sửa C85 **runtime** hoặc tuyên bố rõ raw sensing is i.i.d. and run robust correlated sensitivity **end-to-end**; nếu tăng sensor correlation ở main, phải re-eval false positives.
2. Thu nguồn hiệu chuẩn hợp lệ cho Pd/Pf, optical GSD/area; làm calibration/sensitivity theo h, góc, speed, occlusion. Không thêm action chưa đo.
3. Xây dựng matched SimPy UavNetSim-v1 vs GPU MAC waveform, PDR/latency/interference under equal input packets; không ép phải byte-for-byte nếu báo scope surrogate, nhưng phải nêu sai khác.
4. Pair CPU/GPU moving traces, sensor map, DS beliefs, APP packet/GCS TP, physical shield & reward. GPU 4096×900 at least Base/RF-stress true; packet-conservation, energy-conservation.
5. Chốt config, seeds, test definitions, reproducible baseline `act_dim=2`; MATD3/MASAC **nguyên bản** chỉ triển khai sau khi Phase1 chốt, chưa tuning/reward hacking.
6. Mọi action mới muốn được giữ cần tăng GCS delivered targets/AUC hoặc một metric mục tiêu định nghĩa trước, giữ discovery/safety, báo paired CI, cost GPU, phản chứng strong deterministic baseline.

`SCENARIO_FROZEN=false`; `PHASE2_BASELINE_STARTED=false`. Không push GitHub, không train model, không báo V123 FULL GPU PASS. Mọi thứ phải tái lập bằng code, log và manifest trong gói.

## Nguồn học thuật có thể kiểm chứng

- Zhang et al., *IEEE Transactions on Intelligent Vehicles* 2024, DS belief for multi-UAV reconnaissance, https://doi.org/10.1109/TIV.2024.3352581 ; nguồn IEEE https://ieeexplore.ieee.org/document/10388472
- Alam & Moh, *IEEE Transactions on Mobile Computing* 2024, joint trajectory/frequency/routing; DOI https://doi.org/10.1109/TMC.2024.3403890
- Zhou et al., *UavNetSim-v1*, arXiv:2507.09852 and upstream https://github.com/Zihao-Felix-Zhou/UavNetSim . **Không đồng nhất** custom kernel với source SimPy.
- Zwick et al. 2023, *Sensors*, sensor-dependent GSD/angle and target detection, https://doi.org/10.3390/s23020664
- Chacon Rios et al. 2025, *Sensors*, zoom-aided UAV coverage fixed GSD, https://doi.org/10.3390/s25175220
- Zeng, Xu & Zhang, *IEEE Transactions on Wireless Communications* 2019 rotary energy, https://doi.org/10.1109/TWC.2019.2902559
- Shafer 1976 *A Mathematical Theory of Evidence* original book, DOI https://doi.org/10.1515/9780691214696

**Phân loại chất lượng:** IEEE TMC/TWC và IEEE TIV có danh tiếng ngành; quartile thay đổi theo SCImago/Web of Science, chuyên ngành, năm; bài Sensors không tự động coi là CORE A*. Không phát minh citation, không tuyên bố đã đọc toàn văn IEEE bị paywall nếu chỉ xem abstract.
