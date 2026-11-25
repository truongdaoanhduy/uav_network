# KIỂM ĐỊNH ĐỘC LẬP V5.4 — 6 UAV + GCS, CSMA/CA, DS/AoI, battery, Warp

Ngày: 2026-11-25. Đối tượng: ZIP user cung cấp `uav_v54_7radio_gcs_csmaca_and_warp_code_20261125.zip` (4 Python src, 1 file 9 test); repo nhánh mục tiêu `experiment/zhang-xue-v53-audited-battery-contract-20261125`. Không kết luận 9/9 PASS = training-ready.

## GATE STATUS (không nhảy gate)

| Gate | Đã làm | Bằng chứng | Kết luận |
|---|---|---|---|
| 1 — Kiểm tra v5.4 nguyên bản | Chạy `python3 tests/test_v54_7radio_csmaca_and_warp.py` trong bản sao nguyên trạng | 9/9 PASS / 0.679 s | PASS cho các trường hợp có test; không phải chứng minh hết MAC |
| 2 — Regression/sửa CPU | Chỉnh sửa các lỗi dưới đây, thêm test 10–20 | `reports/v54_20tests_stdout.txt`: 20/20 PASS | PASS có điều kiện, CPU reference only |
| 2b — Long roll-out | 983 bước seed44 với policy vận tốc cố định (phiên bản trước sửa khoảng cách ban đầu), kết thúc hết pin và chỉ 2/15 TP GCS | Console run, không dùng làm benchmark chính thức vì code sau đó thay đổi scene khởi tạo | Chỉ xác nhận vòng lặp chạy dài; CẦN chạy lại sau patch motion |
| 3 — CPU–Warp parity | `scripts_gate3_preflight.py` trả exit code 2 có chủ đích | `reports/v54_gate3_preflight.json` | **BLOCKED**: chưa có full device MAC/PHY/env |
| 4 — T4 peak VRAM/profile | Không chạy | Chưa có kernel/agent training end-to-end | **NOT RUN** |
| 5 — Train 4096 envs | Không chạy, không dùng Kaggle key.txt | Không đủ parity/VRAM | **NOT RUN** |

Repo sandbox đã xác nhận nhánh mục tiêu tồn tại và worktree ở `uav_network_zhang_xue_v34_research_20261125` là clean, commit `9526e65`. Sandbox plugin shell hiện không có `numpy`, `pytest`, `warp` (Python được gọi thuộc CLI Anything hub) nên **chưa thể test 4 file trên interpreter của máy người dùng**, và ZIP user đính kèm không tự mount vào môi trường sandbox plugin. **Chưa commit/push 4 file lên GitHub**. Bộ cài staging được đóng gói trong ZIP để apply mà không ghi đè code cũ; sau đó phải test trên environment thật trước khi commit.

## 1. Kịch bản đúng theo file v5.4

- 6 actor UAV ID0..5 + GCS radio ID6; bản đồ 1000x1000 m, lưới 20x20 (ô 50m), 15 mục tiêu cố định ngẫu nhiên; 5 nhà hình trụ bán kính45m, cao70–80m, 2 vùng blackout Manhattan bán kính3 ô. Không phải 3000m/35 target của một nhánh nghiên cứu khác.
- Bay **2D có cao độ cố định 50m**; điều khiển vận tốc xy liên tục, v_max25m/s, a_max5m/s², dt1s, horizon1500s; cảm biến FOV radius50m (tương ứng footprint hình tròn của HFOV90° tại z50 nếu nhìn thẳng xuống), Pd0.85, Pf0.03. Không có action cao độ 3D.
- DS m_f=m_e=0.80, threshold T=0.90 (v5.3 profile threshold0.95), initial vacuous mass (0,0,1) qua counters; GCS cộng theo origin rồi tính mass, không phải bản sao Eq.(16) MAX mỗi UAV của Zhang.
- Mạng tự viết lấy cảm hứng UavNetSim classic: 7 radio, PHY 2.4GHz và 2 Mbps, TX0.5W, CCA-82dBm, 6dB SINR, NLoS exponent3 và excess12dB, half duplex, DIFS50μs SIFS10μs slot20μs CWmin31, FIFO64, TTL10s, max_retx5; route-vector; MAC ACK và E2E ACK.
- Pin 180000J (50Wh), steady level Zeng2019 + avionics15W + RF airtime (radio GCS không trừ vào UAV).
- Reward thực code: `w_loc_team=0.5`, `r_tp=25`, `w_aoi=6`, `r_fp=-12.5`, FP hold=-0.10/step, airtime=-5*TX seconds và trong bản vá thêm blocked-motion=-0.2/UAV veto/step. `w_p2p=0.25` khai báo nhưng chưa được dùng — **không được ghi là thưởng P2P hoàn chỉnh**.

## 2. Các sai lệch đã chứng minh và đã vá

| Mức | Vị trí | Bằng chứng/lý do | Thay đổi trong ZIP |
|---|---|---|---|
| P0 | `uav_v54_7node_env.py`, reset | Clock reset 0 sau warmup 0.6s nhưng timer/routing đã chạy ở 0.6s => rollback thời gian, lệch tuổi packet | Giữ clock đơn điệu, `epoch_us` dùng tách thời gian nhiệm vụ, fail nếu warmup không hoàn tất |
| P0 | network app dedup | `pkt_id` đổi qua hop; destination dedup theo pkt ID không đảm bảo dedup tầng ứng dụng | Dedup final bằng `(dst, app_src, app_seq)`, E2E ACK vẫn trả lại khi trùng |
| P0 | `e2e_pending` | Dict không có cleanup khi TTL hết; CPU dict có thể tăng không giới hạn, Warp chỉ `P_E2E=16` | Thêm TTL metadata và prune |
| P0 | Warp manifest | 80 arrays không chứa `ack_wm_ts` (+67200B/env), `gcs_last_rx_us` (+3200B/env), `uav_velocities_xy` (+48B/env), chưa tính nhiều field khác | Bổ sung audit fail-closed, chưa tăng array count giả tạo; tối thiểu đã biết **560449B/env** hoặc **1.0690GiB/2048env**, chưa phải peak |
| P1 | `run_until_us` | Return False khi event overflow có thể bị env bỏ qua và tiếp tục cập nhật RL state trên network stale | Fail-fast RuntimeError |
| P1 | AoI GCS | Timestamp rx ghi đè cả khi gói già, khiến `rx` có thể không ứng với generation mới nhất | Chỉ cập nhật last_rx khi generation timestamp **mới hơn** |
| P1 | decentralized actor | Obs index12 đọc `stats[e2e_ack_rx]` tổng toàn mạng | Thay bằng tỷ lệ ACK watermarks trong ledger của riêng UAV |
| P1 | action discrete | `action_to_dest_radio_id` không validate action ngoài [0,6] | Reject invalid action/shape/nonfinite command |
| P1 | HELLO vs queue head | DATA không có route có thể che HELLO control-frame trong cùng queue | HELLO control slot ưu tiên và coalesce trên mỗi radio |
| P1 | initial positions/safety | Khoảng cách 5 cặp UAV kề nhau 46.1m, nhỏ hơn safe distance 50m trong profile v5.3 | Initial spacing 65m; pairwise continuous-time closest-approach veto; thêm telemetry blocked_motion |
| P1 | collision geometry | Chỉ xét endpoint, không xét đường đi và cặp UAV | Xét capsule segment to building + khoảng cách ngắn nhất cặp trong cả bước (cấm CBF; veto đơn giản) |
| P2 | belief exploration | Dùng `max(0,prev_uncertainty - now)` làm thưởng một chiều, có thể được lợi từ dao động uncertainty | Signed delta theo Eq.27 cognitive reward Zhang; chỉ là một lựa chọn reward, cần ablation |
| P2 | AoI name | gọi `mapwide_unbiased` khi cell chưa có update được gán AoI=t theo quy ước | Thêm tên đúng hơn `mapwide_censored`, giữ legacy field cho compatibility; bắt buộc báo coverage |

**Giới hạn của patch chuyển động:** hard veto đưa vận tốc thực về 0 ngay tại thời điểm bị chặn, có thể vi phạm acceleration clamp. Đây là xấp xỉ mô phỏng, không phải mô hình phanh khả thi cho UAV thật. Phải xử lý ở Gate 3 trước claim high-fidelity flight.

## 3. Các xung đột logic chưa được giải quyết — ưu tiên sửa tiếp

| ID/ưu tiên | Xung đột | Vì sao ảnh hưởng RL/mạng | Hướng xử lý và gate nghiệm thu |
|---|---|---|---|
| A/P0 | **Không có Warp step kernel 7-radio.** Chỉ `counts_to_canonical_ds_wp` và partial reset kernel; không có GPU packet scheduler, route, RX, GCS reward | 4096 env vẫn gọi Python/NumPy loops, không song song Warp thật | Thiết kế state machine bounded events kernel/phase, CPU–Warp trace parity, chỉ đo GPU sau đó |
| B/P0 | **Reset kernel Warp chỉ reset một phần** (time, MAC state, q counts, battery), không reset ledger, route, dedup, AoI, target, ACK, RNG | Cross-episode contamination & invalid replay | Full reset lệnh có cùng episode_id/seed cho toàn bộ trường, test E1 done và E2 không done bất biến |
| C/P0 | **Obs actor chỉ 14 telemetry, không có DS map/pattern**, critic state chỉ flatten 6×14. Actor không có thông tin cell nào bất định/đã khảo sát | Không thể tối ưu tìm mục tiêu dựa vào DS map như Zhang/Xue; sai mục tiêu task | Xây local belief patch hoặc sparse CNN / sequence features, neighbor route/AoI; critic full state CTDE chỉ khi train, kiểm tra no-oracle deployment |
| D/P0 | **Kích thước giao diện 6→7 action vs v5.3 6-action; obs14 vs v5.3 121**, reset/signature thay đổi | Actor logits/critic/replay/train pilot không tương thích; chép file không nối training | Versioned contract adapter (OBS/DISC/A_CONT/state/mask), test forward+backward+save+load ngay cả khi chưa học |
| E/P0 | **DS theo Zhang Eq16 MAX per-origin thay vì SUM per-origin**; nhánh v5.4 sử dụng SUM sau provenance | Không byte-faithful reproduction; duplicate evidence/correlation có thể tăng tự tin giả | Document new estimator, paired ablation MAX vs provenance SUM với scene/seeds/cell histories cố định; không gán kết quả nguồn vào extension |
| F/P1 | **AoI chưa được nối vào DS evidence discount**: Xue Eq(9) ψ=exp(-η age); v5.4 chỉ tính AoI reward/telemetry | Không thể gọi là implementation AoI-weighted DS của Xue | Làm opt-in `age_discount` theo evidence event với source_ts bất biến, validate stale/reordering; báo ablation ψ=1 vs ψ(age) |
| G/P1 | **mf=me=0.8 không phụ thuộc Pd/Pf**, dù sensor `.85/.03` | Là hai kiểu calibration khác nhau; threshold0.90 khắt khe ít hơn Zhang .95 | Giữ fixed-mass làm baseline; riêng đề xuất calibrated BPA theo likelihood ratio, benchmark reliability/ECE vs thresholds, không tự gọi 0.8 = Pd |
| H/P1 | **1500s horizon vs pin180000J** | P_hover183.49W => ~981s, P(15)=153.55W => ~1172s, P(25)=263.96W => ~682s; bay liên tục không thể đạt1500s nếu không nạp lại | Nếu battery-limited regime thì giữ terminal sớm và tính success-on-depletion; nếu paper-horizon thì cấp pin/charging nhất quán, không dùng `truncation` che battery death |
| I/P1 | **Network PHY–motion operator splitting**: toàn bộ truyền mạng 1s bằng vị trí cũ, rồi tất cả UAV dịch chuyển | Link gần ngưỡng 500m có thể cắt/kết nối trong bước, ACK timing sai | Substep hoặc cập nhật channel tại packet event theo interpolated trajectory, parity CPU model |
| J/P1 | **Network realism**: UavNetSim classic thực là SimPy, v5.4 là simulator viết lại, không copy source. PHY default `general_path_loss` khác 3D building extension | Không thể tuyên bố kết quả thời gian/throughput tương đương UavNetSim gốc | Golden traces/timer và matched comparisons classic UavNetSim/native reference rồi ns-3 nếu cần chuẩn IEEE 802.11 |
| K/P1 | **E2E pending CPU động & Warp fixed P=16**; HELLO priority/dedup cũng thêm CPU field chưa có Warp | Không có fixed-size bounds/overflow policy tương thích | SoA sizing từ state *cần thiết* và queue drop/saturation counters, bounded retry/ACK và test worst-case overload |
| L/P1 | **Reward thiếu P2P** (`w_p2p` unused), reward TP hiện tại dựa trên GCS verified, local coverage reward không có gói | Nhánh P2P có thể trở nên action vô ích; mục tiêu mạng reward throughput/latency không đủ hấp dẫn | Chọn một mục tiêu: target verification GCS + AoI delivery + bitrate+energy; thêm p2p info gain chỉ nếu observable và pairwise ack; monitor reward component scales, tránh double counting |
| M/P1 | **Bật safety veto sát biên** có thể dẫn tới bão `blocked_motion` và tanh saturation (đã quan sát MATD3 run cũ) | Actor nhận rất nhiều action khác nhau nhưng cùng state next, gradient giảm về0 | Penalize invalid fraction; safe-action projection sliding hoặc differentiable bounded control với physics-consistent braking; đo sat rate / actor grad / invalid rate; không tái dùng CBF nếu user đã bỏ |
| N/P1 | **Sensing chỉ kiểm tra FOV radius, không có ray occlusion vật cản** | Có thể nhìn mục tiêu qua nhà với UAV z50 < roof 70–80 | Ray-cast UAV→ground cell hoặc declare sensor model ignores occlusion, ablation occluded vs unoccluded; mask true target không leak |
| O/P1 | **TX energy** chỉ 3W khi airtime TX; RF electronic/receive/idle khác và TX parameter0.5W không map rõ PA efficiency | Chính sách có thể spam control frame với chi phí phi thực tế | Power budget define P_rf(P_tx), RX/idle PA/baseband, measure per radio TX/RX/idle incl ACK HELLO, ablate energy-vs-delivery |
| P/P2 | **GCS command** hiện network E2E ACK/control phát từ GCS, nó là radio non-learning node; GCS battery unlimited | Hợp lý nếu GCS cấp nguồn liên tục nhưng phải nói rõ, không phải 7 UAV | Count roles/queues/radios separately (actor6, radios7), E2E ACK airtime charge to GCS power ledger |

## 4. Đối chiếu nguồn (không dùng Q1/A* như nhãn tự động)

1. Zhang et al., *IEEE Transactions on Intelligent Vehicles* 2024, DOI https://doi.org/10.1109/TIV.2024.3352581 — pp1,3-9: fixed-altitude 2D, distributed DS maps, initial (0,0,1), detection counts, Eq(16) MAX and Eq(17) DS, continuous action, reward cognitive Eq(27), discovery Eq(28), collision/edge Eq(29-33), CTDE. **Bản v5.4 là extension GCS+packet, không phải một reproduction tương đương từng thành phần.**
2. Xue et al., *Electronics* 2026, DOI https://doi.org/10.3390/electronics15091875 — pp6–12 Eq(1-15): 2D blockage, belief π1/π0/πφ, buffer local, AoI exponential discount ψ(t,t_s) Eq(9), global decay Eq(10), verification Eq(11) với 2 ngưỡng và blocked-zone precondition, discrete 5 actions. **V5.4 KHÔNG triển khai Eq(9) AoI-weighted DS.**
3. Zeng et al., *IEEE Transactions on Wireless Communications* 2019, DOI https://doi.org/10.1109/TWC.2019.2902559 — rotary-wing steady-level P(v). Không phải dữ liệu chứng minh UAV thực có 50Wh usable capacity hay a_max5.
4. Yates et al., *IEEE JSAC* 2021, DOI https://doi.org/10.1109/JSAC.2021.3065072 — AoI latest receiver-side generated information. Chỉ số never-delivered-cell cần công bố quy ước initial age; sample delivered-only có coverage bias.
5. UavNetSim source: https://github.com/Zihao-Felix-Zhou/UavNetSim . Đã xem bản clone `UavNetSim_v1_classic_PIN_64bc3a24/mac/csma_ca.py` và `phy/large_scale_fading.py`: đây không phải cùng file `phy/a2a.py` như mô tả v5.4, và cách tính urban-building NLoS v5.4 là extension riêng.
6. Bianchi, *IEEE JSAC* 2000, 802.11 DCF, https://personal.math.ubc.ca/~israel/m303/Bianchi_JSAC.pdf — freeze/restart after idle DIFS, exponential backoff, SIFS ACK. Đúng mô hình tham chiếu cho FSM, nhưng chỉ custom event simulation không làm code chuẩn IEEE-certified.
7. NVIDIA Warp PyTorch interop: https://nvidia.github.io/warp/v1.19/user_guide/interoperability/pytorch.html — zero-copy + matching stream; `wp.from_torch` không tự đồng bộ khi 2 stream chạy bất đồng bộ. Warp graph runtime: https://nvidia.github.io/warp/v1.18/user_guide/runtime.html.
8. Tesla T4 16GiB theo NVIDIA: https://www.nvidia.com/en-us/data-center/tesla-t4/ . Hai T4 có hai vùng VRAM tách biệt, không cộng VRAM thành một GPU32GiB.

## 5. VRAM và training Warp E4096/U4/B8192

- **80 mảng khai báo** = 490001B/env; 2048 env/GPU = 0.9346GiB/GPU. Đây là lower bound, **không phải peak memory measured**.
- **Chỉ tính 3 field thiếu đã thấy**: ack_wm_ts67200 + gcs_last_rx_us3200 + velocities_xy48 = 70448B/env; lower bound được nâng lên **560449B/env = 1.0690GiB/2048env**, vẫn bỏ qua nhiều field động, SoA alignment & temporary, replay, model, optimizer, RNG, CUDA graphs.
- Replay từ profile v5.3 theo doc repo **6080B/joint transition**, capacity1M ~5.66GiB *nếu đặt trên một GPU*, 500k ~2.83GiB. Không lấy tổng VRAM hai GPU để cấp replay trên một GPU.
- Hiện chưa hề có Warp network episode kernel. Cần profile `torch.cuda.max_memory_allocated`, Warp memory pool, `nvidia-smi`, throughput environment steps/s, kernel trace capture, dropped-event count. T4 không có trong ChatGPT container; user Kaggle account2 **chưa được dùng** vì parity gate còn fail.
- Trước training: (i) golden CPU 1/2-hop/ACK/dedup/blackout/TTL + sensor+motion + battery, (ii) GPU E1/E2/E64 parity trace with deterministic event ordering, (iii) reset while another env is alive, (iv) E2048/GPU profile, (v) adapter CTDE hybrid MATD3/MASAC, replay/gradient/termination/truncation, (vi) E4096 controlled stress, rồi train.

## 6. Dữ liệu 80-step sau bản vá

Source: `reports/v54_gate2_remeasured_80step.json` (seed20261125, fixed velocity policy, two comm policies, fresh run). Vì đã sửa initial spacing, collision handling, reward và HELLO, **không được so sánh trực tiếp numeric score mới với original v5.4 report**.

- never_send: reward **-77.63012**, 0 DATA GCS, 0 TP confirmed, AoI map-wide-censored40.5s.
- always_GCS: reward **-47.8078**, 474 DATA GCS, 474 E2E ACK, 1/15 target GCS-confirmed, 6.5% delivered-cell coverage, first-confirm latency0.0024s, delivered-cell AoI18.921s, mapwide-censored39.063s.
- Chênh lệch reward +29.82232, **không phải chứng minh MARL học tốt**: chính sách vận tốc cố định bị boundary collision veto nên `r_blocked_motion=-1.2` ở bước cuối; giá trị absolute âm phản ánh reward scale sau sửa. Cần ablation và policy heuristic kiểm chứng thay vì chọn một seed.

## 7. Tệp/gate cần xây tiếp

1. `contracts/v55_scene_sensor_contract.yaml`: định nghĩa rõ geometry, target visibility/occlusion, seed, radio IDs, battery/terminal, metric definitions; baseline vs extensions.
2. `uav_network/environment/gpu/zhang_xue_v55_7node_warp.py`: full array allocator + reset + 7-radio event scheduler, sender/receiver/relay/ACK/PHY kernels, CUDA Graph bounded loop; **không đủ chỉ viết 80-array manifest**.
3. `tests/test_v55_cpu_warp_event_parity.py`: trace per event, `gen_ts`, queues, SINR, MAC/E2E ACK, all 7 radios, battery, reward, done.
4. `uav_network/experiments/train_zhang_xue_v55.py`: adapter actor local belief observation, hybrid actions7, critic global CTDE, replay terminal/truncation, MATD3 saturation diagnostics, W&B/HF checkpoint/manifest.
5. `scripts/profile_v55_t4.py`: profile true CUDA peak/throughput memory for E64→E256→E1024→E2048/GPU, compare with current v5.3 baseline. Không sử dụng account key.txt trước khi gate parity đạt.

**Kết luận:** CPU reference v5.4 đã có một số sửa kỹ thuật có test, nhưng chưa đạt full-branch integration, chưa có GPU parity, chưa có training-worthy observation & fixed-size event contract. Nếu làm theo "bước nào chắc bước đó", bước tiếp theo là **Gate 3: biến 80 mảng thành GPU environment thực và chứng minh parity theo trace**, không phải launch E4096.
