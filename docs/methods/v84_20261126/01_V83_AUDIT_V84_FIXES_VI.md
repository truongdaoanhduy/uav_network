# V83 được thẩm định độc lập — mã nguồn và bản sửa V84

## Kết luận

V83 phát hiện đúng lỗi góc biên mà V82 bỏ sót. Kiểm tra phanh X/Y độc lập chỉ là điều kiện cần, không bảo đảm đủ gia tốc L2 chung. Phanh đồng tuyến và geometry fallback của V83 được giữ sau khi tái hiện2/300 lỗi ở V82→0/300 ở V83 và kiểm tra CPU/CUDA.

Tuy nhiên, “replay raw là lỗi P0” không phải kết luận tổng quát. Paper Krasowski mục2.5 có cả naive, adaption penalty, safe action và both; dual là hướng ablation có căn cứ, không phải bản sửa bắt buộc bảo đảm hội tụ. V84 sửa 7 lỗi triển khai cụ thể có RED→GREEN, giữ raw mặc định và không đổi phương trình learner.

Không đổi map1000, 6UAV+GCS,15targets,grid20×20, altitude50m,dt1,vmax15,amax5,safe_d50. Không đưa CBF trở lại, không cho Actor xem target truth, không tắt overflow/fail-closed.

Đọc 02_RUNS.csv, 03_V83_VERDICT_MATRIX.csv và 04_V80_24_ISSUES_UPDATED.csv. Thư mục history giữ nguyên số liệu cũ, không đổi nhãn run.

## 1. Các thành phần và xung đột thực tế

| Phân hệ | Hiện trạng | Kết luận |
|---|---|---|
| Map | 1000m,400cells,5 cylinder buildings,2 blackout;392 ô legal để đặt target | Không phải scenario3km/35targets cũ. GPU chưa tùy ý đổi số UAV/grid/map dù comment đầu YAML legacy còn nói có thể đổi tự do; lấy validator GPU làm hợp đồng thực thi. |
| NFZ/blackout | Cylinder building footprints tham gia flight guard; hai vùng blackout dùng cho radio | Chưa có lớp NFZ đa giác/động độc lập. Không coi blackout RF đồng nghĩa cấm bay. |
| Pin/động lực học | e_max=180000J, all_depleted, fixed altitude; enable_accel_energy=false | Mô hình động học và năng lượng steady-level có giả thiết; chưa là full 6-DOF, hạ cánh hoặc recharge. |
| Motion/safety | XY fixed50m, clamp L2,66candidate,3sweep, swept detector | L2 wall fix đúng nhưng không chứng minh joint feasibility khi nhiều UAV/tòa nhà cùng ép nhau. |
| Sensor/belief | Pd.85/Pf.03,FOV50m,occlusion3D,mass.8,eta.02,UQ8.8,origin-ledger sum | Calibration, count-aging, packet AoI và paper max-fusion là các giả thiết khác nhau. Không bypass guard eta0. |
| Actor | Encoded518; own XY/velocity/battery/radio/queue, GCS features, pooled maps | Thiếu explicit peer kinematics. Khoảng cách bốn biên suy ra được từ XY; không thể nói hoàn toàn không có thông tin tường. |
| Critic | CTDE từ6×518 joint observations | Không tự động là toàn bộ Markov state của radio; không đổi equations MATD3/MASAC. |
| Reward | GCS TP25+AoI bonus6, local uncertainty credit.5; safety100,blocked_motion.2 và FP costs | Local credit không tự oracle leak. Có thể đổi objective so GCS-only; cần ablation. Chi phí correction đã có nên tuple safe phải xử lý đúng dấu. |
| Network | Custom microsecond discrete-event CSMA/CA; HELLO, MAC/E2E ACK, TTL, custody, retry | Golden tests nội bộ không bằng full packet-event parity với upstream SimPy. |
| RF | TX.1W,noise4.389577e-13W,12dB/cylinder | Noise khớp upstream hiện hành, khác TableI2025. Loss12dB chưa calibration ITU/3GPP. |
| Warp | World và learner cùng physicalGPU1, cuda:0 khi visible=1 | Chưa có V84T4,2GPU hoặc convergence; E4096×3 chỉ integration smoke. |

## 2. Vì sao L2 wall correction có cơ sở?

Đặt s=||v||, giới hạn gia tốc a. Backup phanh dọc -v có quãng dừng từng trục d_i=|v_i|s/(2a). Còn đủ khoảng cách cho backup này là điều kiện đủ bảo thủ, không phải toàn bộ viability kernel.

Với brake-before-move của code, c=max(1-a*dt/s,0),v_next=c*v. Khi s≥a*dt, phần dư tới tường theo hướng chuyển động tăng thêm |v_i|a*dt²/(2s)≥0 sau bước phanh. Khi s<a*dt, v_next=0 và điều kiện cũng được giữ. Đây là lập luận cho tường trong không gian trống, dưới backup đồng tuyến.

V83 fallback giữ candidate an toàn hình học gần v_brake nhất khi không tìm được candidate thỏa mọi peer. Điều này không chứng minh separation với peers; detector vẫn terminate khi cần và giữ nguyên pos/vel của world lỗi.

Bằng chứng:
- 08_v82_wall_RED.json:2/300; seeds125,150 tái hiện được cùng tọa độ.
- 09_v83_wall_GREEN.json:0/300;10.000 trạng thái ở biên tập đủ an toàn,4 bước phanh,0 violation.
- 11_wall_warp.json:10.000 trạng thái CPU và10.000 CUDA,0 backup rejected; sai khác vận tốc≤1.1e-6.
- Full CUDA suite kiểm cả CPU–Warp4world×16step và fail-closed.

Test “6UAV vào bốn góc” của V83 dừng khi có pair violation. Vì vậy0 boundary/building trong tiền tố đã chạy không chứng minh cả6 UAV tiếp tục đủ40 bước. Các JSONL V82 cũng không lưu mọi position/velocity để xác nhận tất cả tọa độ/bước mà báo cáo V83 trích.

10.000 mẫu là kiểm thử, không phải tự nó là chứng minh định lý. Lập luận trên không bao gồm peers, obstacle geometry, model uncertainty hoặc mọi trạng thái ban đầu.

## 3. Raw/executed/dual: xác định đúng critic và reward

Nếu shield S nằm trong môi trường thì f_S(s,a)=f(s,S(s,a)), reward R_S bao gồm correction cost. Tuple(s,a_raw,R_S,s_next) khớp môi trường đó. Nhiều raw actions có cùng kết quả có thể tạo Q plateau thật; không tự chứng minh overestimation hay data mismatch.

Physical/safe-action critic là một formulation khác. Paper Gros phân tích projected policy và các điều kiện gradient; nó không chứng nhận gradient của shield rời rạc66candidate hiện tại là unbiased.

| Mode V84 | Tuple / objective | Giới hạn |
|---|---|---|
| raw, mặc định | Một tuple proposal, reward môi trường gốc | Baseline shielded-environment; không đổi reward. |
| raw_penalized | Một tuple proposal, reward gốc−lambda*correction_cost | Adaption-penalty objective được công bố; native/training reward log riêng. |
| executed | Action v_safe/vmax trên bước đã commit; reward bỏ lại chi phí correction của proposal | Experimental relabelling; chưa chứng nhận projection idempotent hoặc toàn bộ actor-target semantics. |
| dual | Safe tuple ở trên và raw penalized tuple khi có correction hình học | Hướng both có trong literature; thay phân phối replay, không tự chứng minh tốt hơn. |

Cost V84=sum_u||(v_nominal−v_safe)/vmax||², chỉ UAV active và world không physical failure. Clipping gia tốc tự nhiên không bị gọi là nguy hiểm. Lambda1 là cấu hình thử, không phải giá trị tối ưu.

## 4. Bảy lỗi có RED→GREEN

| Lỗi | Bằng chứng | Sửa V84 và lý do |
|---|---|---|
| Phạt cả actuator clipping | v_nominal=v_safe nhưng raw command khác vì amax; V83 vẫn thêm unsafe tuple | So nominal với safe; mask inactive và failed. Plant dynamics khác geometric safety intervention. |
| Dùng terminated thay failed | Successful terminal đã commit nhưng executed label còn raw | Chỉ failed flag quyết định motion không commit. |
| Reward safe tuple còn correction cost của proposal | Đổi label sang safe nhưng giữ w_blocked_motion; penalty còn dùng raw−exec | Safe tuple bỏ lại correction cost gốc; raw penalty theo nominal−safe. Relabelling vẫn experimental. |
| Warmup bị tuple tổng hợp đẩy sớm | Original dual học trước4096 real transitions | Lịch học theo vector_step×envs và đủ batch; tách real_transitions/replay_rows/extra_rows. |
| Sparse JSONL bỏ mất events | E2,horizon2,steps4,log_interval3: final4 truncations nhưng cộng JSONL chỉ2 | Reason counters theo cửa sổ từ log trước; có window_start và cumulative_episode_count. |
| Diagnostics dùng sai activation | Actor SiLU nhưng `_actor_output_diagnostics` hard-code ReLU, nên pre_tanh/entropy không mô tả đúng Actor | Dùng cùng activation của Actor; test cả shared/independent, ReLU/SiLU, kiểm state và RNG không đổi. |
| Exception làm mất totals | Inject failure ởstep3 sau2steps đã hoàn tất: trả truncations0 | Đọc lại device totals trong error path; giữ counters của completed steps, báo nếu readback lỗi. |

05_v84_semantics_RED.log:6 failure. Fixture expected dtype được chỉnh float32 cho khớp replay, không đổi giá trị mong đợi. 07_v84_replay_regression_GREEN.log:23/23 gồm regression V82. Suite trước sửa diagnostics: CPU 435 PASS/5 skip, gated CUDA 436 PASS/4 skip. Suite cuối và raw logs tương ứng được ghi ở SOURCE_FINAL_FROZEN.json; không gộp các lượt kiểm tra thành một lượt.

Bản sửa diagnostics được phát hiện ở bước kiểm cuối: fixture ban đầu vô tình còn 3 continuous dimensions đã được sửa đúng 2 motion dimensions trước khi đánh giá RED. Log 18 cho 2 test ReLU PASS, 2 SiLU FAIL; log 20 cho đủ 4 activation/shared cases và 6 runner cases PASS. Log 19 ghi lỗi chọn sai tên test ở lệnh kiểm tra; giữ nguyên để không che lịch sử.

Code:
- `uav_network/algorithms/matd3/diagnostics.py`: khớp activation thật; thêm tên metric activation-zero tổng quát, giữ alias lịch sử. Tỷ lệ zero trên một batch không phải dormant-neuron fraction.
- replay_v84.py tập trung action/cost/reward/tuple semantics; raw reward giữ nguyên.
- run_v81_real_rollout.py giữ tên file, báo V84; warmup thật, window counters, exception recovery, objective labels.
- rollout_v82_contracts.py thêm raw_penalized opt-in.
- test_v84_replay_and_logging_semantics.py kiểm hành vi Torch/Warp và runner.
- Giữ3 sửa physics V83 sau kiểm chứng; không thay sensor/RF/obs hay learner losses.

## 5. Các lượt chạy thật

Suite cuối: **439 PASS/5 skip CPU; 440 PASS/4 skip gated CUDA; 100 subtests mỗi lượt**. Gated CUDA bật RUN_V80_CUDA_GATE=1 và device cuda:0 sau khi chỉ expose physical GPU1. Bốn skip gồm ba V79 opt-in hard-code device khác và một kiểm tra chỉ dành host không CUDA. Không gọi toàn bộ 440 test là kernel CUDA test.

| Run | E/steps | Updates | Extra rows | Safety | Boundary | Pair | Horizon | Full15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 04_original_v83_masac_dual_e64_s300 | 64/300 | 269 | 19200 | 0 | 0 | 0 | 0 | 0 |
| 04_original_v83_masac_dual_e64_s900 | 64/900 | 869 | 57600 | 0 | 0 | 0 | 64 | 0 |
| 04_original_v83_masac_raw_e64_s300 | 64/300 | 237 | 0 | 3 | 0 | 3 | 0 | 0 |
| 04_original_v83_matd3_dual_e64_s300 | 64/300 | 251 | 14657 | 1 | 0 | 1 | 0 | 0 |
| 04_original_v83_matd3_dual_e64_s900 | 64/900 | 851 | 53050 | 5 | 0 | 5 | 58 | 3 |
| 04_original_v83_matd3_raw_e64_s300 | 64/300 | 237 | 0 | 21 | 0 | 21 | 0 | 0 |
| 13_v84_masac_dual_e4096_s3 | 4096/3 | 2 | 2674 | 0 | 0 | 0 | 0 | 0 |
| 13_v84_masac_dual_e64_s300 | 64/300 | 237 | 8535 | 3 | 0 | 3 | 0 | 0 |
| 13_v84_masac_dual_e64_s900 | 64/900 | 837 | 19148 | 3 | 0 | 3 | 62 | 0 |
| 13_v84_masac_raw_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13_v84_masac_raw_e64_s300 | 64/300 | 237 | 0 | 3 | 0 | 3 | 0 | 0 |
| 13_v84_masac_raw_penalized_e64_s300 | 64/300 | 237 | 0 | 3 | 0 | 3 | 0 | 0 |
| 13_v84_matd3_dual_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13_v84_matd3_dual_e64_s300 | 64/300 | 237 | 12897 | 2 | 0 | 2 | 0 | 0 |
| 13_v84_matd3_dual_e64_s900 | 64/900 | 837 | 50637 | 5 | 0 | 5 | 56 | 4 |
| 13_v84_matd3_raw_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 13_v84_matd3_raw_e64_s300 | 64/300 | 237 | 0 | 21 | 0 | 21 | 0 | 0 |
| 13_v84_matd3_raw_penalized_e64_s300 | 64/300 | 237 | 0 | 11 | 0 | 11 | 0 | 0 |
| 24_final_v84_masac_dual_e4096_s3 | 4096/3 | 2 | 2674 | 0 | 0 | 0 | 0 | 0 |
| 24_final_v84_masac_dual_e64_s300 | 64/300 | 237 | 8535 | 3 | 0 | 3 | 0 | 0 |
| 24_final_v84_masac_dual_e64_s900 | 64/900 | 837 | 19148 | 3 | 0 | 3 | 62 | 0 |
| 24_final_v84_masac_raw_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24_final_v84_masac_raw_e64_s300 | 64/300 | 237 | 0 | 3 | 0 | 3 | 0 | 0 |
| 24_final_v84_masac_raw_penalized_e64_s300 | 64/300 | 237 | 0 | 3 | 0 | 3 | 0 | 0 |
| 24_final_v84_matd3_dual_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24_final_v84_matd3_dual_e64_s300 | 64/300 | 237 | 12897 | 2 | 0 | 2 | 0 | 0 |
| 24_final_v84_matd3_dual_e64_s900 | 64/900 | 837 | 50637 | 5 | 0 | 5 | 56 | 4 |
| 24_final_v84_matd3_raw_e4096_s3 | 4096/3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24_final_v84_matd3_raw_e64_s300 | 64/300 | 237 | 0 | 21 | 0 | 21 | 0 | 0 |
| 24_final_v84_matd3_raw_penalized_e64_s300 | 64/300 | 237 | 0 | 11 | 0 | 11 | 0 | 0 |

Đã có 30 cohort: 6 V83 bất biến, 12 V84 trước sửa diagnostics, 12 V84 cuối. Đối chiếu 12 cặp trước/sau cho native rewards, counters, mission snapshots và core optimizer metrics khớp; max sai khác metric kiểm tra = 0. Không suy từ đó thành lưu đủ mọi world state hay weights bitwise. Xem 26_FINAL_RUN_ACCOUNTING_VERIFIED.json.

Ở 300 bước, cùng 19.200 transitions thật và 237 updates: MATD3 raw/penalized/dual có 21/11/2 safety failures; MASAC 3/3/3. Ở 900 bước dual (837 updates), MATD3 có 5 pair failures, 4 episode full15, 56 horizon ends; MASAC có 3 pair failures, 0 full15, 62 horizon ends. Tất cả final cohorts có boundary flags=0. Đây là seed44 online diagnostics, không phải kết quả đánh giá policy độc lập.

E64 cohort: H64,B128,seed44,horizon900,replay65536. E4096 smoke:B8192,H64,3steps. CUDA_VISIBLE_DEVICES=1 nên cuda:0 là physicalGPU1; không dùng process GPU0.

Các JSON V83 nguyên bản còn status string V82_ROLLOUT_COMPLETED dù version=V83. Đối chiếu bằng SHA ZIP đầu vào và mã bất biến, không suy phiên bản từ tên PASS/status. git_commit=null ở source giải nén; git_dirty=false tại đó không phải bằng chứng một Git checkout sạch.

Original V83 raw/dual không có cùng update budget vì warmup dựa row replay. V84 chốt cùng real-transition budget. Không gán mọi chênh lệch V83 dual cho learning tuple.

Các run prefix `13_` được giữ như bằng chứng trước khi sửa diagnostics; các run prefix `24_final_` dùng mã cuối có diagnostics đúng. Các trường pre_tanh/entropy/tanh-derivative do probe hard-code ReLU ở V81–V83 và `13_` không được dùng như phép đo chính xác của Actor SiLU. Điều này không đổi safety counters hay learning trajectory; so sánh `13_` với `24_final_` kiểm trực tiếp việc giữ nguyên hành vi.

Raw V84 đối chiếu raw V83 là kiểm tra bảo toàn baseline. Dual/penalty còn chỉ seed44; success trong online rollout có học không phải held-out evaluation/convergence.

Safety flags là violation của candidate bị detector chặn, không phải số UAV chết đã commit. Boundary/pair có thể cùng bật. Full15 GCS success khác horizon truncation. Snapshot delivery cuối là live worlds sau autoreset, không phải mean completed-episode evaluation.

## 6. Những nhận định khoa học phải điều chỉnh

- “Raw replay là lỗi P0” quá mạnh; naive/both đều có trong paper. Chọn mode cần MDP/action/reward rõ ràng.
- “10.000 mẫu chứng minh định lý” sai cách diễn đạt; xem giả thiết mục2.
- “Đã hết D2H stall” chưa đúng. Có counter batching, nhưng overflow guard, optimizer finite checks, scalar metrics và dual nonzero vẫn có host readback. V84 sửa mất thông tin log; chưa tuyên bố throughput tăng.
- Hai positive hits stale xuống mass.9 sau16.75s là đúng. GCS true-positive latch không bị xóa chỉ vì mass giảm.
- Prior15/400 không đúng đồng đều cho generator: target chỉ đặt trên392legal cells, prior ô legal15/392, log-odds≈−3.224; ô bị loại có prior0.
- ln(.85/.03)=3.344 và ln(.15/.97)=−1.867 là đúng; không tự quy định ánh xạ duy nhất sang DS masses.
- Actor có own XY và velocity nên suy ra khoảng cách bốn biên. Thiếu peer kinematics là thật; HELLO age/validity cần schema/checkpoint mới.
- “Local credit làm đứt gãy communication”, “−100 gây paralysis” chưa có ablation nhân quả. Không giảm safety penalty để làm đẹp số.
- uint16 raw counts cùng2bytes/element, đủ900increments; nhưng phải sửa encoder/decoder/fusion/API/validator/payload semantics và fixtures, không bỏ /256 một nơi.
- arXiv1805.00826 là overview LTE drones, không phải chính TR36.777; nguồn chuẩn là portal3GPP và version cụ thể.
- UavNetSim-v1 là alias redirect cùng repo, không phải404. Pin hiện tại c2703f24eb58cdb557a5f67ded5c7b0368a964ed.
- NLOS15–23dB của mô hình air-to-ground không tự hiệu chuẩn mỗi A2A cylinder; chỉ gọi sensitivity sweep khi chưa đo đúng bối cảnh.

## 7. Đủ24 mục V80 và việc tiếp theo

04_V80_24_ISSUES_UPDATED.csv giữ nguyên proof/source/action cũ và trạng thái mới cho từng ID. #01–08 giữ sửa runtime; #09 đóng thêm wall-corner nhưng joint viability OPEN; #10 spawn không tự sai khi initial velocity0; #11 obs migration; #12/#23 raw-count/eta0 guard; #13 DS same-ledger test không bằng paper fusion parity; #14/#15 reward/airtime ablation; #16 noise config difference; #17 calibration/#18 upstream event parity OPEN; #19 scale loss không tự learning-rate bug; #20 convergence/#21 2GPU/#22 lifetime-success distribution OPEN; #24 logging đúng hơn nhưng performance chưa đóng.

Thứ tự:
1. Joint safety state bank/adversarial nhiều seed, từ tập viable được định nghĩa; theo dõi correction/termination và fail-closed, không suy bảo đảm từ3sweep.
2. Chốt raw MDP hoặc physical-action formulation; so sánh equal real transitions/updates, effective replay age và sample weighting.
3. HELLO nhận qua RF có timestamp/TTL/validity; obstacle/blackout features chỉ từ map/sensor được phép biết. Version hóa obs/checkpoint.
4. Raw-count/Bayes là migration và ablation riêng; tách target persistence khỏi message AoI bằng protocol cụ thể.
5. GCS-only/local/potential shaping với cùng seeds/budgets; không tự gọi min-uncertainty reward là policy invariant.
6. Match upstream packet IDs/traffic/ACK/retry/TTL/queue/custody/energy; trace trước, PDR/delay/AoI distribution sau. Đổi dt phải đồng bộ mọi clock.
7. Sau các gate mới mở E4096×900/full budget, checkpoint/resume và held-out discovery/delivery/coverage/AUC/capped delay.

## 8. Lịch sử và nguồn

Giữ notebook/log Kaggle V77/V78, V79CPU, Vast V80/V82 và toàn bộ lượt hiện tại. Không submit thêm Kaggle khi Vast còn dùng được; không đọc/in key.txt. CSV lịch sử là record tìm được, không nhất thiết một record=một run độc lập.

Nguồn gốc ở05_SOURCES.md: Krasowski TMLR2023 mục2.5; Gros IFAC2020; Agrawal/Sreenath RSS2017; Batra decentralized swarm; NVIDIA Warp docs;3GPP portal. Nguồn ICML A*/T-IV Q1 và Zhang DS từ V82 giữ kèm. Log V80 ReLU đo được fraction đầu ra bằng 0, chưa đủ kết luận neuron chết lâu dài; không xem SiLU là bằng chứng đã giải quyết toàn bộ nguyên nhân safety. Không gán A*/Q1 cho mọi nguồn hay tiêu chuẩn.

Chưa có V84T4/2GPU, nhiều-seed confidence interval, full upstream SimPy event parity hoặc bảo đảm mọi6-UAV state đều feasible. OPEN có lý do và phép thử, không bị đánh dấu FIXED.
