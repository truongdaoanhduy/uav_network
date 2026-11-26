# V91 — KIỂM TOÁN KỊCH BẢN SCENARIO-FIRST — 27/11/2026

**KẾT LUẬN: CANDIDATE_NOT_FROZEN. NO_GO_BASELINE / NO_TUNING.** Không sửa thuật toán MASAC/MATD3. Chạy trên Vast GPU1 (CUDA_VISIBLE_DEVICES=1, torch cuda:0).

## 1. Phạm vi được cố định
Bản đồ 1000×1000m, 6 UAV, 15 mục tiêu tĩnh trên 392 cell hợp lệ, belief map 20×20, độ cao50m, dt1s, horizon900. Vmax15m/s, amax5m/s², khoảng cách an toàn50m; FoV50m, Pd=.85, Pf=.03. Continuous action XY, discrete network NO_SEND/GCS/PEER. TX power0.1W cố định; enable_tx_power_control=false, KHÔNG có TX action. V90 gốc commit 17f505d1aab0d5b588a9967941e71bd89367f1d9; V91 là worktree riêng. YAML V90 DS giữ nguyên, Bayes V91 là opt-in.

## 2. Test đã chạy và có log
| Hạng mục | Thực tế | Raw evidence |
|---|---|---|
| V90 CUDA full | 486 PASS, 4 skipped, 101 subtests | v90_cuda_full_pytest_20261127.log |
| V90 dynamic random | 192 world-episode x900, 3 seed; 0 pair/boundary/building/safety-failure/overflow | v90_random_stress_3seeds_64worlds_900steps.json |
| V91 Bayes focused CPU+CUDA | 5 PASS | v91_bayesian_test_first.log |
| V91 full CPU | 489 PASS, 6 skipped, 102 subtests | v91_full_cpu_pytest.log |
| V91 full CUDA | 491 PASS, 4 skipped, 102 subtests | v91_full_cuda_pytest.log |
| V91 4096×900 stationary | 8/8 internal gate PASS, 4096/4096 horizon, 0 safety failure/overflow/reward mismatch; 61.1662s | v91_bayesian_e4096_s900.json |
| V91 oracle target-route feasibility | 3 seed đạt 15/15 targets ở steps 150,163,212, 0 safety failure/false-confirm observed | v91_oracle_feasibility_3seeds.json |
| Independent-cell DP | 5 observations: DS miss .16479, Bayes miss .026612; expected FPever among 377 empties .359 vs .407 | v91_bayes_exact_dp.json |
| Network differential | 4 PASS, 1 FAIL, 1 OPEN | v91_network_six_gate_vast_cuda.json |
| Action semantic probe, 8 env×200 stationary | NO_SEND 0 GCS; GCS 9552 packets/7 TP/1 FP; PEER 9552 peer packets/0 GCS targets | v91_discrete_action_ablation_stationary.json |

**Giới hạn:** stationary pass không chứng minh khả năng RL học, replay/backward fit VRAM hay truyền thông trung thực với UavNetSim. Oracle planner dùng vị trí mục tiêu thật trong DIAGNOSTIC, không đưa vào actor; không phải kết quả decentralized RL.

## 3. Belief map, đúng/sai và code sửa
Zhang et al. IEEE T-IV 2024 https://ieeexplore.ieee.org/document/10388472 đề xuất DS uncertainty/target map. V90 DS fixed masses mf=me=.8 không phải calibrated posterior khi Pd=.85, Pf=.03. V91 opt-in independent-cell Bayes log-odds:
L = ln[(15/392)/(1-15/392)] + p ln(.85/.03) + n ln(.15/.97).
Target posterior=sigmoid(L), empty=1-target, extra uncertainty indicator=.5^(p+n). Kênh extra là **coverage heuristic**, KHÔNG phải DS mass hoặc Shannon entropy; ba giá trị không nhất thiết tổng1. Mô hình exact trên 1 cell với measurement conditionally independent và prior marginal15/392, KHÔNG exact joint posterior bản đồ có chính xác15 target.
Raw uint16 counts theo origin+generation được hợp nhất thông qua packet, không double count cùng nguồn; evidence tĩnh không suy giảm theo AoI nhưng data freshness được đo riêng. Code đã kiểm công thức trên CPU/CUDA; DP là xác suất lý thuyết, không phải UAV experiment. Khi đổi inference backend, một phần reward uncertainty semantics cũng đổi: **đây là ablation inference+reward nếu chưa khóa reward tương đương**.

## 4. Mạng — sáu gate khác nhau, không nhập nhằng
UavNetSim GitHub https://github.com/Zihao-Felix-Zhou/UavNetSim commit c2703f24eb58cdb557a5f67ded5c7b0368a964ed; article https://arxiv.org/abs/2507.09852.
- 01 DCF timing: PASS DIFS50us,SIFS10us,slot20us,CWmin31,ACK timeout180us,ACK airtime120us như upstream.
- 02 DCF CW sample distribution: **FAIL upstream parity**. Upstream randint(0,CW-1) = 0..30 first try; Warp modulo(CW+1)=0..31. Đây là khác implementation, KHÔNG đủ kết luận Warp vi phạm IEEE. Không âm thầm sửa.
- 03 PHY CUDA/NumPy: PASS internal maxrel1.87e-7; KHÔNG phải RF field calibration.
- 04 hidden terminal: PASS geometric CCA+GCS SNR topology; chưa chạy concurrent packet collision PDR.
- 05 GPU queue bounded test: PASS 4world×25, 432 GCS deliveries, qmax0/64, 0 overflow. Chưa đủ chứng minh TTL/custody under congestion.
- 06 UavNetSim upstream paired macro 32 seeds: **OPEN**. Upstream Greedy/8192-bit payload/HELLO360us; Warp HELLO-DV compact packet/HELLO440us. Workloads khác nhau nên không được gọi bit-exact UavNetSim-v1.
- RF building loss12dB hiện minh họa, chưa calibration thực. Bianchi IEEE JSAC 2000 https://ieeexplore.ieee.org/document/840210 dùng bão hòa và lý tưởng, KHÔNG bảo chứng multi-hop hidden terminal.

## 5. Kịch bản & những quyết định chưa chốt
- Action XY có ý nghĩa. Fixed altitude nên không có vz/takeoff/landing. TX power fixed không phải action, không thêm cho đủ số chiều.
- Compact3 network action có hiệu quả đo được, nhưng PEER-only chưa tạo GCS confirmations. Cần chứng minh PEER→forward GCS trong partition/store-carry bằng packet trace.
- Reward 8 components tổng đúng scalar; w_p2p=0 và lambda_b=0 không nhất thiết code bug nhưng chưa tạo utility incentive cho PEER và energy. Chốt mục tiêu trước baseline, tránh thưởng packet spam. Bật false-positive unlatch, log FP-ever và FP-active riêng.
- Safety V90 Tier-2A fail-closed 0/192 random, không chứng minh universal invariant. Spawn separation ban đầu123.69m, FoV/occlusion tested.
- 4096 stationary còn ~8.255% pin mean ở step900, 3375 TP trên 61440 targets và 0 full completion, còn oracle 3/3 đạt full15 với thông tin đặc quyền. Mission khả thi trong thử nghiệm nhưng chưa có non-oracle baseline.

## 6. Gates cần đóng trước SCENARIO_FROZEN
(1) PHY/MAC: matched 32seed upstream paired runs, CW strategy, hidden terminal actual PDR, retries, TTL conservation, distance/building attenuation calibrated. (2) Belief: reliability, FP/FN with non-oracle mission, origin packet no double count, DS vs Bayes benchmark. (3) Safety: dynamic geometry/building/blackout scripted held-out, rigorous rollback semantics. (4) Action/reward: lock comm paths, targets/false confirmations, uncertainty, energy/AoI, observation/action schema and metrics; prove PEER forwarding useful. (5) 4096×900 and several seeds deterministically test, source/YAML SHA immutable.
Only after all science and engineering criteria are verified: tag SCENARIO_FROZEN, THEN build original unmodified MATD3/MASAC baseline, THEN tuning. No Kaggle new GPU runs, no branch main modifications.
