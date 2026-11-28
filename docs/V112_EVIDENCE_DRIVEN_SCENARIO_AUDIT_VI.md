# V112 — KIỂM TOÁN KỊCH BẢN DS-ONLY: CODE + GPU + KHOA HỌC

**Ngày:** 28/11/2026. **Quyết định:** `NOT SCENARIO_FROZEN`. **Không training, không sửa 30 tệp thuật toán MATD3/MASAC, không push GitHub.** Đọc tài liệu này cùng `outputs/v110_reviewer_audit/02_E2E_STATE_ACTION_MATH.md`, `05_FULL_CONFLICT_CAUSAL_MATRIX.csv`, `docs/V111_CONFLICT_RESOLUTION_AND_PROVENANCE_VI.md`, cùng bằng chứng thô `outputs/v112_validation/`.

## 1. Thứ tự gate và mức độ bằng chứng

1. **ZIP INPUT:** `UAV_V111_CLEAN_SCENARIO_AND_AUDIT_20261128.zip`, SHA-256 `5e96959368ea8abe9df43d74a68cfeb3ebbb0090c03cf99fe94c1a3195f4d591`. ZIP integrity PASS, copy SSH lên Vast trùng SHA.
2. **Code QA:** trên Vast RTX3060, full pytest 591 PASS / 6 SKIP / 114 subtests PASS, exit 0, 41.09 s. Báo cáo V109 CUDA 4096×900 gentle_patrol được giữ trong `outputs/v109_vast/`; **không xem là proof cho mọi action**.
3. **Confound C20:** 6 YAML Base, RF-only, RF+PEER, RF+PEER+MULE, Base+PEER và Base+TH0.99. Đối chiếu full dataclass Scenario và assert **mỗi edge khác đúng một trường**.
4. **RF/PEER/MULE và DS:** 5 master-seed độc lập 44–48 ×4 thế giới GPU/seed ×900 steps, cùng heuristic NON-ORACLE local obs. Đo **first terminal**, không cộng thời gian sau terminal; thống kê bootstrap phân cụm theo 5 master seeds, *không* coi 20 thế giới là 20 seed độc lập.
5. **C14:** 4096 GPU worlds ×900 transitions với action ngoài gentle_patrol, giám sát sticky mọi bước trước masked auto-reset. PASS không chứng minh đảm bảo an toàn cho mọi policy.
6. **Freeze gate:** chỉ khi còn chứng cứ DS correlated mission-level; packet-event UavNetSim SimPy↔Warp parity; RF/sensor/energy calibration; reachability và policy/action semantics. **Không chuyển Phase 2 MATD3/MASAC canonical khi chưa chốt.**

## 2. Các quyết định action / observation / state hợp lệ

- Actor policy baseline canonical nên nhận `(E,6,553)` được encode từ 549 local observation + 1 `alive` true từ SOC/`obs[4]>0` + 3 communication validity mask, `critic state = E×(6×553) = E×3318`, không đưa true target map vào actor/critic. *Cảnh báo:* test taint V111 mới xác minh ngoài FoV cho fixture một seed; cần taint qua nhiều trajectory và kênh telemetry để chứng minh rộng.
- Action vật lý: continuous XY normalized `a_x,a_y ∈[-1,1]`, qua kinematic projection `v_nom=clamp(a,vmax)` và acceleration disk `a_max=5`, vận tốc giới hạn 15 m/s, độ cao UAV 50m cố định. GCS là đài cố định ở `(50m,50m,15m)` trong dữ liệu mô phỏng, **không biết bay, không là action**.
- Communication environment mode `compact_3`: `0=NO_APP_SEND`, `1=GCS_SEND`, `2=PEER_SEND` có action mask. **NO_APP_SEND không có nghĩa radio tắt:** HELLO/ACK/routing control còn hoạt động. Khi `PEER=false`, lựa chọn 2 không có tác dụng. Baseline **MATD3/MASAC gốc là continuous** nên chỉ học XY, còn communication phải được điều khiển bằng heuristic cố định giống nhau cho mọi baseline; không tuyên bố algorithm học discrete communication nếu chưa triển khai kiến trúc hybrid chính thức (Phase 2).
- `tx_power_w=0.1 W` cố định thuộc giả định mạng, **không phải action RL**. Lý do: nghiên cứu muốn phân tách lập kế hoạch đường bay/belief delivery khỏi power control, tránh thêm chiều action không xác định đóng góp; vẫn xuất hiện trong SNR, CCA và năng lượng TX. `enable_tx_power_control=false` active. UavNetSim có **tham số TRANSMITTING_POWER** trong `utils/config.py`, không bắt buộc dùng RL tối ưu công suất.
- DS-only: `belief_inference_mode=dempster_shafer`, `m_f=m_e=.8` evidence mass, nguyên lý DS từ Zhang 2024; `pd=.85,pf=.03` là xác suất sinh mẫu Bernoulli tự chọn, **không phải mass từ paper được hiệu chuẩn cho sensor**. Bản đồ 20×20 belief cho **15 static targets**. CPU có legacy Bayes code trong archive/compatibility nhưng **YAML active từ chối Bayes**, không được gọi đây là thuật toán Bayes→DS đang chạy.
- `obs_pool_mode=semantic`: phép gộp 5×10×10 local map không có tổng mass bằng 1 => gọi đúng `pooled spatial belief features`, không gọi là BBA chuẩn.
- Reward: `first_visit` chỉ thưởng mỗi ô mới; xác nhận TP/FP và latency thêm reward riêng; `w_p2p=0`, `lambda_b=0`, `continuous_aoi_penalty_w=0` **không đóng góp reward hiện hành**, không in dưới dạng term đang hoạt động trong Methods. `ds_aoi_eta=0`: có AoI telemetry nhưng **KHÔNG** AoI-discounted DS của Xue 2026.

## 3. Toán học và nguồn gốc không được đánh đồng

**DS từ Zhang 2024 (phương pháp, không phải mọi hệ số V111):** với đếm positive `p`/negative `n`, `a=(1-m_f)^p`, `b=(1-m_e)^n`, `D=a+b-ab`, và

\[
m(T)=\frac{(1-a)b}{D},\quad m(F)=\frac{a(1-b)}{D},\quad m(U)=\frac{ab}{D}.
\]

Từ `m_f=m_e=.8`, `p,n` nguyên và không decay: `confirm_th=.90` và `.95` **tương đương trong miền count**, cả hai cần `p-n≥2`; `.99` cần `p-n≥3` khi mass U đáp ứng ceiling. Đây là đặc thù rời rạc của kịch bản, **không phải khẳng định hai threshold tương đương với mọi loại DS khác**. Origin-ledger cộng SUM khác công thức đếm MAX của Zhang, cần kiểm chứng tránh double count / phụ thuộc quan sát.

**UavNetSim công bố và commit source có giá trị nhiễu khác nhau:** bài UavNetSim-v1 `https://arxiv.org/html/2507.09852v1` công bố bảng thí nghiệm noise `4×10^-11 W`; file `utils/config.py` tại commit `c2703f24eb58cdb557a5f67ded5c7b0368a964ed` khai báo `THERMAL_NOISE_DENSITY_DBM_HZ=-174`, `RECEIVER_NOISE_FIGURE_DB=7` và `noise_power_watt()` tính qua bandwidth. Với 22MHz, suy ra 4.389577×10^-13 W. Base theo model nguồn noise thermal; RF-stress theo TABLE trong paper. Chênh 19.6dB, **đều có nguồn nhưng không thể gọi cùng một cấu hình**. UavNetSim `TRANSMITTING_POWER=0.1 W` là radio parameter, không bắt RL dùng TX head.

**Sai khác gói tin đang mở (C33):** chính source cũ `scripts/gpu/v98_simpy_vs_warp_mac_parity.py` khai báo Warp `data_us=280+32*entries` (một entry ~312µs) vs upstream 1024-byte payload + header ~8656bits/2Mbps = 4328µs; **độ dài không tương đương** nếu so sánh như cùng traffic model. Nén evidence là lựa chọn *ứng dụng* có thể hợp lý, nhưng phải mô tả encoding/trường hợp truyền, băng thông sử dụng và làm packet-size sweep; **không được claim packet-event SimPy parity**. Các khác biệt khác: contention window, ACK channel hold, retry-route invalidation, polling vs event-driven.

**Năng lượng:** công thức propulsion từ Zeng 2019 là nguồn cấp độ mô hình; battery 180kJ, avionics, TX overhead và hệ số năng lượng là thiết kế của V111, cần đo/sweep. `E_tx=P_tx·t_air` xuất hiện trong UavNetSim paper nhưng không đủ để nói đã hiệu chuẩn pin của UAV thực. 12 dB/building và chiếu an toàn là *scenario/implementation assumptions*.

## 4. Kết quả thử nghiệm bước 3 (đọc số gốc trong JSON trong ZIP)

| Config / thay đổi so với cặp đối chứng | GCS delivery (%) | Đếm PEER trung bình | Safety/Overflow |
|---|---:|---:|---|
| Base (thermal noise, PEER OFF) | 98.000 | 0 | 0 / 0 |
| RF-only (paper RF-stress noise, PEER OFF) | 56.667 | 0 | 0 / 0 |
| RF-stress, PEER ON / MULE OFF | 24.000 | 1,430.05 | 0 / 0 |
| RF-stress, PEER ON / MULE ON | 24.667 | 1,275.45 | 0 / 0 |
| Base RF, PEER ON / MULE OFF | 96.333 | 320.60 | 0 / 0 |
| Base RF, confirm threshold 0.99 | 95.333 | 0 | 0 / 0 |

**So sánh cặp theo seed (right–left, bootstrap 95% 5 seed clusters):** PEER trong RF-stress làm GCS delivery giảm **32.667 điểm phần trăm**, CI **[-40.667, -26.333]**; MULE bổ sung tăng **0.667 điểm**, CI **[-5.000, 6.000]** (không có bằng chứng lợi ích chắc chắn); nâng DS threshold .90→.99 giảm FP từng xảy ra trung bình **0.3/world → 0/world** ở fixture này nhưng cũng giảm delivery **2.667 điểm**, CI **[-4.000, -1.000]**. RF noise riêng làm giảm delivery **41.333 điểm**, CI **[-48.0,-35.0]**. Toàn bộ đây là kết quả **heuristic local non-oracle**, không phải của policy MATD3/MASAC đã học. Kết luận hẹp: **bật PEER không tự động có lợi, trong setup hiện tại còn làm xấu delivery**; không được loại bỏ khả năng relay có lợi ở workload phân mảnh khác. MULE không chứng minh thêm giá trị trong phép thử này.

Cần kiểm soát thí nghiệm tránh nhầm hiệu ứng: thay đổi PEER làm **đổi cả lượng traffic và trajectory gián tiếp qua local observation**, vì controller phụ thuộc observation; đó là **total causal effect of feature toggle under fixed controller**, không phải mô phỏng cố định từng packet ở cùng vị trí. Tương quan theo seed được bootstrap phân cụm, 5 seeds vẫn còn ít để khẳng định tổng quát. Các failure có thể do PEER queue traffic/backoff/retry/drop; số thực ở các JSON `stats_retry`, `stats_drop`, `tx_airtime_s_total`.

## 5. Bảng nguồn học thuật và điều được phép claim

| Nguồn | DOI / URL | Hợp lý để dẫn gì | Tuyệt đối không nhận thay |
|---|---|---|---|
| Zhang et al. 2024 IEEE T-IV | https://doi.org/10.1109/TIV.2024.3352581 | DS belief-map, count-based evidence, search | SUM ledger, các hệ số chưa hiệu chuẩn và `first_visit` là sáng kiến của Zhang |
| Zeng et al. 2019 IEEE TWC | https://doi.org/10.1109/TWC.2019.2902559 | Rotor propulsion power expression | Khẳng định 180kJ đúng pin UAV thật |
| Zhou et al. UavNetSim-v1 | https://arxiv.org/html/2507.09852v1 , https://github.com/Zihao-Felix-Zhou/UavNetSim | Baseline radio/MAC/routing architecture, paper table noise, TX 0.1W | Warp identical packet-event UavNetSim SimPy/Sionna |
| Xue et al. 2026 Electronics | https://doi.org/10.3390/electronics15091875 | AoI-aware DS under spatially structured comm constraints | Coi V111 `ds_aoi_eta=0` là AoI-weighted fusion |
| Denoeux (2008), Conjunctive and disjunctive combination of belief functions induced by nondistinct bodies of evidence | https://www.sciencedirect.com/science/article/pii/S0004370207001063 | Cảnh báo và phương pháp kết hợp evidence phụ thuộc | Chứng minh sensor V111 độc lập |
| TD3, SAC (Fujimoto; Haarnoja et al. 2018) | https://proceedings.mlr.press/v80/fujimoto18a.html ; https://proceedings.mlr.press/v80/haarnoja18b.html | Algorithm canonical continuous baseline sau này | Claim V111 đang train một learner hybrid mới |

**Phân biệt uy tín:** tên venue IEEE Transactions/ICML và mức độ uy tín của công trình **không chứng minh** các giả định mới tự đặt trong YAML; Q1 (journal quartile) và A* (conference ranking) là hai hệ phân hạng khác nhau, phụ thuộc năm/danh mục. Nếu chưa kiểm tra nguồn xếp hạng cho năm, không đóng dấu Q1/A* như một chứng cứ cho một thuật toán.

## 6. Quyết định kịch bản và các thành phần vô nghĩa/thiếu tính năng

- **GIỮ và KHÔNG BIẾN THÀNH ACTION:** `tx_power=0.1W`, XYZ fixed altitude, sensor Pd/Pf as declared modeling assumptions, DS, E2E delivery telemetry. Chỉ thêm power-control ablation nếu mục tiêu paper thực sự là optimize năng lượng/SINR.
- **GIỮ NHƯNG KHÔNG CLAIM ACTIVE REWARD:** `w_p2p=0`, `lambda_b=0`, `continuous_aoi_penalty_w=0`, `ds_aoi_eta=0`, `enable_accel_energy=false`. Dọn khỏi phương trình *active* nếu viết paper; giữ trong config để ablation tùy chọn, không thêm các penalty có thể xung đột với objective mà chưa có lý do/thử nghiệm.
- **KHÔNG ĐƯA LEARNED DISCRETE HEAD VÀO CANONICAL MATD3/MASAC:** actor action XY continuous; comm3 fixed algorithmic heuristic, ghi rành mạch. Dùng cùng comm controller cho hai baseline.
- **ĐÁNH GIÁ LẠI PEER/MULE:** Kết quả V112 phản bác utility positive đối với heuristic hiện dùng. Cần đo update novelty/bytes-per-evidence, queue/backoff, custody, GCS confirmed TP và phân tích throughput; nếu đưa PEER thành đóng góp mới, phải có workload positive and negative và so sánh latency/PDR trade-off. Không mặc định bật PEER trên canonical baseline.
- **CHƯA NÊN THÊM:** Bayesian log-odds, learnable TX head, trộn AoI discount vào DS, entropy bonus mới, tăng thuật toán hybrid, reward bonus mới. Các thứ này sẽ làm confound và tăng chiều giải thích nếu chưa chứng minh mục đích bằng ablation.

## 7. BLOCKER còn mở trước freeze

1. DS correlation dưới **full mission + GCS confirmation**, có kiểm biến quan sát phụ thuộc và threshold sweep; V109 synthetic 50k trial rho0/rho.7 chỉ là một-cell model, V112 5-seed mission dùng giả định Bernoulli hiện hành.
2. Packet-level UavNetSim SimPy-vs-Warp parity với offered load khớp, ACK, CW, interferer, frame bytes, retry, queue, GCS routes. **Hiện chưa có**, không được ký `SCENARIO_FROZEN`.
3. `PEER/MULE` positive utility/policy: cặp V112 PEER có hại, MULE lợi ích không được chứng minh; nghiên cứu tính hợp lý, giải pháp hạn chế gossip/priority/suppress khi route, và chứng minh có lợi riêng.
4. Offline dynamic reachability/battery/network delivery feasibility, sensitivity 12 dB/building, noise, Pd/Pf, DS mass .8, energy.
5. Long-run adversarial motion and full action masks; CUDA pass trên một vài action family không chứng minh recursive feasibility all policies.
6. Phase 2 canonical TD3/SAC-compatible MATD3/MASAC **chưa chạy**, không được đánh đồng gate kịch bản với baseline convergence.

## 8. Hướng xử lý step by step

`B1 zip/hash → B2 V111 full pytest → B3 phân nhân tố C20 + paired 5 seeds → B4 C14 4096×900 adverse modes → B5 điều tra packet mismatch và PEER harmful → B6 correlated DS mission-level / reachability → B7 source-verified freeze candidate → B8 canonical MATD3/MASAC baseline (sau B7) → B9 tuning (sau B8)`. Chỉ ký gate nếu thỏa điều kiện và log cho phép; một gate không có log PASS thì phải báo mở/blocked.

## 9. FINAL V112 GPU C14 stress gate (Vast, 28/11/2026)

Đã kết thúc **4/4 PASS**; mỗi trường hợp **4096 × 900 bước** với masked auto-reset sau mỗi step. Base random 44.842s, RF random 115.513s, RF boundary-push 98.621s, RF command-churn 102.317s. Trong cả bốn lần: **0 world safety-failure, 0 overflow, 0 nonfinite**, max component reward mismatch không vượt `1.91e-6`; peak sampled VRAM `2579 MiB`. Đây là valid-input sampled policy-family stress, không phải formal proof. **Kết luận freeze vẫn FAIL / OPEN** do packet parity/DS correlation và chưa chứng minh PEER utility.

**Truy vết log nguyên bản:** `/workspace/v112_validation/output/` trên Vast và `/home/aduy/Downloads/UAV_V112_RAW_VAST_LOGS/` trên laptop người dùng (20 tệp gồm `.json`, `.log`, `.exit`, mã nguồn thực chạy). Kết quả cấu trúc tóm tắt trong `outputs/v112_validation/02_paired_summary_verified_from_vast.json` và `03_stress_summary_verified_from_vast.json`. ZIP ChatGPT của lần này chứa source và log lịch sử V111 + tóm tắt V112; kiểm tra `source_of_raw_gpu_logs` trong `00_GATE_STATUS.json` để biết nguồn thô ở laptop; không đánh đồng bản tóm tắt với raw JSON 20-world.

**Liên kết paper chính xác:** Denoeux (2008), *Conjunctive and disjunctive combination of belief functions induced by nondistinct bodies of evidence*, Artificial Intelligence, DOI `10.1016/j.artint.2007.05.008`.
