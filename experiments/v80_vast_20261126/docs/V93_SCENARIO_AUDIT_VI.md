# Kiểm toán kịch bản UAV V93 — 27/11/2026

Trạng thái: **CANDIDATE_NOT_FROZEN / NO_GO_BASELINE / NO_TUNING**.

Mục đích là xác minh kịch bản và sửa lỗi đã tái hiện. Không có training RL hoặc tuning. Toàn bộ 30 file Python trong uav_network/algorithms giữ nguyên byte so với gói V92. Các kết quả đầy đủ và log ở gói evidence đi kèm; các số lịch sử không được xem là lượt chạy lại.

## 1. Nguồn và phạm vi

- Parent V91: f0433d51b5de1cf90b5b265221c5e9ad606f47b9; branch riêng experiment/v93-scenario-audit-20261127.
- ZIP V92 đầu vào SHA256: fd55d8ad352ad7c30f05a10c90f1f25842ece4c47c34ccc4647c09db92aeffd9. Hai bản người dùng gửi có cùng byte.
- So với V91 trên Sandbox: 331 file giống hệt; V92 thêm hai probe, một test và đổi manifest.
- YAML candidate: research/zhang_xue/configs/scenario_v91_bayes_1km.yaml.
- Máy Vast được chọn lại trong phiên: RTX3060 12GiB, logical cuda:0, driver570.195.03. Máy RTX3060Ti8GiB lúc đầu không phải máy dùng cho kết quả cuối.
- Runtime CUDA kiểm chứng: Python3.12, torch2.7.1+cu128, Warp1.18.0+cu12, numpy1.26.4.
- Baseline MATD3/MASAC và main được giữ nguyên. Không tạo tag SCENARIO_FROZEN.

## 2. Kịch bản thực sự có gì?

| Thành phần | Nội dung source hiện hành | Đánh giá |
|---|---|---|
| Hình học | 1000×1000m, 6 UAV+GCS, 15 target tĩnh, 392 ô target hợp lệ | Giữ; không quay về map3000m/35target |
| Chuyển động | XY, cao50m, dt1s, horizon900, vmax15, amax5, safe50 | Giữ các giới hạn |
| Sensor | FoV tròn50m; Pd.85/Pf.03; có cylinder occlusion3D | Mô hình cảm biến giả định; không phải cảm biến đã hiệu chuẩn thực địa |
| Evidence | Count uint16 riêng theo origin và generation; dữ liệu tĩnh không decay theo AoI | Đúng hướng chống cộng trùng; vẫn cần công bố giả thiết độc lập |
| Bayes | Log odds, prior15/392, threshold.90 | Posterior theo thông tin từng ô; không phải posterior liên hợp có điều kiện mọi quan sát/đúng tổng15 |
| DS | mf=me=.8, threshold.90, uncertainty ceiling.20 | Giữ đối chứng lịch sử; mass không đồng nghĩa xác suất đã calibrated |
| Radio | TX cố định.1W, SNR6dB, CCA−82dBm, noise−93.576dBm, loss12dB/building | Research surrogate; không chứng minh tương đương UavNetSim hay đầy đủ IEEE802.11 |
| Packet | CSMA/CA, HELLO DV, MAC ACK/E2E ACK, retry, queue, TTL, dirty ledger, store-carry-forward | Có thực thi packet, không chỉ range graph |
| Observation | Local549 + mask3; wrapper553; joint state3318 | Actor không đọc target truth; peer kinematics qua HELLO |
| Safety | Candidate disk+coordinate descent, fail-closed | Không CBF; không phải bộ giải tối ưu liên hợp hoặc bảo đảm recursive feasibility đầy đủ |

Action chính xác: continuous (E,6,2), discrete (E,6) với 0=NO_SEND, 1=GCS destination, 2=PEER dissemination. **Không có learned TX-power action cần xóa trong bản này.** GCS destination có thể đi qua relay; PEER không phải công tắc bật/tắt mọi multihop.

## 3. Các lỗi đã tái hiện và sửa

### MAC: cùng khe nhưng mất va chạm

V92 xử lý radio tuần tự; radio đầu phát rồi đóng băng radio còn lại dù cả hai hết backoff cùng lúc. RED cho [ST_TX,ST_IDLE] thay vì hai TX; cả luồng step_v67 và step_v67_masked đều có lỗi.

V93 xử lý DIFS trước, chọn toàn bộ contender từ cùng trạng thái kênh, rồi mới bắt đầu TX. ready_mask ngăn việc đóng băng contender cùng khe. PHY vẫn cộng interference và half-duplex; không biến collision thành success. Test cũng giữ trường hợp counter muộn hơn bị freeze đúng.

Bằng chứng: 02/02b RED, 04 GREEN, 09 CUDA focused, 16 microbench 32 seed. Microprofile cho clear/hidden/forced-tie; CPU và CUDA khớp bộ đếm sự kiện. Đây không phải so sánh với upstream SimPy.

### Probe PEER: đối chứng cũ không kiểm tra điều nó tuyên bố

V92 probe thực tế cho GCS TP 1→1 và peer_relay_unlocks_gcs_confirmation=false, nhưng vẫn exit0. Với GCS-addressed routing, source có thể chuyển packet qua relay dù PEER disabled.

V93 thay bằng hai contact tách rời: contact A source→relay, không ai đến GCS; contact B relay→GCS, source không đến relay/GCS. Chỉ enable_peer thay đổi, data-mule giữ true, cùng sensing/actions. Có source-origin evidence ở relay trước contact B và ở GCS sau đó, TP0→1. Chuyển vị trí relay là can thiệp fixture mạng, không phải chứng cứ bay đúng động học.

### Báo cáo thành công giả và metrics sau terminal

- V92 hướng dẫn --seeds nhưng script chỉ có --seed; V93 nhận cả --seed/--seeds và --output/--out rõ ràng.
- Kết thúc chương trình không được đồng nghĩa gate PASS. Probe mạng trả exit2 nếu điều kiện causal fail; mission probe có PROBE_VALID/PROBE_INVALID.
- V92 tiếp tục step các world đã done trong lúc chờ world khác; metrics cuối có thể bị thay đổi sau terminal. V93 lưu snapshot lần đầu done riêng cho từng world, trước bước tiếp theo/reset.
- Traces NPZ giữ counts/truth của evaluator, trạng thái, action và hashes để tái tính. Truth không đi vào controller.

### Action không hợp lệ

RED xác nhận NaN có thể làm hỏng trạng thái vật lý; Inf bị clamp âm thầm; discrete=3 được nhận như action hợp lệ. V93 kiểm tra trên device, đánh dấu safety bit16 và fail-closed trước commit. Không sửa loss/actor/critic.

### Trạng thái không thể tránh va chạm ở bước kế tiếp

Trace seed44/world7: ở step143, fallback chỉ an toàn trong một bước đưa UAV3/5 vào trạng thái mà đến cuối bước144, kể cả hai UAV đổi vận tốc tối đa5m/s mỗi chiếc, khoảng cách lớn nhất chỉ **44.993321m <50m**.

Chứng minh riêng: tại trạng thái sau commit, r là relative position, v là relative velocity, dt=1. Mọi endpoint kế tiếp có dạng r+v+δ với ||δ||≤10; bởi bất đẳng thức tam giác, khoảng cách không vượt ||r+v||+10=44.993321. Đây là chứng chỉ bất khả thi, không phải lỗi làm tròn.

V93 kiểm tra trước commit bound ở trạng thái dự kiến: ||r+2*v_new*dt||+(a_i+a_j)*dt². Nếu bound lạc quan vẫn <50m, từ chối transition bằng bit32 và giữ vị trí/vận tốc trước đó. Tolerance1mm cộng vào bound chỉ tránh false rejection do FP32; không giảm safe_d50. RED ca step143 và GREEN đã được lưu.

**Giới hạn còn lại:** guard này là điều kiện cần, không phải lời giải phối hợp nhiều UAV hay bảo đảm recursive feasibility. Các world vẫn có thể bị từ chối và kết thúc; không gọi đó là “đã hết vấn đề safety”.

## 4. Những kết luận V92 cần sửa

| Kết luận V92 | Thẩm định V93 |
|---|---|
| CW0..31 nghĩa là toàn Warp đúng IEEE802.11 | Chỉ convention miền backoff phù hợp; còn slot timing, retries, CCA, capture, ACK... Không suy ra compliance toàn bộ |
| Upstream CW0..30 là lỗi cần bắt Warp làm theo | Delta source có thật; giữ0..31. Exact upstream equivalence vẫn không đạt |
| “Calibrated surrogate” sau khi đọc Sargent | Sargent mô tả quy trình validation, không cấp chứng nhận calibration. Chỉ gọi research surrogate |
| GCS action tức là direct RF | Sai: đó là endpoint ứng dụng, đường đi có thể nhiều hop |
| Số ô mất GCS = số ô được relay cứu | Chưa đủ: cần topology relay khả thi, lịch contact, packet/ACK và nhiệm vụ; link-budget graph chỉ là upper bound hình học |
| Noise+12dB tương đương map2km; +18dB tương đương3km | Sai nếu pathloss exponent2: hệ số khoảng cách lần lượt3.981 và7.943; gấp2/gấp3 là6.021/9.542dB trong mô hình LoS đơn giản |
| Cần Bayes thắng DS mới freeze | Sai về phương pháp: correctness không phụ thuộc phương pháp yêu thích thắng |
| Hai command CUDA đủ khóa | Không: còn integrity của metrics, collision, safety, scope network, reward confound |
| ACE được quy trực tiếp cho Naeini2015/Guo2017 | Hai nguồn liên quan calibration/ECE; adaptive metrics cần dẫn Nixon2019 và định nghĩa triển khai chính xác |

Giữ range CW không phải là che một parity failure; parity với upstream và tính hợp lý của convention là hai câu hỏi khác nhau.

## 5. Belief, reward và đối chứng

V93 thêm Brier, fixed-bin ECE và adaptive positive-class ECE trên **cùng counts, cùng truth, chỉ ô hợp lệ đã thăm**. Tách tập team original evidence và GCS delivered evidence. Báo kích thước tập, prevalence, số visit và bins. Không coi các cell cùng world là mẫu iid để tuyên bố significance.

Với DS, báo BetP(T)=m(T)+m(U)/2 cho probability scoring, đồng thời giữ score m(T) dưới nhãn diagnostic. Đây là lựa chọn scoring được công bố, không phải biến DS thành Bayes hay sửa production inference.

**Reward confound chưa biến mất trong YAML:** ví dụ một positive đầu tiên, DS m(U)=.2 còn Bayes coverage=.5. Với w_loc=.5, bonus từ unseen là .4 so với .25. Kênh Bayes là mức đã quan sát, không phải Shannon entropy hoặc DS mass. Vì vậy các mission runs DS/Bayes không phải ablation chỉ thay estimator. Bảng paired-count calibration không dùng tổng return để kết luận backend tốt hơn.

Reward kernel vẫn gồm: exploration, GCS true-confirm/freshness, FP-ever penalty, FP-active hold, safety failure, intervention, airtime và mapwide AoI term. lambda_b=0 và continuous AoI penalty mặc định0 làm các term tương ứng bằng0; đây là lựa chọn mục tiêu, không tự động là bug. Flight/RF vẫn tiêu hao pin. Không thêm penalty “cho đẹp” trước khi chốt nhiệm vụ.

## 6. RF và lý do chưa học TX power

Tính lại từ phy_v67.py với loss12dB/building:

| Noise tăng | LoS slant range | Ô direct/all400 | Ô direct/legal392 |
|---|---:|---:|---:|
| 0dB | 2377.871m | 256 | 254 |
| +6dB | 1191.758m | 239 | 239 |
| +12dB | 597.294m | 120 | 120 |
| +18dB | 299.356m | 41 | 41 |

Đó là từng link không có interference, không phải throughput/PDR end-to-end. TX .1W không vô nghĩa như tham số PHY; learned TX action là câu hỏi khác và hiện disabled. Giữ nguồn RF cố định để baseline có action space rõ ràng.

Loss12dB/cylinder, blackout và noise là mô hình danh định. Những paper LoS xác suất hoặc fading không chứng minh vật liệu cụ thể có loss12dB. Cần sensitivity/empirical data trước tuyên bố calibrated cho đô thị thực.

## 7. Warp song song và baseline sau này

Một thread/world sở hữu event loop radio của world đó; PHY chia row để tránh write race. Cùng seed, Warp CPU/CUDA matching là bằng chứng computational verification theo workload. Nó không thay thế independent model validation.

E4096 world network SoA đã tính đúng **598629 byte/world =2.283588GiB**, chưa gồm world buffers/context. Profile GPU cuối đi cùng log và memory samples; đó là scenario rollout, không có replay/backward/optimizer.

Observation/state dimension3318 làm replay hiện tại rất lớn: TensorReplayBuffer FP32 lưu cả obs/state và next counterparts khoảng53192byte/row, x1e6≈49.54GiB. Bản state-derived FP16 là cấu trúc khác; không được lấy footprint của nó gán cho runner dùng TensorReplayBuffer. Chưa chạy hay tuning baseline để “fit” VRAM.

Installer Warp cũ lấy wheel CUDA13 từ PyPI và thất bại trên driver570. V93 chọn wheel1.18+cu12 chính thức trên máy này, giữ nguyên driver. Các version cài thực tế có JSON provenance. Python13/CPU Sandbox không được trộn với CUDA12/Python12 khi nói bitwise full mission reproducibility.

## 8. Điều còn phải chốt trước freeze

1. **Safety phối hợp:** bit32 chặn một lớp trạng thái bất khả thi nhưng heuristic vẫn có terminal rejections. Thiết kế/kiểm định recovery/planner phối hợp nếu tiêu chí nhiệm vụ đòi không bị reject. Không xóa flags hay nới50/15/5 để làm gate xanh.
2. **Network independent validation:** 32 seeds ở V93 là Warp CPU↔CUDA, không phải upstream UavNetSim SimPy↔Warp. DV/HELLO/payload/RNG hiện khác. Phải xác định common microprofile hoặc chấp nhận explicit research-surrogate scope; chưa có macro equivalence.
3. **Reward-matched inference ablation:** chỉ bắt buộc khi muốn tuyên bố thay riêng DS↔Bayes cải thiện RL. Cần reward/communication/observability protocol được giữ tương đương, hoặc công bố khác biệt.
4. **RF validity:** nếu mục tiêu là môi trường danh định mô phỏng, công bố giả thiết và sensitivity; nếu tuyên bố phản ánh đô thị thực, cần calibration data. Không dùng chữ calibrated khi chưa có.
5. Chỉ sau khi quyết định/phép đo tương ứng hoàn tất mới tạo tag SCENARIO_FROZEN rồi chạy baseline continuous nguyên bản. Không có bằng chứng full training/convergence trong gói này.

## 9. Nguồn chính và mức sử dụng

- Zhang et al., IEEE T-IV2024, DOI10.1109/TIV.2024.3352581: đã đọc PDF người dùng. Fixed-altitude2D, belief/evidence, communication đơn giản và DCDDPG riêng. mf/me lấy theo sensor history; .8 là cấu hình thí nghiệm, không hằng số bắt buộc mọi kịch bản.
- UavNetSim-v1: https://arxiv.org/abs/2507.09852 và pinned GitHub c2703f24eb58cdb557a5f67ded5c7b0368a964ed. Đã đọc paper/source MAC/PHY; dùng làm reference, không nhận source Warp là nguyên bản upstream.
- Bianchi2000, IEEE JSAC, https://www.eng.buffalo.edu/~tmelodia/papers/bianchi.pdf: DCF/slotted contention, mô hình saturation với assumptions; không áp dụng nguyên xi để xác nhận traffic đa chặng mission.
- Malone/Duffy/Leith2007, DOI10.1109/TNET.2006.890136: non-saturated traffic assumptions; cần xác minh matching workload trước đối chiếu throughput.
- Sargent2010, https://www.informs-sim.org/wsc10papers/016.pdf: conceptual/operational/data validation, không thay test bằng nhãn.
- Guo2017, https://proceedings.mlr.press/v70/guo17a.html; Nixon2019, https://arxiv.org/abs/1904.01685: calibration và ảnh hưởng binning. Metric V93 là binary positive-class adaptive ECE, không tự nhận full multiclass ACE.
- Smets/Kennes, pignistic transformation: https://iridia.ulb.ac.be/~psmets/TBM-InformSystem.pdf. Probability scoring và belief mass là hai cách biểu diễn.
- Zeng/Xu/Zhang, IEEE TWC2019, DOI10.1109/TWC.2019.2902559, https://arxiv.org/abs/1804.02238: propulsion steady-speed; việc bỏ acceleration energy là giả thiết xấp xỉ cần công bố.
- NVIDIA official Warp1.18 release và installation/compatibility docs: https://github.com/NVIDIA/warp/releases/tag/v1.18.0.
- Danh hiệu Q1/A* không tự chứng minh một hệ số hay kịch bản cụ thể đúng. Gói này ghi venue/DOI; không khẳng định đã kiểm tra mọi quartile/ranking năm2026.

## 10. Đọc evidence

- Các log RED chứng minh lỗi trước sửa; GREEN và full regression chứng minh phạm vi đã sửa.
- JSON/NPZ mission ghi kết quả đúng tại first terminal. PROBE_INVALID không bị đổi nhãn PASS.
- Kết quả network microprofile và causal PEER có scope riêng; không cộng chúng thành “mọi gate upstream đã xong”.
- source_v93 là source có thể chạy; patches có diff; history chứa evidence V90/V91/V92 cùng digest và bảng runs lịch sử.
- File reports/VALIDATION_RESULTS.json ở ZIP là thống kê cuối; không suy diễn số test từ tiêu đề phiên bản.
