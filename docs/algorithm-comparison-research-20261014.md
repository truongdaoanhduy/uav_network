# Cơ sở nghiên cứu và thiết kế metrics so sánh thuật toán UAV

Ngày: 2026-10-14. Repo: `/home/aduy/Documents/NCKH/uav_network-cpu-flow-20261014`.
Mốc khảo sát: nhánh `pql-async-20261014`, commit `166be2233d156e43c8805eb8757d09a977189701`.
Phạm vi: đọc mã và nguồn sơ cấp; không chạy training/evaluation. Các công thức và ngưỡng triển khai dưới đây là **thiết kế của dự án**, không phải công thức được chép từ bài báo.

Kịch bản cần so sánh: 6 UAV, bản đồ 1000 × 1000 m, 25 mục tiêu, 15 vật cản, 1000 bước với dt = 1 s; tìm mục tiêu rồi chuyển báo cáo tới GCS, có relay, ngân sách năng lượng và kiểm soát an toàn. Phải dùng resolved config của checkpoint. `configs/default.yaml` ở mốc khảo sát vẫn có 10 mục tiêu và max_steps = 800.

## 1. Bảy nguồn sơ cấp phù hợp nhất

| Nguồn | Nội dung đã kiểm tra trực tiếp | Phần áp dụng và giới hạn |
| --- | --- | --- |
| Le, Chatzinotas, Vu, **Cooperative UAVs for Remote Data Collection under Limited Communications: An Asynchronous Multiagent Learning Framework**, 2026, accepted IEEE TWC. [Bài và toàn văn](https://arxiv.org/html/2601.10849v1) | Mục III: khám phá nhu cầu dữ liệu chưa biết, trao đổi trong phạm vi liên lạc giới hạn; mục VII-B: tỷ lệ dữ liệu thu được, thời gian hoàn thành, hiệu quả năng lượng; cùng 1000 tình huống kiểm thử. | Rất gần bài toán cộng tác, hạn chế liên lạc và ngân sách năng lượng. Asynchronous ở đây là quyết định hành động UAV trong semi-MDP; không đồng nhất với bất đồng bộ collector/learner của PQL. |
| Wang et al., **Cooperative Data Collection with Multiple UAVs for Information Freshness in the Internet of Things**, 2023. [Toàn văn](https://arxiv.org/html/2303.01381v1) | Abstract và mục III: Dec-POMDP, centralized learning/decentralized execution, ràng buộc động học/năng lượng/tránh va chạm. Mục III-D đo AoI ở UAV nhận dữ liệu. | Hỗ trợ đánh giá nhiệm vụ cùng năng lượng và an toàn. AoI của bài đo tại UAV; không được gọi độ trễ GCS của repo là metric giống hệt. |
| Chen et al., **Age–energy-aware trajectory planning for UAV-assisted data collection in Internet of Things**, IET Communications, 2023. [Bài gốc](https://doi.org/10.1049/cmu2.12603) | Abstract, mục 2.1: UAV thu rồi chuyển gói tới base station; thông tin chỉ cập nhật ở phía đích khi base station nhận; nghiên cứu trade-off độ mới và năng lượng. | Hỗ trợ phân biệt “đã phát hiện/thu” với “đích đã nhận”. Repo hiện báo cáo mỗi mục tiêu một lần, khác cập nhật trạng thái lặp lại của AoI. |
| Zeng, Xu, Zhang, **Energy Minimization for Wireless Communication with Rotary-Wing UAV**, IEEE TWC, 2019; preprint 2018. [Bản tác giả](https://arxiv.org/abs/1804.02238) | Abstract: tổng năng lượng UAV gồm propulsion và communication, thỏa yêu cầu throughput; cùng tối ưu quỹ đạo, phân bổ thời gian truyền và thời gian hoàn thành. | Cần cộng năng lượng bay và RF UAV khi so sánh. Không suy ra mô hình năng lượng đơn giản của repo đã được xác thực vật lý theo bài. |
| Henderson et al., **Deep Reinforcement Learning that Matters**, AAAI 2018. [Bản tác giả](https://arxiv.org/abs/1709.06560) | Abstract: biến thiên phương pháp/môi trường làm khó tái lập và diễn giải; cần báo cáo thí nghiệm và mức ý nghĩa rõ ràng. | Một checkpoint tốt chưa đủ để kết luận thuật toán tốt hơn; lưu đầy đủ thiết lập, seeds, phiên bản và kết quả từng run. |
| Agarwal et al., **Deep Reinforcement Learning at the Edge of the Statistical Precipice**, NeurIPS 2021. [Toàn văn](https://arxiv.org/html/2108.13264v4) | Mục 4.1: stratified bootstrap CI; điểm đơn lẻ bỏ qua bất định. Appendix A.5 và hình A.18 cảnh báo bootstrap của mean trên một task với ít run có thể thiếu coverage. | Dùng CI và phân phối kết quả. Không coi nhiều episode của một checkpoint là nhiều training run độc lập; không áp dụng IQM chuẩn hóa Atari một cách máy móc cho một mission. |
| Patterson et al., **Empirical Design in Reinforcement Learning**, JMLR 2024. [Toàn văn](https://arxiv.org/html/2304.01315v2) | Mục 2.1: ngân sách interaction thay vì chỉ số episode; mục 4.4: so sánh paired differences, tách RNG agent/environment; mục 7: kiểm soát seeds và không loại run kém. | Cùng seed tình huống, cùng ngân sách, CI cho chênh lệch; công bố cả training seeds và evaluation seeds. Tuning và chọn checkpoint là một phần protocol phải ghi trước. |

## 2. Metrics tối thiểu: định nghĩa của dự án

Ký hiệu: K = số mục tiêu; U = số UAV; T = số bước episode; H = max_steps × dt; F = số mục tiêu riêng biệt ít nhất một UAV biết; D = số mục tiêu riêng biệt GCS đã nhận đầy đủ. Với benchmark này K = 25, U = 6, H = 1000 s. Không tính discovery lặp lại hoặc relay như một mục tiêu mới.

| Metric | Định nghĩa và mẫu số | Chiều mong muốn | Khả dụng tại mốc khảo sát |
| --- | --- | --- | --- |
| Discovery | 100 × F / K | tăng | `search_rate_percent` đã có |
| GCS delivery / coverage | 100 × D / K | tăng; metric nhiệm vụ chính | `delivery_rate_percent` đã có |
| Delivery given discovery | 100 × D / F; null nếu F = 0 | tăng; xem cùng F/K | Suy ra từ info, đặt tên rõ; đây là tỷ lệ **mục tiêu**, không phải PDR của gói |
| Mission success | 1 nếu D = K trước/khi deadline; success rate = tổng success / tổng episode | tăng | `success` đã có; chưa có điều kiện thành công bổ sung về an toàn |
| Restricted mission completion | T × dt nếu success, ngược lại H; trung bình qua mọi episode | giảm | Suy ra ngay; không dùng episode kết thúc sớm do hết pin như “hoàn thành nhanh” |
| Delivered-report delay | Trung bình (GCS nhận − created_s) của báo cáo mang lại first delivery; null nếu D = 0 | giảm, luôn báo cùng D/K và D | Có mean thành công; trường cũ trả 0 nếu không có delivery nên cần sửa khi tổng hợp |
| UAV energy | E = propulsion_energy_j + uav_communication_energy_j | giảm với mức delivery tương đương | Có; không cộng năng lượng GCS vào pin UAV |
| Energy per delivered target | Tổng E qua tất cả episode / tổng D; null nếu tổng D = 0; tính cả năng lượng episode thất bại | giảm, báo kèm delivery/success | Suy ra; không thay D = 0 bằng 1, không loại episode D = 0 |
| Packet PDR | 100 × tổng received_packets / tổng generated_packets; null nếu mẫu số = 0 | tăng; metric mạng phụ | Có counts; là gói fragment theo hop tới UAV hoặc GCS, không phải logical report tới GCS |
| RF collision | số sự kiện collision mạng; có thể báo / tổng PHY attempts nếu định nghĩa attempts được xác nhận | giảm | `network.collisions` là collision sóng vô tuyến |
| Shield/block rate | shield_interventions hoặc blocked_moves / (U × T); tổng counts / tổng opportunities khi gộp episode | giảm nhưng phụ thuộc nhiệm vụ | Có; là hành động bị can thiệp/chặn, không phải va chạm vật lý đã xảy ra |
| Report losses | expired_reports, dropped_reports, regenerated_reports và buffer_reports cuối episode | giảm loss; diễn giải theo mission | Có; là sự kiện báo cáo, có thể nhiều thế hệ cho cùng mục tiêu |
| Return | Tổng team reward không chiết khấu | tăng; metric phụ thuộc reward weights | Có; không thay delivery/safety bằng reward |

Không đổi nhãn packet PDR thành report PDR. Nếu cần **report-instance PDR thật**, thêm counters identity `(target_id, source_id, created_s)`: mẫu số là số report instances được tạo và chấp nhận vào buffer; tử số là số identity đến GCS đầy đủ trước deadline, gồm các instance trùng mục tiêu. Buffer admission failure báo riêng. Forward/fragment/retry không tăng mẫu số report. Hiện D chỉ đếm mục tiêu lần đầu tới GCS nên chưa đủ tính metric này.

## 3. Độ trễ gồm cả thất bại và rediscovery

Mean `network.report_e2e_delay_s` hiện chỉ xét báo cáo đến thành công, và bắt đầu từ thời điểm tạo **thế hệ báo cáo thắng**, nên không gồm thời gian đã chờ từ lần phát hiện đầu tiên nếu TTL hết rồi tạo lại. Một policy chỉ giao rất ít báo cáo nhanh có thể có mean thấp.

Đề xuất lưu hai timestamp theo mục tiêu: `first_discovered_s[k]` và `first_gcs_received_s[k]`. Mục tiêu chưa có event dùng null khi xuất JSON, sentinel -1 trong state. Camera discovery đầu tiên ghi một lần; peer relay không ghi discovery mới; GCS first delivery ghi một lần với packet simulator time thật.

Metric bao phủ mọi mục tiêu: `restricted_target_delivery_time_s = mean_k(b_k)`, trong đó b_k là first GCS receive time tính từ mission start nếu giao thành công, ngược lại H. Mẫu số luôn K, nhỏ hơn tốt hơn; không phát hiện, TTL loss, hết pin và hết deadline đều không bị loại. Có thể diễn giải là diện tích phần mục tiêu chưa đến GCS trong [0,H], chia K.

Metric riêng cho chặng báo cáo: trung bình `b_k − a_k` cho các mục tiêu đã phát hiện; nếu chưa giao dùng `H − a_k`. Báo cùng số discovered, số delivered và số chưa giao. Đây là restricted cost được định nghĩa cho benchmark, **không gọi là ước lượng mean latency không kiểm duyệt**. Hết pin là terminal failure của mission, không phải mất theo dõi ngẫu nhiên để bỏ khỏi mẫu hoặc tự động dùng Kaplan–Meier.

## 4. Observability hiện tại và vị trí sửa nhỏ nhất

| Nơi | Có sẵn / vấn đề | Sửa nhỏ nhất |
| --- | --- | --- |
| `uav_network/experiments/evaluation.py` | Deterministic policy, reset seed = eval_seed + episode; lưu raw info; chỉ có 4 mean aggregates. | Tính các metrics đã có và CI; xuất explicit seed bank, resolved mission settings, checkpoint identity/backend/revision, counts và null cho mẫu số không xác định. |
| `uav_network/experiments/diagnostics.py` | Rolling return/search/delivery/success; warning delivery gap/report loss. Subprocess evaluation giữ RNG learner và cùng eval seeds. | Thêm key tổng hợp mới vào eval reporting, giữ training rolling metrics riêng khỏi evaluation. |
| `uav_network/experiments/tracking.py` | Đã cộng total UAV energy khi log episode; final_metrics chỉ gộp 4 metrics. | Dùng một hàm tổng hợp chung với evaluation để không lệch định nghĩa. |
| `uav_network/communication/reports.py` | `known`, `delivered`, report identity/created_s; successful delay/hops lists, không có target event timestamps. | Khởi tạo 2 arrays [K]; ghi first discovery trong `discover`, first GCS timestamp trong `transfer`; report-instance counters chỉ nếu cần. |
| `uav_network/environment/env.py` | info đã có F,D, energy, loss và motion counters. | Xuất timestamps/metrics nhiệm vụ mới; không đổi reward, observation hoặc termination để chỉ thêm đo lường. |
| `uav_network/environment/gpu/state.py` | Có known/delivered, scalar delay sum/count; không có per-target timestamps. | Khai báo và allocate 2 arrays [num_envs,K] float64; host copy chỉ khi episode kết thúc. |
| `uav_network/communication/gpu/mac.py`, `.../gpu/reports.py` | Reset state; camera sensing; transfer cập nhật first delivery và delay sums. | Reset timestamp sentinel trong reset kernel; ghi discovery/receive trong các hàm hiện có. |
| `uav_network/communication/gpu/metrics.py` | world_info/network_info đổi selected GPU world thành CPU info tương tự native. | Xuất timestamps và cùng semantics/null như CPU. |
| `uav_network/environment/motion.py`, `.../gpu/motion.py` | Swept geometry chặn obstacle/peer; counts ghi rejected commands. | Nếu cần measured safety: kiểm tra **quỹ đạo thực hiện** sau funding/shield, xuất obstacle, boundary, pair violations; gộp tỷ lệ episode có bất kỳ violation. Không lấy blocked hoặc RF collision thay cho safety violation. |

Safety thresholds phải lấy từ resolved config (`obstacle_clearance_m`, `safety_distance_m`, map boundary). Với pair events, đếm mỗi cặp không thứ tự một lần mỗi bước; không gọi count/p(UAV-step) là xác suất. Nếu chưa instrument thực tế, ghi metric unavailable; không mặc định zero.

## 5. Protocol so sánh được đề xuất

Đây là protocol dự án rút ra từ các nguyên tắc của nguồn ở mục 1, không phải một protocol nguyên văn trong một bài:

1. Khóa kịch bản, reward, shield, network/energy settings, backend và upstream revision. So sánh thuật toán trên cùng backend; native và Warp là một thí nghiệm fidelity riêng, cần kiểm chứng trước khi gộp.
2. Khóa ngân sách environment transitions/global_step và quy tắc chọn checkpoint. Báo thêm wall-clock, updates và sample reuse vì collector/learner bất đồng bộ có thể đạt khác số updates trong cùng thời gian. Không kết luận sample efficiency chỉ từ episode hoặc giờ training.
3. Lập seed manifest: training runs, validation bank để chọn checkpoint, final-test bank riêng. `seed + episode` hiện tiện nhưng không bảo đảm disjoint: training dài có thể chạm eval_seed = 10044. Kiểm tra set intersection trước khi chạy.
4. Final evaluation dùng deterministic actions và cùng danh sách tình huống cho mọi thuật toán. CPU evaluation dùng đúng backend lưu trong checkpoint. Cùng seed không có nghĩa stream mọi network event vẫn giống nhau nếu policies tạo khác sự kiện.
5. Mỗi training seed tạo một policy độc lập, đánh giá policy đó trên toàn test bank. Lưu ma trận metric [training run, scenario seed]; số scenario không thay cho số training runs. Một training run chỉ cho kết luận về checkpoint đó.
6. Với hai checkpoint cố định, bootstrap **cùng indices scenario** ở hai phía và CI cho mean paired difference. Với nhiều training runs, resample runs và scenario bank, giữ scenario pairing; chỉ pair training runs nếu có thiết kế pairing thực sự. Không flatten toàn bộ [run,episode] như các mẫu độc lập.
7. Công bố số runs/seeds, mean ± 95% CI và paired differences của delivery/success/restricted completion, cùng energy và safety. Chọn metric chính trước; CI từ quá ít training runs là thăm dò. Giữ failures và nguyên nhân dừng; không bỏ run xấu hoặc chọn best seed trên test bank.

Ưu tiên triển khai ngay: các summary từ info đã có + paired comparison trên seed bank. Per-target event timestamps là bổ sung nhỏ có giá trị nhất để đo tiến độ nhiệm vụ và thời gian tới GCS không bỏ thất bại. Không cần đổi reward hoặc chạy training để hoàn thành phần instrumentation này.

## 6. Triển khai sau khảo sát

Phương án triển khai cuối dùng observer trong CPU evaluation, giữ per-target
**first observed** discovery/delivery timestamps ở boundary mỗi step. Độ phân
giải là dt; không thay simulator kernels/state/reward/observation hoặc gọi đây
là timestamp GCS packet chính xác. Observer còn đo swept actual trajectories
với thresholds của config và tolerance1mm. Capped target delivery/AUC vẫn tính
cả failures, chưa phát hiện và rediscovery waits theo first observed discovery.

CLI `compare` dùng30 common scenes mặc định tại seed2000000044, JSON/CSV và
paired scenario bootstrap. Protocol kiểm tra training range (kể cả seeds trong
record history), current validation bank, và ghi exploratory cho lịch sử resume
chưa đầy đủ. Một seed training44 vẫn không được diễn giải như30 training runs.
Xem [hướng dẫn và phạm vi đo lường](training-flow-comparison-20261014.md).
