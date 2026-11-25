# Kiểm định độc lập v5.2 và bản sửa v5.3

Ngày 25/11/2026. Source kiểm tra: `truongdaoanhduy/uav_network`, commit nền `f664e7e570c951cd069d12faa9c3be0cd913d930`. Nhánh thực hiện: `experiment/zhang-xue-v53-audited-battery-contract-20261125`. Commit cuối, lệnh chạy và kết quả máy nằm trong `evidence/verification.json`.

**Kết luận: chưa thể chấp nhận tuyên bố “xử lý toàn bộ 20 P0 / Gate 0 hoàn chỉnh”.** Gói v5.2 có các phép tính hữu ích, nhưng còn lỗi DS số học, sai cách tính pin, AoI bị đặt tên gây hiểu nhầm và mô phỏng đồ thị được trình bày quá mức thành mạng gói tin. Code repo đang chạy vẫn là CPU pilot sáu radio; các kernel Warp Zhang–Xue là các component đã kiểm thử riêng. Chưa có đường gọi thống nhất `6 actor → 7 radio → GCS thật nhận DATA → fusion/AoI/reward → replay` trên CUDA.

Đã triển khai các sửa đổi có căn cứ và kiểm thử vào nhánh riêng. Phần chưa xây xong được ghi thành điều kiện chặn, không đổi tên một diagnostic thành production. Không khởi chạy training dài hoặc tiêu quota Kaggle account 2 trong lần kiểm định này.

## Đọc gói này

| File/thư mục | Nội dung |
|---|---|
| `01_DECISIONS_AND_NEXT_GATES.md` | Phán quyết từng giải pháp và thứ tự triển khai tiếp |
| `02_PAPER_AND_PRIMARY_SOURCES.md` | Đối chiếu hai PDF toàn văn, công thức, phiên bản và URL sơ cấp |
| `03_PRODUCTION_NETWORK_AUDIT.md` | Đường gọi thật, vị trí hardcode sáu nút, pin UavNetSim và lỗi TX |
| `04_V52_COUNTEREXAMPLES.md` | Phản ví dụ độc lập, số liệu trước sửa và giới hạn test cũ |
| `repo_patch/` | Diff và các file đã sửa trong repo; không phải một repository thay thế |
| `original_v52/` | Nội dung gói người dùng đưa vào, giữ nguyên để tái lập |
| `corrected_reference/` | DS/pin/AoI đã sửa, vẫn ghi rõ là chẩn đoán đồ thị biết toàn cục |
| `evidence/` | Log RED/GREEN, phản ví dụ, memory budget, xác minh và hash |

## Kịch bản đang có trong code

| Thành phần | Repo nền v4.11 đã thực hiện | v5.2 đề xuất | Xung đột/ảnh hưởng |
|---|---|---|---|
| Bản đồ | 1 km, 20×20 ô, ô 50 m | Cùng kích thước | Phần này nhất quán |
| Mục tiêu | 10 target tĩnh được sinh ngoài NFZ | 15 | Khác scenario; không trộn checkpoint/metric |
| Actor/radio | 6 actor, 6 radio, action discrete 6 | 6 actor, 7 radio, action discrete 7 | Thêm radio GCS không đồng nghĩa thêm actor |
| Độ cao/FoV | z=100 m, FoV90°, tâm ô trong hình tròn bán kính100 m | z=50 m, bán kính50 m | Diện tích quan sát giảm khoảng4 lần, số ô thực phụ thuộc vị trí |
| Vật cản | 2 đĩa NFZ, ngăn đường bay | 5 trụ cao70–80 m | NFZ không tự động là che RF/camera 3D |
| Blackout | 2 vùng Manhattan được sinh theo seed | 2 vùng cố định | Phân bố scene và độ khó khác nhau |
| Tốc độ/gia tốc | v_max50 m/s, reject chuyển động không an toàn; không giới hạn gia tốc | v_max25 m/s, a_max5 m/s² trên YAML | Episode tham chiếu không áp dụng a_max; đo được27,41 m/s² |
| DS | Hàm log ổn định; counter riêng theo origin; MAX hoặc MAX_PLUS_SUM | Hàm đóng “canonical” mới | Hàm mới bị lỗi chuẩn hóa; không thay ngược vào hàm repo đang đúng |
| Xác nhận | threshold0,95; team-own pooled belief | threshold0,90; GCS latch | Cả ngưỡng và thực thể xác nhận thay đổi |
| Pin | Chưa có trong CPU pilot nền | 180 kJ, Zeng P(v)+avionics+RF | Cần mode riêng và trừ theo từng UAV |
| Kết thúc | Nhiệm vụ finite horizon200 bước, có time trong obs | H1500 chỉ là safety cap, pin/task terminal | Đây là đổi MDP, không chỉ đổi tên `done` |
| GCS/AoI | Chưa có GCS trong env hiện tại | GCS ID6, AoI từ generation time | Chưa được nối vào đường chạy thật |
| PHY/MAC | Mini reference hoặc opt-in native classic SimPy | LoS/NLoS3D + CSMA/CA7radio | Không được đồng nhất hai backend |
| Warp | Có sensor/motion/queue/routing/payload/ACK kernels và test | Unified device-resident env | Chưa có class/collector hợp nhất đã chứng minh |
| Learner | CPU pilot dùng MATD3/MASAC thực | E4096, nhiều GPU | Kết quả component không xác nhận fit VRAM hay hội tụ |

## Xác minh và giới hạn bộ test

- Bộ test Zhang–Xue trước sửa: **145 passed**.
- Bộ hồi quy Zhang–Xue sau sửa: **163 passed**, thêm 47 subtests passed; dùng đúng native classic checkout, không có skip trong lượt xác minh này.
- Bản tham chiếu hiệu chỉnh: **21 passed** cả tại workspace và sau khi đặt vào repo thật; chín test mới đều đã thất bại trên mã gốc trước khi sửa.
- Review độc lập không phát hiện lỗi critical/important trong patch đã giao.
- Lượt `pytest -q` toàn repo **không xanh và chưa hoàn tất**: dừng ở 490,55 s trong chuỗi test training CPU cũ, báo 70 failed, 1363 passed, 99 skipped, 2077 subtests passed. Đây là tổng tại thời điểm dừng, không phải kết quả toàn bộ suite. Danh sách tên test và đối chiếu commit nền ở `evidence/full_suite_status.json` và `evidence/full_suite_analysis.md`. Không gọi toàn bộ failures là lỗi mới hay đã sửa nếu chưa có đối chứng.

## Các lỗi và nhận định quan trọng nhất

1. **P0 — DS v5.2 sai số học ở counter dài.** Với `(18,18)`, tổng mass chỉ `0,524288`; `(30,20)` cho target mass `0,01048576` thay vì `0,9999998976`; `(1500,1500)` cho `(0,0,0)`. Nguyên nhân là `1-K` bị triệt tiêu rồi bị ép xuống `1e-12`. Bản sửa dùng tổng các giao còn sống và công thức log đã scale. Tổng xung đột hoàn toàn phải báo là không xác định, không trả ba số0. Giả định độc lập của evidence vẫn là một giả định khoa học cần công bố.
2. **P0 — Diagnostic7 nút chưa có packet MAC.** Code dựng adjacency toàn cục, BFS từ GCS và chuyển ledger trực tiếp. Nó đọc ledger bên nhận để biết ô nào dirty. Không có traffic HELLO, tranh chấp, mất ACK, backoff, retry, queue hay TTL thực trong episode. Đặt queue/TTL bằng0 vẫn cho kết quả cũ. `769,97 > 139,08` là kết quả của kịch bản đồ thị và reward đã chọn, không chứng minh policy học relay hoặc mạng đã đúng.
3. **P0 — GCS chưa có trong flow production.** CPU dùng team-own belief để thưởng và kết thúc, không chờ DATA tới GCS. Routing/path mask/envelope/queue/payload Warp vẫn có ràng buộc6. Nếu chỉ đổi `uav_count=7`, UAV thứ7 trở thành actor có sensing/motion, sai vai trò GCS. Phải tách `actor_n=6`, `sensor_origin_n=6`, `radio_n=7` xuyên mọi module.
4. **P1 — AoI0 s bị diễn giải quá rộng.** Báo cáo cũ chỉ lấy11 lần xác nhận TP đầu tiên. Trace cùng episode có4.833 lần entry đến GCS,807 lần mang thông tin tuổi>0, tuổi delivery lớn nhất86 s. Trung bình theo thời gian và các ô đã nhận là51,23 s. Bản sửa đổi tên metric và ghi rõ mẫu số. Các ô chưa từng nhận vẫn cần thêm metric/cap riêng khi thiết kế production.
5. **P1 — Pin tính sai chuyển động và người phát.** Có17 UAV-step đứng yên và53 UAV-step dưới15 m/s, nhưng mã cũ luôn dùng15 m/s; mọi pin bằng nhau dù số lần phát khác nhau. Bản tham chiếu đã dùng tốc độ dịch chuyển thực, DATA TX của sender và ACK TX của receiver; ACK của GCS không trừ pin UAV. Cạn pin của quỹ đạo này đổi1173→1171 bước. Đây chưa phải hiệu chuẩn thời lượng bay phần cứng.
6. **P1 — Native backend âm thầm đổi công suất.** Đã tái hiện yêu cầu0,02 W nhưng thực chạy0,5 W. Caller không truyền config xuống adapter, adapter dùng `utils.config` toàn cục. Đã sửa caller và từ chối công suất chưa hỗ trợ trước khi nạp native. Không giả vờ đã hỗ trợ ablation0,1 W native theo instance: việc đó cần cô lập tiến trình hoặc cơ chế cấu hình không làm đổi mạng đang sống.
7. **P1 — Link budget đúng số nhưng trộn nguồn.** Khoảng557 m/26,95 m là đúng với công thức và tham số của candidate. Tuy nhiên classic pin`64bc3a24` không có `phy/a2a.py`; urban alpha3+12dB đến từ bản mới được vendor. Phải gọi đây là urban adapter được chọn, ghi commit và kiểm tra sensitivity. Khoảng noise-limited không phải range bảo đảm khi có interference.
8. **P1 — Q64 không có bảo đảm “worst case60”.** Giới hạn60 chỉ đúng nếu đã chứng minh tổng arrival≤6gói/s tại queue, TTL10 s, không có queue/control/ACK/phát lại bổ sung và quy tắc hết hạn phù hợp. Mã chưa chứng minh các điều kiện ấy. Tính queue theo offered load, forwarding, control và deadline; đo drop theo nguyên nhân cùng p95/p99/p99,9 occupancy ở các Q32/64/128/200. Không hạ Q chỉ để fit bộ nhớ.
9. **P1 — Ước lượng VRAM chưa đầy đủ.** Candidate tính payload+watermark và vài số dự trù; chưa tách receiver-origin counts và timestamps khỏi sender ACK watermarks. Với2048 world,7radio,6origin,400cell,Q64,K62, bốn mảng ấy đã là867.958.784 B≈0,80835 GiB/GPU. Chưa tính metadata, MAC/routing, scene, belief caches, mạng học, Adam, minibatch/activations, eval và context. Con số0,586 GiB không phải measured peak.
10. **P1 — Packed entry mới chưa đi trên dây.** Bản tham chiếu pack20-bit timestamp+origin/cell và hai16-bit counter là đủ độ rộng cho1500 lần lấy mẫu ở1Hz. Packet repo hiện dùng entry8B nhưng trường reserved, không chứa generation time; GPU sidecar cũng chưa đổi đồng bộ. Không được suy ra “8B như nhau nên đã tương thích”. Epoch/sequence trong envelope, thứ tự byte, version và tất cả parser phải cùng đổi.

## Các sửa đổi đã thực hiện

| Vị trí | Thay đổi thực sự | Giới hạn có chủ đích |
|---|---|---|
| `zhang_xue_coop.py` | Loader YAML flat kiểm tra khóa lạ/trùng, kiểu, finite và bounds; thêm `physical_battery` opt-in; SOC cục bộ tại obs index4; trừ năng lượng theo chuyển động đã chấp nhận; cờ terminal/truncation | Vẫn CPU6radio, team-confirmation, không GCS; mặc định finite-horizon cũ được giữ |
| `zhang_xue_battery.py` | Zeng speed-only propulsion, induced term dạng reciprocal ổn định, cấu hình có đơn vị và kiểm tra tham số | Không mô hình acceleration, climb, gió hoặc calibration phần cứng |
| Native classic adapter | Nhận yêu cầu TX từ env; reject giá trị khác0,5 W thay vì chạy sai experiment | Chưa có power override độc lập theo instance |
| `train_zhang_xue_pilot.py` | `--config`, thực sự nạp YAML; replay `dones=terminated`, reset theo terminated hoặc truncated; giữ final obs trước reset | Không đổi công thức SAC/TD3; không dùng checkpoint môi trường cũ |
| `zhang_xue_memory_budget.py` | Tính byte từ layout replay đang có; ghi lower bound của các mảng mạng đã định nghĩa và danh sách phần thiếu | Không tuyên bố tổngVRAM/throughput CUDA |
| `configs/zhang_xue_v53_cpu_battery_pilot.yaml` | Profile chạy được cho CPU6radio: z50,15target,v25,H1500,pin180kJ | Ghi rõ2NFZ+2blackout, không5building/GCS; threshold0,95 và masses0,8 là profile khai báo |
| Bản tham chiếu hiệu chỉnh | Sửa DS/pin/timestamp, đổi tên AoI, ghi `oracle_connectivity_diagnostic_no_packet_mac` | Không thay BFS bằng một MAC giả đơn giản rồi gọi là UavNetSim |

RF energy trong CPU pilot **bị từ chối nếu yêu cầu hệ số khác0** vì backend chưa xuất TX airtime từng radio đầy đủ. `info` ghi rõ thiếu phần này; không chia đều RF và gọi là pin vật lý hoàn chỉnh. Reference graph có TX durations giải tích và vẫn chỉ là diagnostic. Khi port sang production cần TX/RX/idle/ACK/HELLO/retry theo event thực, công suất bức xạ khác công suất đầu vào PA/baseband.

## Căn cứ khoa học và những phần phải diễn đạt lại

- **Zhang2024:** Eq16 là MAX; per-origin MAX rồi SUM là mở rộng, không phải reproduction Eq16. Các mass0,8 và threshold0,95 thuộc baseline bài; Pd/Pf và mass evidence không tự động đồng nghĩa. Nếu muốn calibrate theo altitude/Pd/Pf, cần định nghĩa cơ chế sinh BBA và thí nghiệm calibration riêng. Không lấy Bayesian posterior prior0 rồi cập nhật Bayes thông thường: prior0 làm posterior mắc ở0; DS vacuous uncertainty khác xác suất prior0.
- **Xue2026:** đã có DS/AoI-like freshness/blackout/buffer/age discount; actor nhận global maps. Có thể dùng làm Ideal Sync ablation khi công bố khác biệt thông tin và task. Không gọi mọi policy dùng global maps là decentralized execution có cùng information budget với packet network.
- **Pardo2018:** time-to-deadline cục bộ không tự thân là leak. Deadline thật cần state/time và terminal; cap kỹ thuật cần bootstrap. SOC không tự chứng minh local observation Markov: nhiệm vụ vẫn có mục tiêu và trạng thái mạng chưa quan sát. Bản sửa chỉ áp dụng terminal/truncation phù hợp mode đã khai báo.
- **Zeng2019:** P(v) dùng làm baseline bay bằng ổn định; không chứng minh bộ pin50Wh, a_max5 hoặc thời lượng1500s phù hợp UAV thật. Tốc độ25m/s là lựa chọn/tham khảo, không giới hạn phổ quát do bài quy định. Thêm gia tốc không làm mất hiệu lực mọi P(v), nhưng phải công bố xấp xỉ và kiểm tra turning.
- **Yates2021:** AoI=`t−generation time` của thông tin mới nhất receiver thực nhận. Giữ per-entry/per-origin freshness, không refresh bằng thời điểm relay/retransmit; packet mới chưa chắc chứa evidence mới. Cần báo tuổi cả ô chưa nhận, delivery age và coverage để tránh chọn mẫu đẹp.
- **RAI-CoS2026:** đã có provenance/CI/delayed evidence/byte scheduling và recurrent decentralized control; network traces replay là một giới hạn cụ thể, không phải bài “chỉtracking”. Hướng có thể kiểm chứng của dự án là tìm target ẩn, GCS và co-simulation traffic/motion/kênh hiện hành. Chưa có cơ sở tuyên bố “đầu tiên”.
- **UavNetSim:** mô hình packet là hữu ích nhưng không phải RF waveform hoặc chứng nhận IEEE802.11. Giữ source-compatible classic benchmark riêng với urban extension; đối chiếu một số tình huống chuẩn với ns-3 sẽ tăng độ tin cậy của port.

Không xác nhận nhãn **Q1/Q2/CORE A*** trong báo cáo này vì chưa có hồ sơ xếp hạng đúng hệ thống/năm/ngành. Hai PDF đã đọc toàn văn và các tài liệu chính thức vẫn là căn cứ trực tiếp cho công thức. Chi tiết nguồn, số trang và giới hạn truy cập WuTMC nằm trong file02; thiếu full text không phải bằng chứng bài đó không có một cơ chế.

## Bộ nhớ và chạy Warp song song

Replay thực lưu current+next observations FP16, current+next centralized state FP16, actionsFP32, reward/doneFP32, current+next masksuint8. Với6agent,D_obs121,D_state726,A_cont2,A_disc6, mỗi joint transition là **6080 B**: capacity1M là6.080.000.000 B≈5,66244 GiB, capacity500k≈2,83122 GiB. State concat đã lặp lại thông tin observation nhưng vẫn đang được cấp phát riêng; muốn bỏ lặp phải đổi layout/interface, không tự trừ khỏi phép tính.

E4096 chia2048/2048 chỉ áp dụng nếu scheduler thật sự chia environments và storage giữa hai T4. VRAM hai card không tự cộng thành một card32GB. Nếu một card làm rollout, một card giữ replay/backward thì phải lập ngân sách từng card theo đúng vai trò.

Rủi ro và tiêu chí tiếp theo: host `.numpy()`/loop E gây đồng bộ và copy; cấp phát buffer mỗi step làm tăng peak; event bounded-loop có thể bỏ sự kiện khi quá tải; race GCS fusion/queue ownership; world RNG/episode-reset sai; timestamp overflow khi cộng sentinel; mask7 và shape6 bị lệch; CUDA stream của PyTorch/Warp chưa phối hợp; phi tuyến reward hoặc bỏ FP32 master weights khi tùy tiệnFP16. Tất cả phải test trên đường gọi thống nhất trước khi đo tốc độ. CUDA Graph có hỗ trợ một số allocation qua mempool ở phiên bản mới; không khẳng định mọi allocation đều cấm. Profile repo đang pin Warp1.18.0, nên kiểm tra đúng API/version và warm-up kernel thực.

`H+TTL=1,51e9us` nằm trongint32 là phép tính đúng, nhưng chưa kiểm chứng mọi cộng/trừ INF2e9, APP_ACK lifetime, event stale và reset đồng thời. Queue hiện dùngint64; không hạ xuốngint32 chỉ vì horizon vừa. Nếu cầnint32 dùng sentinel guarded arithmetic và test ranh giới. Trường timestamp20bit hiện có độ phân giải1s; không dùng nó để tuyên bố delay microsecond bằng0.

## Cách dùng code đã sửa

Từ repo đã áp dụng patch, sử dụng môi trường Python có dependency dự án:

```bash
PYTHONPATH=. python -m pytest -q tests/test_zhang_xue_v53_*.py
PYTHONPATH=. python -m uav_network.experiments.zhang_xue_memory_budget --envs-per-gpu 2048 --capacity 500000 --obs-dim 121 --state-dim 726 --disc-dim 6
```

Lệnh CPU pilot khi cần kiểm tra plumbing (không phải thí nghiệm mạng7 nút):

```bash
PYTHONPATH=. python -m uav_network.experiments.train_zhang_xue_pilot --algorithm matd3 --config configs/zhang_xue_v53_cpu_battery_pilot.yaml --episodes 1 --eval-episodes 1 --learning-starts 2000
```

Profile này được nạp thật qua loader. YAML v5.2 nested7 nút bị reject rõ ràng khi đưa vào CPU6radio, thay vì bị bỏ qua trường hoặc chạy bằng defaults. Chạy mô hình tham chiếu từ thư mục `corrected_reference` bằng `python -m unittest discover -s tests -v`.

**Việc tiếp theo là map/scene contract → network7radio/GCS → scheduler Warp hợp nhất → parity và profile T4 → training.** Danh sách nghiệm thu cụ thể ở file01. Không cần train hàng nghìn episode để chứng minh các lỗi config, DS, timestamp hoặc số radio.
