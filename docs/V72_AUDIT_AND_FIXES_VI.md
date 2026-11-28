# V72 — Kiểm định lại V71, sửa lỗi tái hiện và giới hạn bằng chứng

Ngày 26/11/2026, giờ Việt Nam. Mã đầu vào là ZIP V71 đính kèm; bản bàn giao V70 và đoạn nhận xét là giả thuyết cần kiểm tra, không thay thế source. ZIP gốc được giữ nguyên. Các thay đổi thực hiện trên bản sao, có Git baseline cục bộ; không push GitHub.

## 1. Kết luận và cách đọc

V71 sửa đúng nhiều điểm, nhưng chưa đủ để gọi môi trường huấn luyện Warp hoàn chỉnh hoặc tương đương UavNetSim. V72 sửa các lỗi tái hiện được trong môi trường, reward, MAC, relay và Warp; bổ sung kiểm thử chạy kernel thật trên backend CPU. Đây không phải bằng chứng CUDA/T4 hay kết quả học hội tụ.

- Baseline V71 chạy lại: **242 pytest PASS, 64 subtest PASS**; khác số trong báo cáo cũ vì lần này có PyTorch CPU và Warp 1.18.0.
- QA cuối: **311 pytest PASS, 64 subtest PASS; 13 gate PASS / 2 CUDA-readiness gate BLOCKED / 0 FAIL**. Có một deprecation warning từ plugin anyio của runtime, không phải lỗi source.
- Kết quả cuối và exit code từng gate: `outputs/qa_evidence_v72/results.json`, `SUMMARY.md`, `pytest_junit.xml`.
- Bằng chứng lỗi trước/sau sửa: `outputs/qa_evidence_v72/red_green/`.
- Nguồn tham khảo, phương án thay đổi khoa học, xung đột config và lộ trình GPU: `V72_RESEARCH_AND_GPU_PLAN_VI.md`.
- `V72_BUG_STATUS.json` phân biệt **FIXED_CPU**, **FIXED_WARP_CPU**, **OPEN**, **BLOCKED**, và thay đổi có chủ ý.

## 2. Các thành phần thực sự có trong source

| Thành phần | Đang thực hiện | Giới hạn hiện tại |
|---|---|---|
| Map/kịch bản | 1 km, 20×20 ô 50 m; 15 target; 6 UAV + 1 GCS; 5 building, 2 blackout | 2/3 km là stress diện tích, không cùng độ khó |
| Chuyển động | Velocity XY, altitude cố định 50 m, v≤15 m/s, a≤5 m/s², dt=1 s | Không phải policy bay 3D; shield không phải chứng minh an toàn phần cứng |
| Sensor | FoV tròn r=50 m; Pd=.85/Pf=.03; có kiểm tra che khuất tia 3D | Pd/Pf là mô hình độc lập lý tưởng; không camera thực/không uncertainty calibration thực nghiệm |
| Belief | Local ledger theo origin; mT,mF,mU; UQ8.8 counts; decay .02/s; xác nhận .90 và mU≤.20 | Count aging khác mass discount; fixed BBA .8 không phải posterior Bayes đã hiệu chuẩn |
| Action | 2 motor liên tục + compact3: SILENT/GCS/PEER; TX chính cố định .1 W | Learned RF và per_peer là biến thể riêng |
| Reward | Local information gain; TP/FP tại GCS; first-confirm freshness; FP hold; shield/blackout/RF penalties | Không tự động tối ưu PDR; phần AoI chính không phải time-average AoI |
| CPU network | Packet, PHY/SINR, CSMA/CA, MAC ACK, E2E ACK, TTL, HELLO/path-vector, store-carry-forward | Simulator tự viết lấy cảm hứng UavNetSim; không chạy nguyên upstream |
| Warp network | SoA, PHY, sensor-entry injection, queue, MAC/routing/ACK events, reset | Chưa có full GPU motion/sensing/belief/reward/obs/replay/update/eval |
| MATD3/MASAC | Learner, hybrid actions, replay, target networks; CPU real-transition smoke | H32×32/B4 trong gate; chưa có cấu hình paper/trainer vector Warp hoàn chỉnh |
| Telemetry | Các hàm thống kê/plot và QA | Có file utility không đồng nghĩa live W&B/HF đã nối vào trainer này |

Environment trả 514 chiều/UAV: 14 telemetry + 5×10×10 pooled maps. Adapter CPU mở rộng thành518 chiều cho learner; critic state là nối các actor observations đã mở rộng (3108 chiều ở6UAV). GT target chỉ ở simulator/reward/audit plot, không đưa vào actor. Shield vẫn giả định biết vị trí đội và hình học cản trở; phải công bố giả định này khi gọi hệ thống decentralized.

## 3. Kiểm chứng các tuyên bố sửa của V71

| Tuyên bố V71 | Phán quyết sau kiểm tra | V72 thực hiện |
|---|---|---|
| Giữ tx_node trước khi xóa RX lock | Đúng; các kernel biên dịch và chạy được trên Warp CPU | Giữ; không suy thành CUDA PASS |
| CCA khác ngưỡng decode SINR; rx_lock_on_cca=False | Đúng ở mức thiết kế hiện tại | Giữ; carrier sensing và decode được kiểm tra riêng |
| Pending E2E không bị ghi đè; dirty bỏ đích đang pending | Đúng với một outstanding request/đích | Giữ; không gọi là cùng semantics CPU nhiều request |
| Sửa event overflow bằng completed | Chưa đủ: event cuối vừa đủ budget vẫn bị báo tràn | Sửa điều kiện hoàn tất, sticky device fault và test đủ/thiếu budget |
| Tắt host guard để tránh D2H | Chưa đủ: không được phép dùng transition đã overflow | Device cũng chặn state lỗi; bỏ host read không đồng nghĩa bỏ correctness guard |
| Reset/input validation đã đủ | Chưa đủ: grid1 bị mất chiều; bool/fraction dimensions và eta chưa chặt | Sửa rank theo tên scalar; validate dimensions/eta; reset và invalid-generation tests |
| Guard VRAM đã an toàn | Chưa đủ: ImportError torch có thể bỏ qua guard | Query free VRAM qua chính Warp trước allocation |
| Smoke có counters giao gói | Hữu ích nhưng trước đây chưa chạy kernel | Một workload dùng chung CPU/CUDA, assert counters **và payload/generation** |
| Giữ gói chưa có route, retry/backoff, half-duplex, HELLO revoke | Các hướng sửa đúng | Thêm sửa stale HELLO expiry, ACK overhearer, arm queue và relay retry dedup |
| CCA aggregate | Đúng: tổng tín hiệu thay vì từng TX | Không xác nhận câu “+noise”: cả hai implementation hiện so tổng tín hiệu với CCA |
| Queue coalescing và origin fairness | Chưa đủ: renew TTL sai lifecycle; own-origin luôn ưu tiên; preview làm đổi cursor | Sửa lifetime/pending, getter thuần đọc, rotate cả own/relay khi enqueue thành công |
| GCS clock, phạt FP một lần, PA fixed power, braking gần building | Những sửa này hợp lý và giữ lại | Bổ sung timestamp receipt của evidence novel cho first-confirm freshness |
| Decentralized compact3 | Chưa đủ: option mặc định False; Warp vẫn đọc peer truth | CPU và Warp mặc định HELLO-local; legacy oracle phải bật rõ ràng |
| Clamp r_loc_team, thêm continuous AoI | Clamp là lựa chọn shaping; thiếu aging team và dùng running mean làm current penalty | Sửa aging cùng clock; option AoI dùng current age; giữ cumulative metric riêng |
| Decoder hỗ trợ N UAV | Chưa đủ: low-level RF decoder đã N nhưng full bridge còn hardcode6 | Full learned-RF bridge dùng N và mask width từ environment |
| “Tương đương Xue/CPU, chạy tốt trên T4” | Chưa được chứng minh | Thu hẹp claim; counterexample toán và open gates bên dưới |

## 4. Lỗi xác nhận đã sửa trong V72

| ID | Mức | Lỗi và hậu quả trước sửa | Cách sửa / test hành vi |
|---|---|---|---|
| E01 | P0 | Peer action/mask đổi khi chỉ di chuyển hoặc làm cạn pin peer ẩn | Mặc định local HELLO; cùng local info thì action/mask giữ nguyên |
| E02 | P1 | Đọc observation làm thay đổi thứ tự payload | Mask dùng dirty count; preview không advance scheduler |
| E03 | P1 | Hai positive cũ ở t1 vẫn được cộng như mới tại t101 | Age từng origin trước team fusion; .353 mass không vượt .9 |
| E04 | P1 | Map500 m spawn khoảng35.8 m dù safe50; fleet30 có spawn trong building | Giữ spawn chuẩn hợp lệ; fallback lattice có kiểm tra và fail rõ nếu không đủ chỗ |
| E05 | P1 | Learned RF N4/compact3 bị bridge đòi N6/D7 | Decode dùng N và D thực; test full policy→env |
| E06 | P1 | Option continuous AoI ở t2 dùng1.5 thay vì2 s | Phạt current mapwide age; metric time-average vẫn riêng |
| E07 | P1 | Evidence hỗ trợ cũ đến muộn làm xác nhận nhưng latency báo0 | Theo dõi latest novel RX riêng; repro0→9 s |
| N01 | P0 | Packet hết TTL trong DIFS vẫn phát | Check TTL tại thời điểm bắt đầu phát |
| N02 | P0 | Packet hết hạn khi đang bay vẫn được giao tại đích | MAC ACK vật lý vẫn có; application không nhận evidence hết TTL |
| N03 | P1 | Relay đầy queue ghi dedup trước khi giữ được gói | Commit dedup chỉ sau enqueue thành công; retry có thể nhận custody |
| N04 | P1 | Coalesce gia hạn identity nhưng pending đã hết hạn, ACK không cập nhật watermark | Giữ deadline cũ; prune expired, tạo identity mới; khôi phục pending live nhất quán |
| N05 | P1 | 62 own entries khiến relay origin đói vô hạn | Rotate cả own và relay khi enqueue mới thành công; preview/coalesce không ăn lượt |
| W01 | P1 | Constructor grid1 tạo gcs_latest rank1, struct yêu cầu rank2 | Chỉ collapse field scalar theo danh sách tên |
| W02 | P0 | Đúng event cuối vẫn báo overflow; bỏ host guard có thể bỏ lỗi | Recheck sự kiện còn trong window; device sticky fault chặn step sau |
| W03 | P0 | ACK kết thúc chỉ xóa RX lock bên nhận đích | Xóa tất cả overhearer locks thuộc ACK transmitter |
| W04 | P1 | HELLO quảng bá route đã expired | Prune route trước snapshot DV |
| W05 | P1 | Gói vừa inject phải đợi HELLO tiếp mới tranh kênh | Arm idle work tại đầu step |
| W06 | P1 | ACK mất làm relay enqueue cùng hop packet nhiều lần; retry drop không có counter | Hop-ID dedup sau admission; tăng drop counter khi cạn retry |
| W07 | P0 | Warp cuối đường nhận evidence hết TTL | Chặn application/forwarding, giữ physical ACK |
| W08 | P1 | Gửi lại evidence cũ đổi last_rx như evidence mới | Chỉ đổi freshness receipt khi evidence novel |
| W09 | P0 | Warp peer resolver đọc tọa độ/alive ẩn | Mặc định route-distance từ local HELLO; legacy nearest explicit |
| W10 | P1 | Raw-count mode dùng gen-only watermark nhưng vẫn được API nhận | Từ chối mode chưa hỗ trợ; dimensions/eta và generation fail closed |
| W11 | P1 | Có Warp/CUDA không có torch thì memory guard bị bypass | Dùng Warp-native free_memory; không cấp phát khi vượt budget |
| Q01 | P1 | Scanner bỏ sót prefix token Kaggle | Bổ sung pattern, test synthetic; không sử dụng token thật trong test |
| Q02 | P2 | Chỉ test source strings/counters làm quá mạnh kết luận | Gate CPU chạy kernel thật + payload; paired endpoint fixture không gọi event parity |
| Q04 | P2 | Test 500 dòng quét cả thư viện .venv sau cài đặt | Dùng inventory source chuẩn; không bỏ sót file dự án |

Chi tiết raw RED/GREEN và test file tương ứng nằm trong issue register. Các thay đổi mặc định có ý nghĩa thí nghiệm: chọn peer local, age team reward đúng, TTL/custody/route behavior. Cần train lại và phân biệt version, không so return với V71 như cùng simulator.

## 5. Những điểm chưa được giải quyết bằng việc pytest PASS

| Mức | Trạng thái | Việc còn cần |
|---|---|---|
| P0 | OPEN | Full Warp environment + learner rollout/update/eval chưa được tích hợp |
| P0 | OPEN | Chưa đối chiếu event trace/hidden terminals/collision/retry timing CPU↔Warp |
| P0 | BLOCKED | Không có CUDA driver trong môi trường QA; token account_02 đã thử nhưng Kaggle trả inactive |
| P1 | OPEN | Chưa đo PDR/delay/retry distributions với upstream SimPy/UavNetSim |
| P1 | OPEN | CPU/Warp khác queue, pending concurrency, RNG, dedup window, origin/cell selection và RF/MAC configurability |
| P1 | OPEN | BBA/threshold chưa calibration; cần precision/recall/false-confirm, multi-seed và matched ablation |
| P1 | OPEN | Shield dùng perfect global state; mô hình pin/PA/RX-idle power chưa hiệu chuẩn thiết bị thực |
| P2 | OPEN | Chưa có full training, hội tụ, confidence intervals, baseline communication hoặc live W&B/HF mới |

Ring hop dedup mới giữ16 ID; môi trường tải cao có thể evict ID trước retry. Đây là giới hạn hữu hạn cần stress test, không phải dedup vô hạn. UQ8.8 cũng gây lượng tử hóa; byte equality một fixture không chứng minh tương đương toàn simulator.

## 6. Metric phải gọi đúng tên

- `precision/recall/f1` hiện dựa mission TP latch và active FP latch; không đồng nhất confusion matrix của posterior tại thời điểm hiện tại. `false_confirm_rate` dùng ever-FP trên empty cells. Công bố định nghĩa cùng figure.
- `mean_first_tp_confirm_latency_s` đo freshness của generation mới nhất tại latest novel RX trước lần xác nhận đầu ở cuối bước. Không phải timestamp chính xác DS vượt ngưỡng bên trong micro-events, cũng không phải time từ bắt đầu tìm target tới delivery.
- `current_gcs_aoi_mapwide_censored_s` là age mapwide tức thời, unknown cells dùng elapsed mission time; `mean_gcs_aoi_mapwide_censored_s` là trung bình lấy mẫu theo thời gian. Đây là convention có công bố.
- Coverage/discovery/delivery/AUC là metric nhiệm vụ; PDR/queue/ACK/retry/latency là metric mạng. Không thay thế lẫn nhau.

## 7. Tái lập theo từng bước

1. Cài `bash setup.sh --install`, thêm Warp bằng `bash setup.sh --gpu` nếu có CUDA phù hợp; pin bản kernel1.18.0 của gói này.
2. Chạy `python scripts/maintenance/verify_source_manifests.py --check` và policy full.
3. `python -m pytest -q tests/ research/zhang_xue/tests/`.
4. `python scripts/gpu/warp_packet_cuda_gate.py --device cpu --envs 4` và `python scripts/gpu/packet_contract_gate.py --device cpu`.
5. `python scripts/maintenance/run_qa_bundle.py --output-dir outputs/new_qa --seed 44` để ghi log mới, giữ log release.
6. Trên Kaggle dùng `scripts/gpu/run_kaggle_v72_gates.py`; chạy real CUDA 4env/GPU trước. Không vượt gate network rồi gọi là RL training.

Source hash và cấu hình effective được lưu trong manifests/effective_configs. Logs lịch sử V70/V71 vẫn nằm trong `outputs/qa_evidence/`; chỉ `outputs/qa_evidence_v72/` là QA của lần này.
