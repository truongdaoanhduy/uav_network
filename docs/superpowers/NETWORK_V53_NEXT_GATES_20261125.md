# Phán quyết giải pháp v5.2 và việc cần làm tiếp

“Đúng hướng”, “đã triển khai” và “đã xác nhận end-to-end” là ba mức bằng chứng khác nhau. Bảng này xếp lỗi theo ảnh hưởng đến kết luận và đường chạy, không kế thừa nhãn P0 một cách máy móc từ ZIP cũ.

| Mã | Tuyên bố/giải pháp | Phán quyết | Hành động |
|---|---|---|---|
| C01 | Pending binding được coi là đã sửa loader | Chưa thực hiện chức năng. | Đã thêm loader cho CPU 6 radio; YAML 7 nút vẫn bị từ chối rõ ràng. |
| C02 | 5 building/2 blackout là baseline đã chốt | Chưa có căn cứ biến candidate thành baseline của repo. | Giữ scenario ID và bảng khác biệt; khóa manifest khi xây scene mới. |
| C03 | H1500 chỉ thay con số horizon | Đây là đổi objective nếu deadline trở thành safety cap. | Đã thêm mode pin riêng; giữ mặc định finite horizon. |
| B01 | DS mới chính xác tuyệt đối | Sai số học ở counter lớn hoặc conflict gần 1. | Đã sửa log scaling; tổng conflict hoàn toàn báo lỗi. |
| B02 | MAX_PLUS_SUM là Zhang Eq.16 | Sai attribution: Eq.16 dùng pure MAX. | Giữ hai ablation; mô tả cơ chế per-origin như mở rộng. |
| B03 | Provenance bảo đảm sensor độc lập | Chống trùng event không loại tương quan giữa sensor. | Công bố giả định; kiểm tra correlated noise hoặc fusion thận trọng. |
| B04 | Pd/Pf tự xác định mass 0,8 | Thiếu ánh xạ calibration. | Khai báo riêng Pd/Pf và mass; đánh giá calibration nếu thay đổi. |
| B05 | FoV đủ để xác nhận visibility 3D | FoV không bao gồm camera occlusion. | Tách che camera, che RF và va chạm. |
| A01 | AoI dùng generation time receiver đã nhận | Đúng hướng. | Giữ timestamp bất biến qua relay; chỉ cập nhật sau DATA receipt. |
| A02 | AoI 0 s chứng minh thông tin luôn mới | Sai mẫu số: chỉ 11 lần xác nhận đầu. | Đã đổi tên metric và thêm receiver time-cell mean 51,23 s. |
| A03 | Timestamp giây đủ đo delay MAC | Không đủ phân giải subsecond delay. | Dùng event timestamp cho metric; công bố lượng tử wire timestamp. |
| R01 | Full-Comm thắng Zero-Comm chứng minh reward đúng | Chỉ sanity check của một script. | Thêm thử nghiệm duplicate, cycle, FP, delayed evidence và policy. |
| R02 | Union local reward giải quyết delivery utility | Chỉ giảm đếm lặp local. | GCS TP/delivery/success phải chờ receiver thật nhận. |
| R03 | RF nhỏ nên bắt buộc có airtime penalty | Đây là quyết định objective. | Công bố đơn vị, trọng số và sensitivity; tách backoff khỏi TX energy. |
| R04 | fp_hold loại hết reward exploit | Chưa chứng minh; latch hiện giữ vĩnh viễn. | Test FP đang active, sửa sai, stale FP và repeated confirmation. |
| N01 | 0,5 W và Friis dynamic range | Đúng phép tính theo mô hình chọn. | Giữ một nguồn tham số; phân biệt noise-limited range và SINR thực. |
| N02 | NLoS 26,95 m thuộc classic v1 | Sai phiên bản: urban model đến từ bản vendor mới. | Khóa pin và gọi đúng urban adapter; classic benchmark riêng. |
| N03 | Episode 7 nút thực hiện CSMA/CA | Sai: BFS oracle và copy ledger. | Đã đổi nhãn; packet scheduler vẫn cần xây. |
| N04 | Đổi mọi số 6 thành 7 là thêm GCS | Sai vai trò actor/origin/radio. | Giữ 6 actor, 6 sensor origin; thêm radio/recipient GCS riêng. |
| N05 | MAC ACK đồng nghĩa GCS nhận evidence mới | Sai hai ranh giới. | Tách DATA receipt, APP ACK và novelty gate. |
| N06 | Đổi YAML là có native TX ablation | Mã cũ âm thầm chạy 0,5 W. | Đã fail closed; arbitrary power cần cấu hình native cô lập. |
| N07 | 58 B header chứng minh IEEE 802.11 hoàn chỉnh | Chỉ đúng quy ước của simulator. | Giữ sizing nhất quán và công bố phần MAC/PHY được trừu tượng hóa. |
| Q01 | Q64 vượt worst case 60 | Chưa chứng minh arrival/control/relay bounds. | Đo occupancy và drop; ablation Q32/64/128/200. |
| Q02 | Cùng 8 B nên packing đã tương thích repo | Schema và parser hiện khác. | Version wire format, roundtrip CPU/GPU và giữ immutable payload. |
| T01 | Thay clock bằng pin bảo đảm Markov | SOC không chứng minh toàn bộ local state Markov. | Đã có mode SOC; công bố Dec-POMDP và thông tin triển khai. |
| T02 | Cạn pin terminal, safety cap truncated | Đúng khi task được định nghĩa như vậy. | Đã nối replay mask; giữ final observation và ưu tiên terminal thật. |
| T03 | 1173 s là pin vật lý đã xác nhận | Chỉ phép tính theo quỹ đạo và tham số giả định. | Đã sửa per-UAV energy; reference đổi 1171 bước, chưa hiệu chuẩn hardware. |
| G01 | H+TTL vừa int32 nên scheduler an toàn | Đúng một bound; chưa xét INF, ACK lifetime và reset. | Giữ int64 queue; guard arithmetic và test biên nếu tối ưu int32. |
| G02 | 0,586 GiB/GPU là VRAM mạng | Incomplete analytical estimate. | Đã thêm lower bound của bốn mảng; chưa đo peak CUDA. |
| G03 | FP16 replay 500k–1M chắc fit T4 | Phụ thuộc dimensions, layout, model và minibatches. | Đã thêm byte estimator theo layout thực; profile mỗi GPU. |
| G04 | Có kernel Warp là có full GPU env | Các wrapper và call path chưa hợp nhất. | Preallocate, device views, stream ordering và full-step audit. |
| G05 | Warp CPU E4096 xác nhận T4 throughput | Không chứng minh CUDA hoặc VRAM. | Chỉ ghi kết quả CPU; GPU gate còn thiếu. |
| S01 | Thêm DS/AoI/simulator là novelty đủ mới | Các bài gần đã có nhiều cơ chế tương tự. | Định vị task và feedback qua mạng cụ thể; benchmark công bằng. |

## Thứ tự triển khai tiếp và tiêu chí nghiệm thu

### 1. Chốt một scenario contract có thể thực thi

Profile 15 target, z=50 m, v_max=25 m/s, H=1500 s, 5 buildings và 2 blackout là candidate mới, khác baseline cũ. Cần khai báo GCS dùng nguồn điện riêng, team dừng khi UAV đầu tiên hay tất cả UAV cạn pin, khả năng quan sát của actor và cách xác nhận mục tiêu. YAML phải thực sự cấp tham số cho scene, PHY/MAC, reward và energy; lưu cấu hình đã resolve và hash mỗi run. Đổi một tham số phải làm đổi hành vi tương ứng. Khóa lạ, khóa trùng và backend không hỗ trợ phải báo lỗi.

Tách camera occlusion, RF obstruction và collision. Kiểm tra tâm ô, chiều cao tòa nhà, đường bay và đường truyền. Scene không đặt target trong vùng không thể quan sát, trừ khi nhiệm vụ chủ đích cho phép. Initial UAV/GCS phải hợp lệ. Tỷ lệ diện tích FoV/cell không phải số cell được đo thật.

### 2. Xây CPU reference có sự kiện thực cho 7 radio và GCS

GCS là node 6, không phải actor hay sensor; vẫn cần thực hiện ACK/routing/control theo protocol đã công bố. Dữ liệu giữ origin, epoch, sequence, generation time và counter version xuyên mọi hop. Receiver chỉ cập nhật sau DATA receipt. Sender chọn dirty evidence từ bộ nhớ và ACK/digest thực nhận; không được đọc ledger bên nhận.

Nghiệm thu: một hop, hai hop và sáu hop; hidden terminals; interference bắt đầu giữa DATA; mất ACK; half duplex; queue full; hết TTL đúng biên; no route; route expiry và loop; packet còn sống qua env step; blackout recovery; store–carry–forward; DATA đã nhận nhưng MAC ACK mất; APP ACK trùng; completion thuộc episode cũ. Mỗi trường hợp cần packet trace và kết quả mong đợi độc lập.

Giữ benchmark classic đúng pin và urban extension riêng. Event parity đòi cùng RNG và quy ước sự kiện; khi RNG khác, so phân bố PDR/delay/AoI trên cùng scenario và nhiều seed. Không đòi bitwise giống nhau giữa các engine không cùng semantics.

### 3. Hợp nhất scheduler Warp

Một `ZhangXueWarpEnv` phải có reset/step cùng timeline cho motion, sensing, evidence, HELLO/DV, CCA/backoff, SINR, DATA/ACK, retry, forwarding, TTL, GCS fusion/AoI, reward, energy và final observation. Tách rõ 6 actor, 6 sensor origin và 7 radio. Timer/queue tồn tại qua env step; event được xử lý đúng thứ tự và dùng vị trí nội suy theo mô hình công bố.

Preallocate buffers, cache Torch views, hạn chế host copy/sync. Mỗi tensor có chủ sở hữu; atomic/barrier không làm fusion đếm trùng. Bounded event loop phải báo quá tải thay vì bỏ sự kiện. Partial reset không chạm world khác; RNG độc lập theo world/episode/agent/event. Dùng công thức DS ổn định và kiểm thử wire/parser/packing cùng CPU.

### 4. Đối chứng và T4 preflight

Sau khi CPU reference và scheduler hợp nhất đúng mới dùng account 2 để đo CUDA. Chạy E256 → E1024 → E2048 → E4096. Tách bộ nhớ network, actor, replay, sampled batches trên mỗi card. Báo allocated/reserved/device free/peak, throughput sau warm-up, events/world-step, CPU load, host copy/sync và phiên bản phần mềm.

CUDA Graph chỉ bật sau parity và warm-up. Nếu quá RAM hoặc chậm, xuất báo cáo có căn cứ; không bỏ PHY/MAC hoặc thu nhỏ queue âm thầm để đạt một nhãn PASS.

### 5. Training/evaluation có ngân sách và fairness

MATD3/MASAC dùng matched seeds và joint-transition budgets, eval seeds riêng. Actor không nhận oracle; cấu hình observation/action/reward/threshold được khóa. Không nạp checkpoint khác môi trường. Log TD error, actor/critic gradients, pre-tanh/saturation, invalid motion, masks, queue/drop/TTL/retry, receiver AoI, GCS TP/FP, coverage/AUC, energy và capped delivery time.

Ưu tiên ablation: Ideal Sync/Packet; Direct/Multi-hop; No Relay; No Store–Carry–Forward; LoS/urban; blackout; TX 0,1/0,5 W trên backend thực hỗ trợ; Q=32/64/128/200; pure MAX/per-origin SUM; AoI weight; airtime weight. Deadline/battery là hai task khác nhau và phải ghi rõ. Nhiều seed và khoảng tin cậy cần thiết cho kết luận về policy; một seed diagnostic chưa đủ.

## Trạng thái gate sau bản sửa

| Gate | Trạng thái |
|---|---|
| Kiểm định gói v5.2 | Hoàn tất trong phạm vi source/PDF/log đã liệt kê; không chứng nhận toàn bộ 61 issue cũ khi thiếu bảng gốc |
| CPU pilot: pin/config/replay | Đã triển khai, có test chạy thật |
| DS/pin/AoI tham chiếu | Đã sửa, 21 tests; vẫn là diagnostic đồ thị |
| Scientific contract 7 radio | Chưa hoàn tất |
| Full CPU packet network + GCS | Chưa đạt |
| Unified Warp 7 radio | Chưa đạt |
| CUDA parity/T4 E4096 fit | Chưa chạy/chưa đạt |
| MARL 7 radio convergence/eval | Chưa chạy |
