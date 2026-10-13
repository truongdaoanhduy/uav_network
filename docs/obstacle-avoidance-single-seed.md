# Tránh vật cản nhưng vẫn quan sát target sát obstacle

Ngày kiểm tra: 13/10/2026. Repository đang dùng:
`/home/aduy/Documents/NCKH/uav_network`. Project cũ chỉ được đọc tham khảo.

## Nguyên nhân bị chặn

Trong bản cũ, `move_uavs` kiểm tra toàn bộ đoạn bay của một bước với
vật cản đã cộng `obstacle_clearance_m`, rồi kiểm tra quỹ đạo tương đối
của các cặp UAV. Đoạn không an toàn bị hủy hoàn toàn; vận tốc thực tế về 0.
Đây là từ chối chuyển động trước va chạm, không phải metric đếm các cú đâm.
Lệnh tiếp tục hướng vào obstacle có thể làm UAV đứng kẹt và nhận penalty.
Biên map trước đây được clip riêng, nên UAV cũng có thể kẹt ở biên.

Với MATD3 checkpoint 1.000, seed evaluate 10044: bản cũ có 2.051 obstacle
blocks và 51 peer blocks. Pin không cạn trong lượt kiểm tra này.

## Lựa chọn phương pháp

| Phương pháp | Lợi ích | Hạn chế liên quan nhiệm vụ |
|---|---|---|
| APF lực đẩy theo khoảng cách | Nhẹ, dễ triển khai | Có thể cản tiếp cận goal sát obstacle, dao động hoặc kẹt ở cực tiểu |
| CBF/QP | Tách ràng buộc an toàn và mục tiêu, thường chỉnh lệnh ít | Phải khớp động học/gia tốc và xử lý bất khả thi; riêng safety filter không đảm bảo hết kẹt |
| Bộ lọc dynamic window có phanh và hướng tiếp tuyến | Khớp động học XY đang có, không cần lực đẩy, giữ lệnh an toàn | Lựa chọn cục bộ; không chứng minh hội tụ hoặc tìm đường toàn cục |

Chọn phương án thứ ba cho scenario hiện tại có vật cản tròn, XY, vận tốc/gia
tốc hữu hạn và simulator chạy trên CPU. Đây là **bản thích nghi lấy ý tưởng
từ DWA**, không phải nguyên xi DWA của robot synchro-drive trong paper 1997.

Lo ngại target sát obstacle phù hợp với vấn đề GNRON trong nghiên cứu APF:
lực đẩy có thể lấn át mục tiêu sát vật cản. Với nhiệm vụ này, UAV chỉ cần
đến một điểm hợp lệ mà camera thấy target; không cần bay tới đúng tọa độ
target hoặc đi vào vùng clearance.

## Cơ chế đã triển khai

1. Tính vận tốc mong muốn từ action gia tốc của policy.
2. Giới hạn đồng thời tốc độ và gia tốc từng trục. Chiếu lên giao của
   acceleration box và speed disk; tránh lỗi radial speed clip làm gia tốc
   trục còn lại vượt giới hạn khi bay chéo.
3. Kiểm tra đoạn bay dự kiến **và toàn bộ đường phanh đến khi đứng yên**.
   Không chỉ nhìn một horizon ngắn rồi bỏ qua quãng đường dừng.
4. Nếu lệnh và đường phanh của cả đội an toàn, giữ nguyên lệnh.
5. Nếu không, xét các vận tốc đạt được trong một bước, ưu tiên giảm tốc và
   tiếp tuyến quanh obstacle/peer. Nếu phía đi vòng bị biên hoặc obstacle
   khác bịt, thử phía còn mở. Không dùng lực đẩy tỷ lệ nghịch khoảng cách.
6. Kiểm tra continuous swept separation cho các đường phanh peer đồng bộ.
   Reservation theo thứ tự UAV ID là xác định, có thể ưu tiên ID nhỏ trong
   khu vực đông.
7. Giữ exact guard cuối cùng để bảo vệ trạng thái ban đầu không thể phanh
   kịp hoặc khi UAV khác bất ngờ dừng do thiếu pin.

Đường phanh dùng đúng mô hình tích phân theo bước của project: cập nhật vận
tốc rồi dịch chuyển tuyến tính trong bước. Không tuyên bố bảo đảm cho động
lực học UAV ngoài mô hình này.

Bộ lọc nằm trong môi trường/safety layer, dùng geometry obstacle và vị trí
peer để kiểm tra an toàn. Đây là cùng loại thông tin hình học mà guard cũ
đã có, không phải bộ điều khiển tránh vật cản chỉ dựa trên camera cục bộ.
Actor vẫn nhận partial observation cũ, không được cấp tọa độ target khuất
hoặc target ngoài FOV. Bộ lọc không nhận target, report hay thông tin reward.

Action/observation/state dimensions và reward weights được giữ nguyên.
Replay vẫn lưu action của policy; môi trường ánh xạ action đó qua safety.
Số intervention được ghi riêng; `blocked_moves=0` không có nghĩa policy
đã điều hướng tốt hoặc không cần safety can thiệp.

## Bằng chứng cùng checkpoint, cùng seed

Policy MATD3 1.000 episode được đóng băng; không retrain trước bảng này.
Mỗi hàng là một lượt 800 bước trên **cùng map seed 10044**, không phải
mean/std của nhiều training seed.

| Điều kiện | Return | Search | Delivery | Hard blocks | Boundary clips | Success |
|---|---:|---:|---:|---:|---:|---|
| Reject cũ, mask cũ | -315,985 | 20% | 20% | 2.102 | 2.183 | Không |
| Dynamic window, mask cũ | 393,439 | 100% | 70% | 0 | 0 | Không |
| Dynamic window, mask admission đã sửa | 453,163 | 100% | 80% | 0 | 0 | Không |

Hai lần evaluate dynamic window khớp chính xác cả hash episode và hash quỹ
đạo, với từng lựa chọn mask. Cặp so sánh đầu chỉ thay cơ chế chuyển động.
Hàng cuối thay thêm admission mask; không gán toàn bộ cải thiện cho avoidance.
Không thấy vi phạm swept clearance trong các lượt này.

Ở hàng cuối:
- Cả 6 UAV còn hoạt động; pin thấp nhất 41,47%, trung bình khoảng 46,87%.
- Propulsion cả đội 883.657 J; UAV RF 0,0654 J. Không kết luận tiết kiệm pin.
- Search đủ 10 target, nhưng target ID 3 và 7 chưa về GCS.
- Có 3 lượt report hết TTL, không có drop; buffer cả đội trống khi hết giờ.
- 81 lựa chọn truyền đều được receiver chấp nhận, không còn receiver rejection.
- PDR packet 86,11%; có 25 PHY failures. Hop report trung bình 2,875.
- Có 3.479 boundary interventions và 1.858 commanded stalls trong 4.800
  cơ hội di chuyển. Policy cũ vẫn thường ra lệnh hướng ra ngoài map.

Như vậy, avoidance giải quyết rõ nút thắt tìm kiếm trên map này, nhưng
**chưa hoàn thành toàn bộ nhiệm vụ**. Relay/delivery trước TTL và thoát khỏi
hành vi bám biên vẫn là các điểm cần học lại.

Trong `ReportStore`, TTL xóa report còn giữ trong buffer; `known` không bị
xóa. UAV đã biết target không tự tạo lại report khi nhìn lại target đó.
Một UAV khác chưa biết target vẫn có thể phát hiện và tạo report mới.
Đây là semantics hiện tại; lượt sửa này không âm thầm thay TTL, buffer,
custody hoặc cơ chế tái phát hiện để tăng success.

## Những ca đã kiểm tra

- Target ngoài tầm camera lúc đầu, bị obstacle che; sau detour thấy target
  sát mép obstacle mà không có hard stop.
- Cùng tình huống có native UavNetSim: tìm thấy rồi truyền về GCS, kết thúc
  success bằng controller scripted của kiểm thử.
- Tăng tốc hướng vào obstacle từ trạng thái đứng yên: đi vòng, vẫn giữ
  giới hạn gia tốc.
- Hành lang hẹp nhưng hợp lệ: lệnh bay thẳng giữ nguyên, không bị đẩy lệch.
- Hai UAV đối đầu; một peer đã hết pin đứng yên.
- Obstacle sát biên map: ca chọn phía bịt kín đã tái hiện fail trước sửa,
  sau sửa đi qua phía mở.
- Phanh trước biên; tốc độ tối đa khi bay chéo; checkpoint cũ giữ mode reject.
- Rollout lặp lại khớp từng giá trị với cùng input.

Bộ kiểm thử toàn project: **113 tests, 107 passed, 6 CUDA skipped** trên
host không có CUDA, Python 3.13.2 / NumPy 2.2.6 / PyTorch 2.7.1+cu126.
Native CPU training smoke: MASAC và MATD3, mỗi thuật toán chạy hai lần từ
đầu, seed 44, 2 env, 4 episode, 160 joint transitions, 77 learner updates.
Cả model hash, episode hash và settings hash đều khớp cho mỗi thuật toán.
Smoke xác minh wiring/gradient/tính lặp lại, không chứng minh hội tụ.

## Giới hạn còn lại

- Không có đường vật lý hợp lệ thì local avoidance không thể tạo đường.
- Cụm vật cản tạo hốc kín hoặc vòng lặp vẫn có thể làm local planner kẹt;
  không có chứng minh global completeness.
- Khi tốc độ ban đầu quá lớn và quá gần obstacle, không có lệnh trong giới
  hạn gia tốc nào dừng kịp. Exact guard có thể buộc dừng theo semantics cũ;
  `shield_infeasible` ghi trường hợp này.
- Pin không đủ một bước vẫn áp dụng conservative full-step stop cũ.
- Target bị occlusion cần đổi viewpoint. Nếu FOV/clearance làm mọi viewpoint
  hợp lệ đều không quan sát được target thì đổi avoidance chưa đủ.
- Tanh saturation, phối hợp relay, coverage và TTL không được chữa bằng việc
  có một safety filter. Cần train mới với dynamics mới và đọc các metrics đó.

## Chạy lại và đọc log

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_motion_safety.py \
  --checkpoint outputs/matd3-completed-review/model/checkpoint.pt \
  --seed 10044 --output outputs/motion-comparison-new

PYTHONPATH=. .venv/bin/python scripts/compare_motion_safety.py \
  --checkpoint outputs/matd3-completed-review/model/checkpoint.pt \
  --receiver-admission-mask true --seed 10044 \
  --output outputs/motion-admission-new

PYTHONPATH=. .venv/bin/python scripts/check_motion_repeatability.py \
  --output outputs/motion-repeat-new
```

Default mới: `world.motion_safety=dynamic_window`. Chọn `reject` để làm
baseline. Checkpoint cũ không chứa field này được load bằng mode reject,
nên evaluate lịch sử không tự bị đổi dynamics.

W&B giữ `console="wrap"`. Console từng episode nay in cả nguyên nhân kết
thúc, pin tối thiểu, UAV còn hoạt động, motion counts và routing counts.
W&B thêm intervention/stall/infeasible rates ngoài các raw motion counters.

Profile `configs/motion-study.yaml`: fresh MASAC/MATD3, training root seed
44, evaluation đúng một seed 10044, 100 episode, 800 bước tối đa, 2 env CPU
mỗi learner, batch 512, replay CUDA 100.000, warmup 4.000, 1 update/vector
step, giới hạn mềm 3 giờ. Nhiều map episode trong một training run vẫn là
một training root seed, không phải nhiều độc lập training seeds.

Kết quả dài hoặc GPU của source mới được ghi theo trạng thái thực tế;
time-cutoff không được gọi là complete hoặc một exact repeat thành công.

## Nguồn sơ cấp

- Fox, Burgard, Thrun (1997), DWA: vận tốc đạt được và có thể phanh an toàn:
  https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance
- Ge, Cui (2000), vấn đề GNRON và APF khi goal gần obstacle:
  https://bpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/f/14858/files/2021/07/TRA-potential-00.pdf
- Ames et al. (2017), CBF/QP và safety constraints:
  https://authors.library.caltech.edu/records/jnhr0-1ww05
- Tan, Dimarogonas, equilibria không mong muốn của CBF/QP:
  https://arxiv.org/abs/2104.14895
- PyTorch: cùng seed chưa đủ để bảo đảm giống kết quả trên khác version/platform:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
