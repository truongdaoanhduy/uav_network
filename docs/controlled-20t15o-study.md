# Kịch bản 2D có kiểm soát: 20 target / 15 obstacle

Cập nhật 2026-10-30: khôi phục action học công suất và chọn best bằng validation. Cấu hình chính:
`configs/comparison-20k-hybrid75-1kb.yaml`.
Bản 35/25 trước đây được giữ tại
`configs/comparison-20k-hybrid75-1kb-35t25o.yaml`.

## Vì sao giảm độ khó

Kịch bản cũ chưa có bằng chứng là vô lý hoặc không thể giải. Nó ghép nhiều
quyết định cùng lúc: tìm kiếm không có coverage memory, di chuyển an toàn,
chọn receiver/thời điểm im lặng, công suất phát, custody, buffer và TTL.
Phần audit trước thấy nhiều lần phát dưới công suất yêu cầu ở MASAC,
cùng khoảng cách giữa discovery và delivery. Đây là giả thuyết cần ablation;
chưa đủ để kết luận toàn bộ nguyên nhân chưa ổn định nằm ở kịch bản.

Bản mới giảm target/obstacle. Theo yêu cầu 30/10, baseline tiếp tục học
movement, công suất và routing/silence như action cũ; link-budget power
được giữ làm một ablation riêng. Vẫn giữ mục tiêu
tìm target -> tạo/chuyển tiếp report -> GCS, liên lạc hữu hạn, pin, queue,
TTL, CSMA/retry/ACK và dynamic-window shield.

## Cấu hình và bảy phép so riêng

Tất cả dùng 6 UAV, map 1000 m, 20 target, 15 obstacle, horizon 1200 bước,
report/payload 1024 B và radio range 500 m. Cấu hình production yêu cầu
20.000 episode, E4096, batch8192, replay1M, FP32. Số episode bằng nhau
không bảo đảm số transition bằng nhau khi kết thúc episode sớm.

| Variant | Thay đổi so với baseline |
| --- | --- |
| baseline | Learned power; buffer 3072 B; TTL 300 s; pin 277200 J; coverage tắt |
| link_budget | Chỉ thay power bằng bộ điều khiển nominal +3 dB |
| buffer16 | Chỉ buffer 16384 B = 16 report |
| buffer32 | Chỉ buffer 32768 B = 32 report |
| ttl600 | Chỉ TTL 600 s |
| coverage8 | Chỉ thêm memory cục bộ 8x8 cho mỗi UAV |
| battery2x | Chỉ tăng pin lên 554400 J |

Không gộp buffer/TTL/coverage/pin vào cùng một variant. So từng variant
với baseline mới cho biết yếu tố nào giúp ích.

Trong ablation link_budget, công suất là `clip(P_required * 10^(3/10), P_min, P_max)`, tính theo
receiver đã chọn và vị trí sau di chuyển. Nó dùng nominal link budget;
interference, contention hoặc chạm giới hạn công suất vẫn có thể làm mất gói.
Bộ điều khiển này cần cùng giả định ước lượng link/geometry đang dùng cho
admission; không phải routing biết trước toàn bộ nhiệm vụ.

Action vẫn có ba continuous slot để giữ ABI. Trong link-budget mode, slot
công suất được đặt 0 ở actor/replay/target và bỏ khỏi log-probability,
entropy target SAC. Critic không được học từ giá trị slot bị bỏ qua.
PQL inference và shared/independent actors dùng cùng quy tắc.

Coverage 8x8 chỉ nhớ tâm coarse cell đã nằm trong FOV+LOS của chính UAV đang
hoạt động. Không chia sẻ map, không lấy target truth hoặc receipt từ GCS.
Memory được reset theo từng world và lưu trong exact checkpoint.
Mặc định coverage 0 và learned power của config cũ được giữ để đọc checkpoint cũ.
Cấu hình 20/15 là một lineage mới; không resume checkpoint 35/25 vào đó.

## Evaluation và training seed

Training seeds: 44,100044,200044. Dùng cùng seed theo từng phép so.
Validation: 5 episode ở seed 1000000044..1000000048, mỗi 3686400 joint
transition, CPU worker bất đồng bộ. Final test: 30 episode ở
seed 2000000044..2000000073. Các bank này được kiểm tra không chồng lấn.

30 test episode không phải 30 training seed. Summary tính mean và sample
standard deviation giữa mean của các training seed, hiện seed thiếu,
budget chưa đủ và số transition thực tế. Chỉ nhận final evaluation khớp
SHA256 checkpoint, toàn bộ config, thuật toán, BASE/PQL, FP32, seed bank và
training step. Không tự chọn winner khi budget chưa tương đương.


## Chọn checkpoint tốt nhất

`best.pt` là actor snapshot thực sự đã được periodic validation, không phải
weights mới hơn tại thời điểm child trả kết quả. Thứ tự chọn là success,
delivery, search, return; bằng điểm giữ candidate sớm hơn. Chỉ dùng bank
validation, không dùng final-test seeds hoặc peak reward training.
`best.json` ghi seed bank, score, SHA256, source episode/global_step.

Train result `checkpoint`, final model artifact và CPU evaluation/visualize
dùng best khi có candidate hợp lệ. Artifact thêm alias `best`; metadata
phân biệt episode cuối đã train với episode của policy được chọn.
`resume_latest.pt` vẫn giữ learner/optimizer/replay/RNG/environment cuối để
continuation chính xác, kèm bản best từ segment trước. `best.pt` dùng inference/evaluation; manual BASE warm-start có thể lấy
actors từ best nhưng bắt đầu lineage mới với critics/optimizer/replay và
counters mới, ghi rõ `actor_only_warm_start_fresh_training`. Không dùng
nó thay full resume.

Nếu không có validation hoàn tất hợp lệ, summary ghi
`unvalidated_final_fallback`; không gọi checkpoint cuối là best. Shutdown
vẫn có giới hạn thời gian; xem evaluations/events.jsonl nếu child bị skip,
timeout hoặc cancelled. Summary study tách final training steps khỏi
selected_policy_global_steps, vì best có thể nằm sớm hơn cuối training.

## Lệnh sử dụng

Chạy từ repository root, trong môi trường đã cài dependencies của project.
Bộ 84 YAML cũ trong `outputs/controlled-20t15o-study-20261029/` được tạo khi horizon còn 1800 và chỉ nên coi là lịch sử.
Các manifest guided-power/1200 cũ cũng giữ làm lịch sử. Sau khi đổi learned
power, tạo manifest mới; không sửa YAML resolved cũ rồi bỏ qua fingerprint.
Đây vẫn là 7 variant x 2 thuật toán x 2 mode x 3 seed; chưa submit cloud job.

Tạo một study mới (thư mục đích phải chưa tồn tại):

```bash
python -m uav_network study plan \
  --output-dir outputs/controlled-20t15o-next
```

Có thể chuẩn bị baseline trước bằng cách thêm `--variants baseline`.
Muốn chỉ so hai thuật toán ở BASE thì thêm `--modes base`.
Planner không chạy training. Các YAML đã resolved có thể dùng với luồng
cloud hiện có; study CLI không phải cloud scheduler.

Chạy tuần tự một case trên máy có CUDA và cấu hình phù hợp:

```bash
python -m uav_network study run \
  --manifest outputs/controlled-20t15o-next/manifest.json \
  --case-ids 20t15o-baseline-masac-base-s44
```

Bỏ `--case-ids` sẽ chạy toàn bộ manifest tuần tự. Không dùng lựa chọn này
cho 84 case chỉ để thử pipeline. Output đã tồn tại được bảo vệ khỏi ghi đè.
Run chưa đủ episode budget không được đưa vào final study summary;
dùng cơ chế continuation hiện có để hoàn thành trước khi kết luận.

```bash
python -m uav_network study summarize \
  --manifest outputs/controlled-20t15o-next/manifest.json
```

Trước khi chi ngân sách cho toàn bộ 84 case, nên chạy baseline đủ budget,
xem periodic delivery/search/success, delay, energy và safety, rồi ưu tiên
các ablation theo nút thắt quan sát được. Không tự thay reward hoặc tăng
đồng thời nhiều tài nguyên khi đang so một yếu tố.

## Bằng chứng kiểm tra của lần triển khai

- Feasibility oracle lịch sử trước khi đổi horizon có quyền biết target: 5/5 layout giao 20/20 report,
  seed 3000000044..3000000048; episode 89..194 bước, mean 138.4;
  mean pin cuối 88.07%. Lần kiểm này cho phép horizon 1800 bước; vì mọi episode đều <200 bước,
  các trajectory đó cũng nằm trong giới hạn 1200, nhưng đây không phải một lần chạy feasibility mới.
- Đây là khả thi trên các scene đã thử, không phải kết quả policy học,
  không bảo đảm mọi scene hoặc khả năng hội tụ của MASAC/MATD3.
- Smoke lịch sử 29/10 (baseline guided power): 18 case train+evaluate, baseline đủ 4 thuật toán/mode trên
  3 seed, thêm 6 ablation MATD3 BASE seed 44. Mỗi case 2 episode,
  E2, horizon 8, hidden 8x8, global_step 16, learner update 5, final eval 2.
  Tracking tắt; không dùng số smoke cho paper. Periodic CPU evaluation
  được kiểm riêng bằng một child process thật với profile mới.
- Summary smoke mới xác minh đủ 18 checkpoint; 66 case còn lại là missing,
  không được diễn giải thành kết quả thực nghiệm.
- Kết quả máy đọc được và tổng kiểm tra cuối:
  [evidence](superpowers/evidence/2026-10-29-20t15o-study.json).
- Máy kiểm tra không có CUDA. Native/Warp CPU đã kiểm; chưa xác nhận
  kernel parity/performance trên GPU cho thay đổi này, chưa train 20k
  hoặc kết luận hội tụ của study mới.

## Profile mở rộng: sensing và XYZ

Đã đọc reference, không sửa:
`/home/aduy/Documents/NCKH/uav_search_target/uav_marl/world/sensing.py`
và `configs/task/uav_search.yaml`.
Reference dùng grid 25 m, FOV 90 độ, prior 0.5; Pd/Pf theo mốc độ cao
[0,50,100,150] m là [0.99,0.90,0.80,0.70]/[0.01,0.10,0.20,0.30].
Bayes cập nhật posterior. Confirmation yêu cầu UAV active, altitude<=50 m,
posterior>=0.99 và có positive observation; false confirmation vẫn được phân biệt.
Không sao chép cả profile tham chiếu nặng hơn (map 3 km, 50 target, 30 obstacle).

Ngày30/10 đã triển khai [profile XYZ/Bayesian riêng](xyz-sensing.md), giữ20/15:
action/motion theo z, vật cản có chiều cao, LOS/radio/energy theo XYZ,
camera/belief/confirmation, observation/replay/checkpoint và visualize2D.
Baseline XY trong tài liệu này vẫn giữ nguyên để đối chiếu. Kiểm tra CPU nhỏ
không thay thế số đo training dài hoặc xác nhận khả thi20/20 của XYZ.
Oracle cũ chỉ hỗ trợ XY và báo rõ khi nhận profile XYZ.
