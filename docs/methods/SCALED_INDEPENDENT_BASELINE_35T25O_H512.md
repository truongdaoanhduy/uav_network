# Thí nghiệm độc lập35 mục tiêu /25 vật cản /512x512

Cấu hình riêng: `configs/comparison-independent-35t25o-h512-xyz-sensing.yaml`.
Giữ nguyên hai notebook20t15o/256 đã giao làm đối chứng.
Bản mới là thí nghiệm tăng kích thước, chưa phải kết quả cải thiện.

## Thay đổi đúng theo yêu cầu

Cả MASAC và MATD3 dùng hidden_sizes=[512,512],35 mục tiêu,25 vật cản.
Đổi tên run/group/output/tags để không trộn với thí nghiệm trước.
Giữ updates_per_step=8 vì người dùng đang hỏi ý kiến về16/32.
Không sửa phương trình thuật toán, learning rate, alpha tự học, initialization,
reward, CBF, pin, sensor slots, số UAV, bản đồ1000m hay trần1500 bước.
Tất cả UAV hết pin vẫn kết thúc episode theo cơ chế đã có.
Seed44,4096env,batch8192,replay1M,NOCHUNK và lịch validation giữ nguyên.

## Dimension thật, tính từ code

| Kiến trúc/kịch bản | Actor mỗi UAV | State critic | Critic kể cả joint action |
|---|---:|---:|---:|
| Run SH cũ20t15o | 204=198+6 agent-ID | 565 | 631 |
| Independent đã giao20t15o/256 | 198 | 565 | 631 |
| Independent mới35t25o/512 | 228 | 830 | 896 |

Nguồn: environment/observations.py và algorithms/models.py.
Joint action gồm6x(4 continuous+7 destination)=66 chiều.
Bốn continuous là XYZ acceleration và RF power.
MASAC có mean4, log_std4 và7 destination logits; MATD3 có4 continuous
và7 destination logits. Tất cả actor mới nhận228 đặc trưng mỗi UAV.

256 hoặc512 là số neuron lớp ẩn, không phải observation dimension.
Linear228→512 và Linear896→512 đều hợp lệ; không cần hidden>=input.
256 trước đó là lựa chọn baseline, chưa được chứng minh tối ưu cho nhiệm vụ.
Mạng rộng hơn chỉ giúp khi năng lực biểu diễn đang là giới hạn; cũng tăng
chi phí và có thể làm tối ưu khó hơn. Reward thấp không tự chứng minh thiếu neuron.

Actor vẫn có4 visible-target slots và3 visible-obstacle slots trong local FOV.
Tăng số vật thể toàn cảnh không tự cho actor thấy toàn bộ tọa độ;
critic CTDE có state toàn cục. Sensor/visibility không bị thay đổi.

## Update8/16/32 thực sự có nghĩa gì

Sau warmup, mỗi vòng collector thu4096 joint transitions, sau đó learner
gọi update U lần, mỗi lần lấy8192 mẫu có hoàn lại từ replay.
U là updates_per_step của code, không phải U update trên từng transition.

| U | Critic calls/collection | MASAC actor/alpha calls | MATD3 actor calls | Mẫu lấy/transition mới |
|---|---:|---:|---:|---:|
| 8 | 8 | 8/8 | 4 | 16 |
| 16 | 16 | 16/16 | 8 | 32 |
| 32 | 32 | 32/32 | 16 | 64 |

Cột cuối là Ux8192/4096, giá trị trung bình về lượng lấy mẫu,
không phải mỗi transition chắc chắn đã được học đúng số lần đó.
Gradient-update/transition là U/4096 theo định nghĩa replay ratio.
MATD3 policy_delay=2; calls áp dụng cho mỗi policy, không cộng6 actor
rồi gọi đó là6 lần kinh nghiệm mới.

16 có thể tận dụng dữ liệu hơn nếu learner đang thiếu update.
32 cũng có thể làm critic thiên lệch, học lặp dữ liệu hoặc bất ổn hơn.
Chi phí phần learner tăng xấp xỉ2x/4x so với8 ở cùng mạng/budget;
thời gian toàn job không nhất thiết đúng2x/4x vì còn collector/eval/I/O.
Giới hạn10 giờ vẫn giữ nguyên; mạng/update lớn có thể khiến run thu ít
transition hơn trước khi hết thời gian. So cùng transition budget khi đánh giá.
Không có bảo đảm đồ thị mượt hơn hoặc policy tốt hơn.

Khuyến nghị: giữ8 cho phép thử đổi scene/width đầu tiên; sau đó so8 và16
trên cùng scene35t25o, cùng mạng512, seed/budget/eval bank.
Chỉ thêm32 khi16 có lợi ổn định. Đổi run label/group nếu đổi U.
Không lấy đường smooth làm tiêu chí; giữ raw eval và cùng cách hiển thị.

Nguồn nghiên cứu gốc:
- Fedus et al., ICML2020, Replay ratio là gradient updates/environment transition:
  https://proceedings.mlr.press/v119/fedus20a.html
- Chen et al., ICLR2021, REDQ kết hợp UTD cao với ensemble và random subset:
  https://arxiv.org/abs/2101.05982
- Code tác giả cảnh báo tích lũy bias khi update quá nhiều trên ít dữ liệu:
  https://github.com/watchernyu/REDQ
Đây là bằng chứng về đánh đổi, không phải kết luận định lượng cho UAV này;
không thêm REDQ/dropout/ensemble vào baseline MASAC hiện tại.

## Chi phí và cách đọc kết quả

| Đại lượng |20t15o/256|35t25o/512|
|---|---:|---:|
| MASAC tham số online,6 actor+6 twin critics |3,457,638|10,994,790|
| MATD3 tham số online,6 actor+6 twin critics |3,451,470|10,982,478|
| Replay CPU gồm born/seen counters |9.91GiB|12.40GiB|
| Replay GPU,25% của1M rows |3.30GiB|4.13GiB|
| Mission score tối đa=60xnum_targets |1200|2100|

Tham số online tăng khoảng3.18x; con số này do tăng cả scene và width.
Trong cùng scene20t15o, chỉ tăng256→512 làm tham số online tăng khoảng2.68x.
Replay trên chỉ là storage, chưa gồm model/optimizer, activation, môi trường
và CUDA workspace. CPU kiểm tra nhỏ không chứng minh đủ VRAM T4 batch8192.
BASE dùng GPU0 collector, GPU1 learner/replay khi có2GPU; VRAM không gộp.

35 mục tiêu với cùng6UAV/pin/1500step là nhiệm vụ khác và có thể khó hơn.
Reward/mission score tuyệt đối không so trực tiếp giữa20 và35 mục tiêu;
báo thêm search%,delivery%,mission_score/(60xnum_targets),success,CBF,
vi phạm sau lọc và tỷ lệ hết pin. Tăng cả scene và width cùng lúc không
đủ để kết luận riêng rằng512 tốt hơn256. Paper cần đối chứng và nhiều seed.

## Export và kiểm chứng

Từ cây Git đã commit sạch:
```bash
python scripts/build_independent_baseline_ui_notebooks.py --scaled --seed 44
```

Tên mới chứa35T25O_H512_U8; không ghi đè notebook20t15o/256.
Notebook MASAC/MATD3 dùng chung source archive đã pin commit.
Fresh Kaggle session, Private,T4x2,Internet,Secrets và NOTEBOOK_REF thật.
Không warm-start checkpoint cũ vì scene/network dimension đã thay đổi.

Đã chạy thu transition thật trên Warp CPU, replay,2 learner updates của
mạng512 cho cả hai thuật toán; actor/critic đổi trọng số và metrics hữu hạn.
36 scene seeds (train44,5validation,30eval) sinh hợp lệ lần đầu, không fallback.
Kiểm tra này chỉ xác nhận đường chạy và shape, chưa đánh giá học/VRAM GPU.
Xem JSON evidence và CURRENT_TASK.md để biết kết quả suite/export cuối cùng.
