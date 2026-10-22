# Đánh giá W&B UAV ngày 21/10/2026

Đối chiếu ngày 22/10/2026, múi giờ Việt Nam. Nguồn: W&B Public API, toàn bộ history không lấy mẫu, output.log, failure.json, cấu hình đã ghi trong từng run, output và job nối tiếp trên Kaggle.

Đã thu 29 run trong khoảng từ 20/10 giờ Việt Nam để bao gồm các run chạy sang ngày 21/10, 24,849 dòng history. Trước các lượt đánh giá mới của phiên này, các run được thu không có video W&B. Đây là kiểm tra tiến độ và lỗi vận hành; không phải kết quả thí nghiệm nhiều seed.

## Kết luận thực hiện

- Không tiếp tục nguyên trạng 4 lượt replay 1 triệu: 3 lượt lỗi lưu Hugging Face, MASAC PQL hết VRAM. Có policy được cứu từ Kaggle; chúng chưa đạt 20.000 episode.
- Nhóm replay 500k có áp lực VRAM thấp hơn. Hiện chưa có bằng chứng replay 1 triệu học tốt hơn 500k khi giữ các yếu tố khác cố định.
- Giữ Gen1; microbenchmark khoảng +0,59% và probe pipeline ngắn khoảng +1,40% chưa đủ để promote rxindex. Probe ngắn đã xong; nhóm 500k đã được gửi phục hồi, trạng thái chi tiết ở phần 9.
- Đã chuyển kho duy95/uav-search-checkpoints sang public theo yêu cầu, xác minh truy cập ẩn danh và checksum. Khôi phục 7 policy có checkpoint; MASAC PQL 1M không có policy .pt được lưu trước OOM.
- Chạy các video đánh giá CPU từ checkpoint giữa chừng, cùng seed 10044. Chỉ một episode cho mỗi policy; không dùng để kết luận hơn/kém về dung lượng replay.

## 1. Vì sao Finished nhưng chưa đủ episode?

Cấu hình thực tế đặt runtime.max_train_seconds = 36000, tức 10 giờ train. Launcher tính thời gian từ một ceiling 39600 giây (11 giờ), rồi trừ thời gian setup và khoản dự phòng lưu checkpoint/chuyển session. Với artifact_timeout_seconds=900, khoản dự phòng ít nhất là 1020 giây. Giới hạn session Kaggle và ngân sách trainer là hai giới hạn khác nhau. W&B chỉ tính thời gian từ lúc run khởi tạo, không bao gồm toàn bộ setup trước đó.

Trainer dừng bình thường vì time_budget rồi gọi W&B finish với exit code 0. Vì vậy run có state=finished, đồng thời episode_budget_complete=false và session_status=session_time_budget_reached. Đây là kết thúc một đoạn training, không phải hoàn tất mục tiêu 20.000 episode. Các run DIAG 1M chạy vài phút là kiểm tra bộ nhớ/thực thi, không phải production training.

| Policy 500k | Vector step đã chạy | Episode ghi nhận | Cập nhật learner | Thời gian train |
|---|---:|---:|---:|---:|
| [MATD3 BASE](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/dc16b47e) | 941 | 0 / 20.000 | 7,024 | 10.01 giờ |
| [MATD3 PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/925d10c4) | 793 | 0 / 20.000 | 6,096 | 10.00 giờ |
| [MASAC BASE](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3efe3a0b) | 1,137 | 0 / 20.000 | 8,592 | 10.00 giờ |
| [MASAC PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/1e0c6017) | 1,741 | 1,296 / 20.000 | 13,680 | 10.01 giờ |

Global step là tổng transition của các môi trường. Ví dụ 2.328.576 / 2.048 = 1.137 bước/môi trường. Khi một episode có thể dài 1.800 bước, hoàn toàn có thể đã cập nhật model 8.592 lần mà chưa có môi trường nào kết thúc episode đầu. Off-policy học ngay khi replay qua warmup, không đợi episode kết thúc.

## 2. Thời gian đi đâu?

MASAC BASE 500k ghi timing/collector_wait_seconds = 33.255,40 giây (9,24 giờ) và timing/learner_seconds = 2.713,87 giây (45,23 phút). Collector wait chiếm khoảng 92,4% ngân sách 10 giờ. MATD3 BASE 500k ghi 35.538,95 giây chờ collector và 455,87 giây learner. Các chỉ số PQL có phần thời gian chạy chồng nhau, không cộng thành tổng wall-time.

Bottleneck quan sát được nằm ở việc sinh transition qua simulator mạng gói tin, không phải thiếu số lượt cập nhật model. Mỗi report 1 MB gồm khoảng 977 fragment 1024 B, được mô phỏng MAC/PHY, backoff, ACK/retry và các sự kiện liên quan cho 2.048 môi trường. Việc tăng thời gian tối đa lên 12 giờ không tự giải quyết nguyên nhân này.

Tốc độ thay đổi mạnh theo giai đoạn: chẳng hạn MASAC BASE 500k mất khoảng 4,62 giờ để tới bước/môi trường 201, khoảng 9,21 giờ để tới 801. Không nên ngoại suy thời gian cả nhiệm vụ bằng một throughput trung bình cố định.

## 3. Vì sao không tự nối tiếp và chưa visualize?

Cả bốn job 500k đã submit child c01. MATD3 BASE tải được checkpoint, tiếp tục tới global_step 3.497.984 rồi gặp lỗi upload checkpoint. Ba child còn lại dừng trước khi tạo W&B training run vì tải resume.pt trả 403: private repository storage limit reached. Đây là lỗi lưu/tải checkpoint, không phải bằng chứng session mới đã train đủ ngân sách.

Luồng auto_continue hiện chỉ chuyển sang CPU visualization khi đủ tổng 20.000 episode; khi chưa đủ, nó chọn GPU continuation. Nếu learner lỗi, continuation bị chặn. Vì vậy yêu cầu video không tạo ra video giữa các đoạn training bằng workflow cũ.

## 4. Replay 1 triệu: lỗi thực tế

| Policy | Global step cuối | Episode trong history | Trạng thái | Nguyên nhân |
|---|---:|---:|---|---|
| [MATD3 BASE](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/fd3417ed) | 3,512,320 | 145 | failed | Hugging Face 403: hết dung lượng private |
| [MATD3 PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/e355261d) | 3,493,888 | 144 | failed | Hugging Face 403: hết dung lượng private |
| [MASAC BASE](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2e2bc292) | 3,424,256 | 128 | failed | Hugging Face 403: hết dung lượng private |
| [MASAC PQL](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/9b0ea463) | 1,929,216 | 0 | failed | CUDA OOM trên GPU1 |

Một vài summary/progress counter thấp hơn số episode thực tế trong history vì run lỗi trước lần cập nhật progress tiếp theo: MASAC BASE 1M có 128 record episode, trong khi progress cuối ghi 124; MATD3 BASE 1M có 145 record, trong khi progress cuối ghi 99. Không dùng counter progress cũ để phủ nhận các record đã ghi.

Replay 500k chiếm 6.552.500.000 byte (~6,10 GiB); replay 1M chiếm 13.105.000.000 byte (~12,21 GiB) trên learner GPU. MASAC PQL 1M OOM khi cần thêm 462 MiB trong khi GPU1 chỉ còn 446,81 MiB. AMP không biến toàn bộ replay FP32 thành FP16. GPU0 và GPU1 giữ vai trò khác nhau nên VRAM không tự chia đôi.

## 5. So sánh ở cùng mốc training

Các nhóm 500k và 1M cùng env=2048, batch=4096, seed=44. Tuy nhiên nhóm 500k dùng FP32/compile=false, commit 316ac1a; nhóm 1M dùng AMP FP16/compile=true, commit 6de210f. Không phải phép thử chỉ thay dung lượng replay.

Bảng sau lấy progress tại global_step=1.599.488, tương ứng 781 bước/môi trường, trước episode đầu kết thúc. Đây là tỷ lệ giao target trong các episode đang chạy, không phải evaluation của policy đã cố định.

| Policy | Delivery 500k | Delivery 1M | Search 500k | Search 1M |
|---|---:|---:|---:|---:|
| MASAC BASE | 56.77% | 56.43% | 94.56% | 92.61% |
| MASAC PQL | 57.43% | 54.46% | 92.07% | 87.88% |
| MATD3 BASE | 13.27% | 10.35% | 73.28% | 73.16% |
| MATD3 PQL | 23.54% | 12.34% | 86.65% | 84.91% |

Ở mốc này, không thấy lợi thế delivery của nhóm 1M; MATD3 PQL 1M còn thấp hơn rõ rệt. Vì có thay đổi AMP/compile/code, chưa thể quy nguyên nhân cho replay. Lựa chọn vận hành tạm thời nghiêng về 500k vì VRAM an toàn hơn, không phải tuyên bố 500k tối ưu về khoa học.

### Env/batch khác nhau

MASAC BASE 4096 env / batch 8192 / replay 500k (run 0a296462, bắt đầu ngày 20/10 và chạy sang 21/10) đạt 134,92 joint transition/s; MASAC BASE 2048 / 4096 / 500k đạt 64,68. Nhưng sau ~10 giờ, số bước/môi trường chỉ là 1186 so với 1137. Tăng số môi trường làm tăng số transition tổng; nó không khiến mỗi UAV trải qua nhiệm vụ nhanh gấp đôi. Hai lượt cũng khác commit, nên đây chỉ là số quan sát vận hành.

### Kết quả các episode training đã kết thúc

Trung bình 20 episode cuối có record đầy đủ: MASAC PQL 500k search 92,14%, delivery 59,43%; MASAC BASE 1M search 94,14%, delivery 60,29%; MATD3 BASE 1M search 80,57%, delivery 14,86%; MATD3 PQL 1M search 89,29%, delivery 25,86%. Các episode này đều kết thúc vì cạn pin; chưa có mission success trong các record được kiểm tra. Đây là training rollout, không phải final evaluation 30 seed.

Relay thực sự hoạt động: 20 episode cuối MASAC PQL 500k trung bình 8,9 report giao GCS qua relay, 11,9 trực tiếp. Với MASAC BASE 1M là 8,9 relay và 12,2 trực tiếp. Tuy vậy delivery và tối ưu pin vẫn chưa đáp ứng toàn nhiệm vụ. Năng lượng propulsion chiếm phần áp đảo so với RF.

## 6. Lưu public và video đánh giá mới

Kho checkpoint: https://huggingface.co/duy95/uav-search-checkpoints

Đã xác minh repository private=false bằng API không đăng nhập. Các đường dẫn checkpoint/revision trước đó được giữ nguyên; 3 policy 1M được cứu từ output Kaggle và tải lên public. Bốn policy 500k được đối chiếu với bản đã có. Đã tải một checkpoint bằng token=False và kiểm tra SHA-256 khớp.

Cấu hình public mặc định và hai profile V3 đã được sửa tại commit [ab60690b76aeaf7d0a72cb10e782480be0cd1df0](https://github.com/truongdaoanhduy/uav_network/commit/ab60690b76aeaf7d0a72cb10e782480be0cd1df0), nhánh `fix/public-checkpoint-storage-20261022`. Toàn bộ 8 profile comparison đang được kiểm tra đều cấu hình public. Fresh verification: **601 passed, 72 skipped, 1203 subtests passed; 8 warnings**. Các kiểm thử cần CUDA vẫn bị skip trên máy CPU; thay đổi chỉ liên quan lựa chọn lưu trữ, không sửa thuật toán hoặc simulator. Git diff --check sạch; remote head khớp commit trên.

Lưu ý khả năng tiếp tục: các policy được cứu để visualize là checkpoint version 2, chỉ đủ đánh giá hoặc warm-start model; không tự coi là full resume có optimizer/replay/environment. Nhóm 500k có resume.pt riêng trong các artifact đã lưu trước đó. Phần 9 ghi kiểm tra full resume và bằng chứng training mới; việc chỉ chuyển kho sang public không tự chứng minh đã khôi phục đầy đủ.

Đã hoàn tất và đối chiếu W&B media/Hugging Face cho **7 video**. Cùng seed 10044, một episode/policy, tối đa 1800 bước, backend Warp chạy trên CPU cho một world; không thay ngữ nghĩa mạng.

| Policy | Replay | Search | Delivery | Relay/direct về GCS | Xem |
|---|---:|---:|---:|---|---|
| MASAC BASE | 500,000 | 71.43% | 51.43% | 9/9 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f1891ce5) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/da5845a5102cedce2cb9156ebd702b15b97d21d7/visualizations/base/masac/f1891ce5/final_episode.mp4) |
| MASAC BASE | 1,000,000 | 25.71% | 17.14% | 5/1 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/62a75f44) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/e008dd61268806ec023e04a5751bb76099db3320/visualizations/base/masac/62a75f44/final_episode.mp4) |
| MASAC PQL | 500,000 | 54.29% | 31.43% | 10/1 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bf6a57ce) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/0f0a21f64770e077909c353c928663c374bd0f9a/visualizations/pql/masac/bf6a57ce/final_episode.mp4) |
| MASAC PQL | 1,000,000 | — | — | — | Không có policy .pt trước OOM |
| MATD3 BASE | 500,000 | 25.71% | 0.00% | 0/0 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2bd26f17) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/c2d6680d4d6c3c15ef71ddffff3e2ef74d593120/visualizations/base/matd3/2bd26f17/final_episode.mp4) |
| MATD3 BASE | 1,000,000 | 51.43% | 11.43% | 0/4 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/4b3288c4) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/25895f70a0f16fed1d69668e641d914b71c4f69e/visualizations/base/matd3/4b3288c4/final_episode.mp4) |
| MATD3 PQL | 500,000 | 42.86% | 14.29% | 0/5 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/9fcc5bb6) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/f6ae6c5984c1d31e33e2c2d66ee9d74f7d7f4ceb/visualizations/pql/matd3/9fcc5bb6/final_episode.mp4) |
| MATD3 PQL | 1,000,000 | 77.14% | 17.14% | 0/6 | [W&B](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/1976a76e) · [MP4 public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/25d1269deec8605992c4f4b17cb9f398844c6745/visualizations/pql/matd3/1976a76e/final_episode.mp4) |

Các video đã hoàn tất đều chưa đạt mission success. Chúng không cùng mức training_global_step nên không thể dùng bảng này như ablation chỉ thay replay. Riêng MASAC PQL 500k có 10/11 report giao GCS qua relay, chứng minh đường relay được dùng; nhưng còn 24/35 target chưa giao và pin trung bình cuối episode chỉ ~0,67%. Ảnh đường bay cuối episode cho thấy nhiều quỹ đạo dọc biên, vùng giữa còn target chưa tìm. Đây là dấu hiệu cần đánh giá và kiểm tra thêm, không phải bằng chứng policy đã tối ưu được nhiệm vụ.

Video MASAC BASE 500k đã hoàn tất: 25/35 target được tìm, 18/35 giao về GCS, gồm 9 relay và 9 trực tiếp. Episode dừng tại 1800 bước vì time_limit, pin trung bình còn khoảng 0,29%. Đã xem ảnh quỹ đạo cuối: một phần UAV đi sát biên, vẫn còn target chưa tìm ở phía trong bản đồ. Đây là checkpoint giữa chừng, chưa tối ưu toàn nhiệm vụ.


## 7. Cách tiếp tục hợp lý

1. Dùng cấu hình public cho checkpoint, xác minh full resume trước khi đốt một session train dài.
2. Chưa chạy lại nguyên trạng MASAC PQL 1M: cần một memory gate bao phủ đủ replay và learner/compile peaks. Ưu tiên 500k nếu chưa có lợi ích đo được của 1M.
3. Giữ target/reward/network semantics và các baseline MASAC/MATD3. Khắc phục collector bottleneck bằng những thay đổi được kiểm chứng parity, không cắt MAC/PHY để tăng tốc.
4. Chưa promote rxindex. Probe Gen1/rxindex hiện có được theo dõi riêng; microbenchmark +0,5942% không đủ làm căn cứ đổi production.
5. Phân biệt rõ W&B state, training budget completion, tiến độ episode đang chạy và evaluation completion. Video mới được ghi nhãn intermediate và gắn checkpoint/hash/source run.

Thông số lấy trực tiếp từ cấu hình W&B: map 1000 m, 6 UAV, 35 target, 25 obstacle, report 1 MB, max_steps 1800, bandwidth 22 MHz, bit rate 2 Mbps. Production train ngày 21/10 có warmup=131072 và updates_per_step=8; probe trong prompt bàn giao dùng warmup=409600 và updates_per_step=16. Không trộn hai cấu hình này khi diễn giải kết quả.

## 8. Probe đã chốt, không đổi scheduler production

Probe dài `ioeufwejewlfl/uav-rxindex-prod-probe-20261022-070011` đã bị timeout tại mốc 10.800 giây do launcher đặt. Log cuối tới Gen1 warmup 100/200, chưa có lượt cập nhật learner hoặc kết quả rxindex. Đây là timeout của phép đo, không phải hết quota tuần.

Probe rút gọn [uav-rxindex-mini-20261022-103205](https://www.kaggle.com/code/ioeufwejewlfl/uav-rxindex-mini-20261022-103205) đã COMPLETE. Nó giữ 2 T4, 2.048 env, batch 4.096, replay capacity 1 triệu, AMP FP16, compile, fused Adam và 16 cập nhật/cycle. Để kiểm tra được đường chạy thật, nó chỉ warmup 2 vector step, đặt learning_starts=4096 rồi chạy 3 learner cycle/nhánh, tổng 48 cập nhật/nhánh. Chỉ hai cycle cuối được dùng đo tốc độ, loại cycle đầu có lazy compile.

| Nhánh | Transition/s đo ở hai cycle cuối | Replay đã có dữ liệu | Dung lượng môi trường |
|---|---:|---:|---:|
| Gen1 | 67,95 | 10.240 / 1.000.000 | 1.190.191.104 B |
| rxindex | 68,90 | 10.240 / 1.000.000 | 1.283.833.856 B |

Chênh lệch khoảng +1,40% trong một lượt A/B rất ngắn, đầu episode, replay chưa đầy. Các bộ đếm traffic cuối hai nhánh khớp nhau, gồm 612.297 DATA attempts. Đây chưa phải kiểm tra đầy đủ mọi simulator array hay bằng chứng lợi thế trên cả nhiệm vụ. Warmup của Gen1 còn chứa compile Warp nên không dùng tỷ số warmup để kết luận scheduler nhanh gấp nhiều lần. Quyết định: **giữ Gen1, không promote rxindex**. Phép đo MASAC BASE ngắn này cũng không chứng minh MASAC PQL 1M hết nguy cơ OOM.

## 9. Đã gửi nối tiếp nhóm 500k

Đã tải toàn bộ resume MASAC PQL 500k từ kho public, kiểm tra SHA-256 `9e80bc403caf6bc89e9e00f8cc1c50253013e793d2a6e0799de5ada6c3c8fa45` khớp. File 7.795.326.398 B là version 3: đủ optimizer, replay 500.000, RNG hai GPU và trạng thái 2.048 môi trường. Metadata khớp cấu hình khoa học và mốc 1.296 episode / 3.565.568 transition / 13.680 cập nhật.

Các job phục hồi giữ nguyên commit khoa học `316ac1ae38d987eccb9e41bfd416955a01d47a62`, seed 44, env 2048, batch 4096, replay 500k, FP32, compile=false và mục tiêu tổng 20.000 episode. Override `tracking.checkpoint_repo_private=false` được truyền rõ cho tất cả job và sẽ được các child kế thừa. Các job mới đều dùng full resume qua immutable W&B artifact; chưa chạy lại nhóm 1M.

| Policy | W&B mới | Mốc full resume ban đầu | Trạng thái kiểm tra 18:06 giờ Việt Nam |
|---|---|---:|---|
| MASAC PQL 500k | [b55009f1](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/b55009f1) | 3.565.568 transition, 1.296 episode | Đã học tiếp: 3.586.048 transition, 1.544 episode, 13.752 cập nhật |
| MASAC BASE 500k | [143c7acc](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/143c7acc) | 2.328.576 transition, 0 episode | Đã học tiếp: 2.895.872 transition, 10.800 cập nhật |
| MATD3 PQL 500k | [18e2823f](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18e2823f) | 1.624.064 transition, 0 episode | Đã học tiếp: 1.638.400 transition, 6.136 cập nhật |
| MATD3 BASE 500k | [47bd6da9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/47bd6da9) | 1.927.168 transition, 0 episode | Đã học tiếp: 1.933.312 transition, 7.040 cập nhật |

Cả bốn run đã ghi `continuation.kind=training_state_exact_environment`, `optimizer_replay_restored=true`, source run tương ứng và storage public trong cấu hình W&B. Global step và số cập nhật learner của từng run đều đã vượt mốc đầu vào; bộ theo dõi ghi đủ bốn cờ training_progress_verified=true. Chưa run nào đạt đủ mục tiêu tổng 20.000 episode.

Đã gắn link run phục hồi và kernel mới vào summary của bốn run nguồn 500k. Bảy run nguồn có checkpoint cũng đã có link video và run đánh giá ở `intermediate_evaluation/*`, ghi rõ một episode/seed 10044 và phạm vi intermediate.

Các loss mới đọc được là số hữu hạn. MASAC PQL vẫn có `diagnostics/report_loss_warning=1`; cảnh báo này đã có ở 1.277/1.296 record episode của run nguồn, nên không phải dấu hiệu mới chỉ xuất hiện sau phục hồi. Mã nguồn bật cờ khi cửa sổ episode gần đây có report hết hạn hoặc bị dropped; đây không phải số lần lỗi resume. Không diễn giải việc khôi phục thành công thành bằng chứng policy đã ổn về chất lượng nhiệm vụ. Các video một seed vẫn có mission success bằng 0.

MATD3 BASE cần quay về full resume lưu thành công ở 1.927.168 transition. Đoạn c01 sau đó đạt 3.497.984 transition / 138 episode nhưng lỗi upload trước khi lưu full resume mới. Output còn policy version 2 và failure snapshot; không có optimizer/replay/environment tương ứng mốc mới để khôi phục chính xác. Đã cứu thêm [policy tại 138 episode lên public](https://huggingface.co/duy95/uav-search-checkpoints/resolve/1c6ff320ac447a939fc2537e019aceeb59da79a5/uav-network/cb43b1a0/step-3497984-episodes-138/checkpoint.pt), SHA-256 `8d8cf12ebe2fc46cc66d75d2120cec67510bd624957760a66f7222524d6250ad`. Đây là policy bổ sung, không thay thế full resume. Job MATD3 BASE được giới hạn 31.895 giây theo quota còn lại; các job kia tối đa 39.600 giây, trainer vẫn chừa thời gian lưu và chuyển phiên.

Video MASAC BASE 500k đã được gửi lại trên cùng kernel CPU, lấy trực tiếp policy 26 MB từ public HF theo revision/hash, bỏ mount output kernel cũ. Bản trước báo running hơn một giờ nhưng chưa có log/W&B run; chưa có đủ bằng chứng để kết luận nguyên nhân phía Kaggle. Sau thay đổi, W&B [f1891ce5](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f1891ce5) đã finished với `evaluation/completed=true`, một episode đủ 1.800 bước. MP4, PNG và metrics đã được lưu public tại revision `da5845a5102cedce2cb9156ebd702b15b97d21d7`, kiểm tra hash và sự hiện diện media W&B. Bộ thu xác nhận ALL_VISUALIZATIONS_COLLECTED 7.

Bộ theo dõi chạy độc lập trên máy chủ đang ghi trạng thái Kaggle và bằng chứng tiến độ W&B của bốn job; cơ chế auto_continue sẵn có sẽ nối tiếp khi segment kết thúc bình thường và còn quota. Báo cáo này là ảnh chụp trạng thái tại thời điểm ghi, không phải trang trạng thái tự cập nhật.

Cập nhật dữ liệu báo cáo: 18:06:42 22/10/2026 (giờ Việt Nam).
