# Audit thời gian training 20.000 episode

W&B đọc trực tiếp lúc **00:43 ngày 24/10/2026**. Các số dưới đây là snapshot có thời điểm; không phải cam kết thời gian hoàn tất.

20.000 episode chưa thể hoàn thành trong segment hiện tại. Với trung bình lịch sử 1.710,259 step/episode, cần khoảng **34,2 triệu transition**; nếu mọi episode chạm trần 1.800 thì cần **36 triệu**. Muốn xong trong 10 giờ cần trung bình **950–1.000 transition/s**, chưa tính preflight, checkpoint, chờ tài nguyên và các episode chưa xong ở batch vector cuối. Con số 36 triệu là workload của 20.000 episode hoàn tất, không phải hard ceiling của global_step khi còn thế giới dở dang.

| Lượt | Episode đã xong | Global step | Tốc độ trung bình segment (transition/s) | Ước lượng giờ GPU còn lại |
| --- | ---: | ---: | ---: | ---: |
| [MASAC BASE 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/63faeffb) | 0 | 5,709,824 | 167.4 | 10–50 |
| [MASAC PQL 1M c01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bc18a145) | 3,059 | 7,196,672 | 64.7 | 116–162 |
| [MATD3 BASE 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2dcbf36c) | 0 | 3,432,448 | 101.1 | 40–89 |
| [MATD3 PQL 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/278634f6) | 0 | 2,850,816 | 87.0 | 56–109 |
| [MASAC BASE 500k c01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/30e43dda) | 4,096 | 10,797,056 | 120.8 | 52–59 |

Khoảng ETA dùng tốc độ trung bình của segment và các cửa sổ W&B gần đây, với workload 34,2–36 triệu transition. Đây là phép ngoại suy nếu mức tải hiện tại tiếp tục, không phải khoảng tin cậy hay giới hạn trên chắc chắn. PQL vừa resume có thể nặng hơn segment trước; episode/reset, mức phát và contention làm tốc độ thay đổi lớn. Thời gian lịch còn chịu quota, hàng đợi và upload.

Zero completed episode ở các run chưa kết thúc lượt môi trường đầu tiên không chứng minh counter bị kẹt: 4.096 môi trường đang tiến dần tới ngưỡng kết thúc. Global step và episode_step_mean vẫn tăng. Các child có completed episode kế thừa và tăng tiếp từ parent.

## Nguyên nhân đã có bằng chứng

1. **Mô phỏng packet/MAC là chi phí lớn nhất ở các pha mạng hoạt động.** Một report 1 MB chia thành 977 fragment 1.024 byte, cộng ACK, backoff, retry, collision và interference. T4 diagnostic cùng 4.096 môi trường/seed44/động học dynamic_window đo 9.726,67 transition/s ở 1 KB và 38,74995 ở 1 MB, chậm khoảng 251 lần. Đây là phép đo simulator, không có learner.
2. **Phần triển khai lặp kiểm tra kênh ở độ phân giải microsecond tạo rất nhiều sự kiện.** Probe có instrumentation cbc7253, FOV rộng hơn production, đếm 4.735.399.723 dispatch cộng dồn qua ba vector step; IDLE và LISTEN chiếm 97,814%. Không gán con số này cho production. Nó chỉ ra vị trí cần tối ưu chính xác, thay vì suy đoán GPU bị lỗi.
3. **Hai GPU phân công môi trường/learner.** GPU0 chạy cả 4.096 môi trường; GPU1 chạy learner/replay. Cumulative utilization lệch không chứng minh một GPU bị hỏng. Các mẫu system metrics đã đọc có oomKills=0; không tìm thấy bằng chứng OOM hiện tại trong phạm vi kiểm tra.
4. **Bottleneck thay đổi theo pha.** Trong cửa sổ 00:30–00:43, MASAC BASE 1M dùng 741,14 s learner/750,18 s elapsed, collector wait chỉ thêm 0,082 s. MATD3 BASE vẫn thêm 723,76 s collector wait/765,18 s; MATD3 PQL thêm 737,87 s collector/741,29 s. Vì vậy AMP/compile/fused Adam có thể giúp pha learner-limited, nhưng không xử lý chi phí MAC.
5. **Cấu hình tăng tốc learner hiện đang tắt.** W&B của cả bốn canonical 1M đều ghi amp_mode=off, compile_models=false, fused_optimizers=false. YAML cũng đặt compile/fused=false; amp dùng mặc định off. Code đã hỗ trợ các tùy chọn, nhưng không được nói cohort đang dùng chúng. Bật AMP thay đổi precision và cần gate số học/hiệu năng riêng; giữ nguyên lượt đang chạy.

6. **Động học không giải thích độ chậm khi report 1 MB hoạt động.** Đổi dynamic_window sang reject trong diagnostic tăng 38,74995 lên 38,79999 transition/s, chỉ khoảng 0,13%. Khi tắt mạng, motion vẫn có chi phí đáng kể; không suy rộng kết quả này thành motion luôn miễn phí.

## Task 6: cache bộ đệm nhận

Candidate cb7c5d6 đưa bytes_used của receiver ra ngoài vòng duyệt target trong packet_step. Bộ đệm không đổi trong vòng này, các source sau tính lại capacity, và transfer vẫn kiểm tra admission cuối cùng. Reviewer chưa thấy lỗi logic ở thay đổi này. Các caller khác, như telemetry, không được thay đổi.

Đã chỉnh lại cách ước lượng chi phí: Warp 1.18 thực hiện short-circuit and/or. Với report 1 MB và buffer 3 MB, mỗi UAV chỉ chứa tối đa ba report. Trong vòng chọn target được sửa, không được giả định cả 35 target đều gọi bytes_used; con số naive source×target×target là upper bound không phản ánh workload production này.

Gate kiểm chứng:

- Full suite candidate: **674 passed, 74 skipped, 8 warnings, 1.221 subtests passed**, exit 0, 243,11 s. CUDA cần gate riêng.
- GCS/control và 1KB peer: [T4 version 2](https://www.kaggle.com/code/bjkjbhm/uav-receiver-cache-t4-comparison-20261024/2). Baseline/candidate cùng cấu hình, B,C,C,B,B,C; ba lần mỗi phía; warmup loại khỏi timing; synchronize CUDA trước/sau khoảng đo. Chạy oracle native-vs-Warp hiện có trên CUDA sau A/B.
- Peer 1MB: [T4 supplement](https://www.kaggle.com/code/bjkjbhm/uav-receiver-cache-peer1m-20261024/1), cùng baseline/candidate/seed/4.096 môi trường, hash mọi **140 array** Warp, gồm mt/mt_index, packet/event đang chờ và energy. GCS control không tự chứng minh lợi ích cho peer.

**Trạng thái gate: đang chờ kết quả; chưa tuyên bố candidate tăng tốc production.**

## Continuation và phạm vi

MASAC PQL 1M parent fc14508c dừng time_budget tại 1.066 episode / 7.045.120 transition, child bc18a145 đã tăng vượt prefix. MASAC BASE 500k parent b9d73bd3 dừng tại 4.096 / 10.309.632, child 30e43dda cũng tiến tiếp. Không tạo duplicate, không dừng hoặc thay source của các lượt production khỏe.

Chỉ coi full training khi đủ episodes_completed >= 20.000, episode_budget_complete=true và stop_reason=episode_budget. Finished/time_budget không phải final evaluation. Để hoàn tất cần continuation đúng cùng account/lineage qua nhiều segment và đủ tài nguyên.

Task 1–3 đã có audit ở phiên trước; Task 4 T4 diagnostic được xác minh hoàn tất ở phiên này; Task 5 đã revert vì không tăng tốc. Task 6 đang chờ gate; chưa tuyên bố toàn bộ audit hay vấn đề tốc độ đã được sửa xong.

Nếu cache không đem lại gain lặp lại vượt biến động phép đo, loại candidate. Hướng ưu tiên tiếp theo là giảm các lượt kiểm tra kênh không đổi trong event engine bằng cách tiến tới mốc thay đổi kênh/timeout kế tiếp, với test giữ nguyên timestamp, backoff/RNG, ACK/retry, custody, reward và energy. Mọi phương án đó cần task riêng và T4/native parity; không đổi report size, horizon hoặc khoa học để làm số đo nhanh hơn.

## Nguồn trong repository

- [Snapshot W&B mới nhất](2026-10-24-performance-latest-cohort.json)
- [Snapshot ban đầu và giả định ETA](2026-10-24-performance-live-audit.json)
- [Component isolation T4](2026-10-24-t4-component-isolation.json)
- [Dispatch probe T4](2026-10-24-t4-dispatch-probe.json)
- [Trạng thái tiếp quản](../CURRENT_TASK.md)
