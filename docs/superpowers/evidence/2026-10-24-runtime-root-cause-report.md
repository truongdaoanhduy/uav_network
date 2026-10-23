# Audit thời gian training 20.000 episode

W&B đọc trực tiếp lúc **01:22 ngày 24/10/2026 (UTC+7)**. Task 6 đã hoàn tất với kết quả âm và candidate đã được gỡ. Vấn đề tốc độ tổng thể còn cần công việc tiếp theo; báo cáo này không tuyên bố đã audit xong từng module.

## Có đủ 20.000 episode trong segment hiện tại không?

**Không.** MASAC BASE 1M và MATD3 BASE 1M đã ghi stop_reason=time_budget, episode_budget_complete=false sau khoảng 10 giờ training. MASAC BASE W&B đã finished; notebook Kaggle vẫn đang kết thúc. MATD3 BASE W&B/Kaggle vẫn running khi đọc, nhưng training đã ghi time_budget. Chưa thấy W&B child mới của hai parent này tại thời điểm snapshot. Đây là trạng thái đóng segment; chưa đủ bằng chứng để kết luận treo hay tự ý tạo một child khác.

| Lượt | Episode hoàn tất | Global step | Trạng thái training | Tốc độ TB segment (transition/s) | Giờ GPU còn lại, ngoại suy |
| --- | ---: | ---: | --- | ---: | ---: |
| [MASAC BASE 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/63faeffb) | 650 | 6,983,680 | kết thúc time_budget | 193.8 | 13–42 |
| [MASAC PQL 1M c01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bc18a145) | 3,815 | 7,294,976 | running | 52.4 | 143–165 |
| [MATD3 BASE 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/2dcbf36c) | 0 | 3,825,664 | kết thúc time_budget | 106.2 | 41–84 |
| [MATD3 PQL 1M](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/278634f6) | 0 | 3,379,200 | running | 96.1 | 45–94 |
| [MASAC BASE 500k c01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/30e43dda) | 4,096 | 11,083,776 | running | 121.0 | 53–57 |

Với trung bình lịch sử 1.710,259 step/episode, 20.000 episode tương ứng khoảng **34,2 triệu transition**; nếu cả 20.000 episode đều chạm trần 1.800 thì là **36 triệu**. Muốn xong từ đầu trong 10 giờ cần khoảng **950–1.000 transition/s**, chưa tính preflight, checkpoint, quota, hàng đợi và phần dở dang của batch vector cuối. 36 triệu không phải hard ceiling của global_step trong môi trường vector.

ETA dùng tốc độ trung bình segment và 50/200 log tiến độ gần nhất có global_step khác nhau, theo đồng hồ run/elapsed_seconds; bỏ phần idle sau training khi save/upload. Phép ngoại suy thay đổi mạnh theo pha, không phải khoảng tin cậy hay giới hạn trên chắc chắn. PQL 1M child mới có khoảng 52 transition/s, thấp hơn parent, nên không được dùng tốc độ parent để hứa thời điểm hoàn tất. Giờ lịch còn phụ thuộc tài nguyên và continuation.

MATD3 có 0 episode hoàn tất vì 4.096 môi trường của đợt đầu mới tới step 934/825, dưới horizon 1.800. Global step và episode_step vẫn tăng; số 0 này không chứng minh training ngừng cập nhật.

## Nguyên nhân có bằng chứng

1. **Kịch bản report 1 MB nhân chi phí mô phỏng packet chính xác.** Một report 1.000.000 byte chia thành 977 fragment 1.024 byte, cộng ACK, backoff, retry, collision và interference. T4 diagnostic cùng 4.096 môi trường, seed44, dynamic_window đo 9.726,67 transition/s ở 1 KB và 38,74995 ở 1 MB, chậm khoảng 251 lần. Đây là phép đo simulator, không phải tốc độ full training. Nó chỉ ra tương tác giữa kích thước report và cách triển khai event engine; chưa có bằng chứng thuật toán RL sai.
2. **MAC polling là vị trí cần profile sâu hơn.** Probe instrumentation cbc7253 có FOV rộng hơn production, đếm 4.735.399.723 dispatch qua ba vector step; IDLE và LISTEN chiếm 97,814% số lượt. Đây là tỷ lệ dispatch, không phải tỷ lệ thời gian và không phải số đo production. Kết quả gợi ý rất nhiều lần kiểm tra kênh không đổi, cần đo và giảm mà vẫn giữ chính xác timestamp/RNG/backoff/ACK.
3. **Hai GPU chia vai trò môi trường và learner.** GPU0 chạy cả 4.096 môi trường; GPU1 chạy learner/replay. Utilization lệch không chứng minh GPU hỏng. Không tìm thấy bằng chứng lỗi phần cứng trong phạm vi kiểm tra. oomKills=0 ở các mẫu system đã đọc chỉ là chỉ số host, không đủ để loại trừ mọi dạng CUDA OOM.
4. **Bottleneck thay đổi theo pha.** Cửa sổ 00:30–00:43: MASAC BASE 1M thêm 741,14 s learner/750,18 s elapsed, collector wait chỉ thêm 0,082 s. MATD3 BASE thêm 723,76 s collector wait/765,18 s; MATD3 PQL thêm 737,87 s collector/741,29 s. PQL chạy overlap nên không cộng các timer để suy ra tổng phần trăm. AMP/compile/fused có thể giúp pha learner, không tự loại chi phí mô phỏng packet.
5. **Tăng tốc learner hiện tắt.** Cả bốn canonical 1M ghi amp_mode=off, compile_models=false, fused_optimizers=false trong W&B. YAML cũng đặt compile/fused=false; AMP mặc định off. Code hỗ trợ các tùy chọn nhưng cohort không dùng chúng. AMP thay đổi precision, nên cần gate số học/hiệu năng riêng trước một lượt tương lai; không đổi nóng các run đang tiến triển.
6. **Motion không giải thích chi phí chính của fixture 1 MB.** Đổi dynamic_window sang reject chỉ tăng 38,74995 lên 38,79999 transition/s, khoảng 0,13%. Khi tắt mạng, motion vẫn có chi phí; không suy rộng thành motion luôn rẻ.

## Task 6: receiver buffer cache — COMPLETE, REVERTED

Candidate cb7c5d6 đưa bytes_used(receiver) ra ngoài vòng chọn target ở uav_network/communication/gpu/mac.py. Reviewer chưa thấy sai logic trên phạm vi đã đọc. Nhưng Warp 1.18 thực hiện short-circuit and/or; report 1 MB với buffer 3 MB nghĩa là source chỉ chứa tối đa ba report. Vì vậy cách đếm naive source×target×target đã phóng đại số lần bytes_used thực sự chạy trong production.

T4 A/B ghim baseline 4cd9358 và candidate cb7c5d6 trong hai cây source riêng, cùng dependency/runtime, 4.096 môi trường, seed44, thứ tự B,C,C,B,B,C, ba lần mỗi bản. Warmup loại khỏi timing, synchronize CUDA trước/sau. Peer 1 MB hash **toàn bộ 140 array đã khởi tạo**, gồm mt/mt_index, packet/event đang chờ và energy; cả sáu trial khớp. Hai case control hash 24 array cũng khớp.

| Case | Baseline median transition/s | Candidate median transition/s | Chênh lệch |
| --- | ---: | ---: | ---: |
| active_1kb_peer | 787.74247218 | 787.55057233 | -0.02436% |
| active_1mb_gcs | 38.49941272 | 38.48308302 | -0.04242% |
| active_1mb_peer | 39.17247858 | 39.16647574 | -0.01532% |

**Không có tăng tốc đo được.** Các chênh lệch rất nhỏ, trong mức dao động phép đo; không kết luận một regression đáng kể từ các con số này. Đã gỡ candidate và test chỉ kiểm tra source; mac.py và tests/test_performance.py hiện khớp byte với baseline 4cd9358.

Xác minh hoàn tất:

- Full suite sau revert: **673 passed, 74 skipped, 8 warnings, 1222 subtests passed in 254.25s (0:04:14)**, exit 0. Đây là full suite CPU của host; các case CUDA tùy chọn còn skip.
- Native–Warp oracle trên candidate/T4 CUDA: **38 test đạt**, 12,333 s, exit 0. Log lưu cùng evidence.
- Peer 1 MB: sáu trial, 140 array/trial khớp, errors=0; control: 12 kết quả case, 24 array/trial khớp, errors=0.
- Reviewer độc lập xác nhận quyết định revert và hai file khớp baseline. State parity áp dụng cho các trajectory đã đo, không phải chứng minh mọi tình huống.

Kaggle cố định: [control version 2](https://www.kaggle.com/code/bjkjbhm/uav-receiver-cache-t4-comparison-20261024/2), [peer 1 MB version 1](https://www.kaggle.com/code/bjkjbhm/uav-receiver-cache-peer1m-20261024/1). Control version 1 dừng trước simulation vì diagnostic đặt sai runtime.device=cuda:0; version 2 sửa thành cuda, smoke kiểm tra trước khi chạy. Không dùng lỗi phiên bản cũ để kết luận simulation hỏng.

## Continuation và thay đổi vận hành

MASAC PQL 1M parent fc14508c dừng time_budget tại 1.066 episode / 7.045.120 transition; child bc18a145 đã vượt prefix và tăng episode tiếp. MASAC BASE 500k parent b9d73bd3 dừng tại 4.096 / 10.309.632; child 30e43dda cũng tiến tiếp. Giữ các child có tiến độ thực, không duplicate, không dừng hoặc đổi source của run khỏe.

Watcher UAV 1M Cohort Recovery từng bị assistant phiên trước tắt lúc 21:22 ngày 23/10 vì Sandbox Resource not found. Phiên này xác minh các kết nối đọc được, bật lại watcher theo giờ, pin recovery controller/source đã kiểm thử 4cd9358, giữ cùng account/lineage và loại experimental performance commits. API xác nhận enabled=true; next_run_time vẫn null và **chưa xác minh lần chạy lịch mới**. Watcher là fallback, khác auto-continuation trong notebook; chưa được coi là cơ chế đã chạy thành công sau khôi phục.

Chỉ công nhận full training khi episodes_completed >= 20.000, episode_budget_complete=true và stop_reason=episode_budget. Finished/time_budget chỉ kết thúc một segment. Cần continuation đúng checkpoint/replay/environment/prefix, cùng account, đủ quota hợp lệ qua nhiều segment. Không đổi account để né quota.

Task 1–3 có audit ở phiên trước; Task 4 T4 diagnostic đã xác minh hoàn tất; Task 5 và Task 6 đều gỡ candidate vì không tăng tốc. Không đưa một tối ưu chưa có lợi ích vào production. Hướng task tiếp theo: profile IDLE/LISTEN và nghiên cứu tiến tới mốc thay đổi kênh/timeout, giữ đúng timestamp, RNG/backoff, ACK/retry, custody/TTL/admission, energy, reward và observations. Chỉ bắt đầu khi có chỉ dẫn tiếp theo theo quy tắc một task mỗi lần.

## Evidence và tài liệu chính thức

- [Snapshot W&B hiện tại](2026-10-24-performance-latest-cohort.json), [snapshot 00:43](2026-10-24-performance-cohort-0043.json), [snapshot đầu và giả định ETA](2026-10-24-performance-live-audit.json).
- [Component isolation](2026-10-24-t4-component-isolation.json), [dispatch counts](2026-10-24-t4-dispatch-probe.json).
- [Peer 1 MB A/B](2026-10-24-t4-receiver-cache-peer1m.json), [control A/B](2026-10-24-t4-receiver-cache-control.json), [CUDA oracle log](2026-10-24-t4-receiver-cache-cuda-oracle.log), [final verification](2026-10-24-task6-final-verification.json).
- [Watcher restored](2026-10-24-recovery-watch-restored.json), [trạng thái tiếp quản](../CURRENT_TASK.md).
- NVIDIA Warp: [timing/profiling và đồng bộ CUDA](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/profiling.html).
- PyTorch: [AMP recipe, autocast và GradScaler](https://docs.pytorch.org/tutorials/recipes/amp_recipe.html), [performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html). Context7 và source Warp 1.18 cài trên host được dùng để kiểm tra API/short-circuit; số đo project đến từ các artifact riêng ở trên.
