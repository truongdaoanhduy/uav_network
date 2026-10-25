# Kiểm tra W&B, 1 KiB và Dec-POMDP — 25/10/2026

Đã kiểm tra trực tiếp W&B/Kaggle, xác minh kết quả MASAC và bản xuất media, tái hiện và sửa các lỗi xác nhận được. Mốc W&B cuối: **16:21:55 ngày 25/10/2026, giờ Việt Nam**; project không có run nào Running. Chưa khởi chạy cohort training 20.000 episode mới sau các bản sửa hôm nay.

## Trạng thái các lượt đã chạy

| Lượt | W&B training cuối | Episode | Report/payload | Đánh giá CPU xác minh hôm nay | Ý nghĩa kết quả |
| --- | --- | ---: | --- | --- | --- |
| MASAC BASE | [bb1adf51](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/bb1adf51) | 20.000 | 1024 / 1024 byte | [44656f8c](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/44656f8c), 30/30 | Trước bản sửa local-report hôm nay |
| MASAC PQL | [12582cc7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/12582cc7) | 20.000 | 1024 / 1024 byte | [831a115e](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/831a115e), 30/30 | Trước bản sửa local-report hôm nay |
| MATD3 BASE | [767cb8da](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/767cb8da) | 20.000 | 1024 / 1024 byte | Có bằng chứng finalization ngày 24/10; không tải lại media trong lượt này | Trước cả bản sửa link-state 6263156 |
| MATD3 PQL | [30a40bf7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/30a40bf7) | 20.000 | 1024 / 1024 byte | Có bằng chứng finalization ngày 24/10; không tải lại media trong lượt này | Trước cả bản sửa link-state 6263156 |

Cả bốn training run đều có `episode_budget_complete=true` và `stop_reason=episode_budget`; không suy ra hoàn thành chỉ từ chữ Finished. Hai lượt MATD3 mới sau 6263156 chưa được submit: kiểm tra W&B không tìm thấy; tài khoản dự kiến14/15 không có owned kernel.

**Đã chuyển primary workload từ 1 MB sang 1 KiB**, không chỉ thay tên run: `network.report_bytes=1024`, `payload_bytes=1024`, `buffer_bytes=3072`. Profile là `configs/comparison-20k-hybrid75-1kb.yaml`. Các profile1 MB còn lại được giữ làm lịch sử/stress test. Replay1 triệu transition là một tham số khác với kích thước report.

Cấu hình thực tế hiện tại: env4096, batch8192, replay1M, CPU75%/GPU25%, 35 target, 25 obstacle, horizon1800, seed44; V3 của repo hiện tại là thế giới XY1000×1000m, altitude100m. Seed final evaluation thực tế là **2000000044**, không phải10044 trong prompt bàn giao cũ.

Hai lượt MASAC đã hoàn tất cả training và pipeline CPU/HF/W&B: `_cpu_complete=true`; tải mới video, ảnh và metrics từ immutable HF URL đều khớp kích thước/SHA256. Kết quả 30 episode: BASE search74,10%, delivery25,24%, success0/30; PQL search77,71%, delivery33,62%, success0/30. Đây là kết quả trước bản sửa hôm nay, không phải kết quả chất lượng của code mới.

## Dec-POMDP/CTDE và các bản sửa

V3 có PA/TX/RX energy, MAC/retry hay critic tập trung không tự làm sai Dec-POMDP. Actor phải quyết định từ observation riêng; critic được dùng state toàn cục trong giai đoạn training. Đây là cấu trúc [centralized training, decentralized execution](https://arxiv.org/html/1706.02275v4). Policy chỉ dùng observation hiện tại là một lớp policy hạn chế; không bắt buộc phải thêm RNN để được gọi là decentralized. Chính sách cục bộ của Dec-POMDP dựa trên thông tin quan sát của agent, như mô tả trong [tài liệu nghiên cứu Dec-POMDP](https://icaps20subpages.icaps-conference.org/in.bgu.ac.il/en/natural_science/cs/Documents/ICAPS2020-DMAP-WS-Papers/A%20Factored%20Approach%20To%20Solving%20Dec-POMDPs.pdf).

Vấn đề được sửa là các đường thông tin không phù hợp với giả định local sensing/radio đã công bố trong project:

| Vấn đề | Hành vi trước sửa | Cách sửa/bằng chứng |
| --- | --- | --- |
| Link-state ngoài radio range | Actor thấy metadata của node không liên lạc được | Bản sửa cũ6263156 đã có trong source training MASAC; xác minh qua GitHub |
| Cờ GCS nhận report làm lộ thông tin từ xa | `ReportStore.discover` và Warp `sense_reports_at` nhìn `delivered[target]` toàn cục để bỏ tạo report hoặc xóa recovery; UAV ở xa suy ra GCS receipt qua buffer/TTL | **c05806a**: bỏ kiểm tra toàn cục khỏi sensing, giữ local capacity/TTL; GCS vẫn deduplicate delivery. TDD7fail→pass; Warp/CUDA kiểm tra riêng |
| Checkpoint khác nghĩa observation nhưng cùng dimension vẫn được nạp | Có thể nối checkpoint cũ vào code mới và trộn thí nghiệm | **2f4b5a8**: lưu/kiểm tra `observation_version=3` cho policy, actor snapshot, BASE/PQL resume; missing/older bị từ chối. Hash protocol và metadata phản ánh phiên bản |
| Trục evaluation CPU và checkpoint step bị sai | Episode logs dùng x=episode nhưng SDK/panel ép x=checkpoint step thiếu/0 | **66a4b70**: mapping riêng năm metric CPU sang `evaluation/episodes_completed`; giữ trục checkpoint cho aggregate validation; ghi actual checkpoint step |
| Xuất lại media lịch sử có thể bị gắn observation hiện tại | Recovery không evaluate lại policy nhưng metadata mặc định phiên bản runtime | **66a4b70**: source version ban đầu unknown, sau đó lấy từ saved protocol; lịch sử thiếu version vẫn unknown |

Routing vẫn do policy chọn next-hop; không thêm heuristic chọn peer gần GCS thay actor. Hai file thuật toán MASAC/MATD3 và primary1KiB config được so sánh byte/diff với HEAD đầu phiên: không thay đổi. Không bật AMP/compile/fused hoặc đổi reward/scenario để làm đẹp kết quả.

## Kết quả cũ và các quyết định

Mọi checkpoint đã train trước c05806a, gồm hai lượt MASAC vừa hoàn thành, chỉ dùng làm lịch sử dưới mô hình local sensing đã sửa. Cohort đúng nghĩa observation mới phải bắt đầu episode0, giữ cùng profile so sánh giữa bốn thuật toán. Loader mới chặn resume sai phiên bản; muốn phân tích checkpoint cũ thì dùng source cũ đã pin.

Đã gắn `historical-pre-local-report-fix` cho sáu MASAC root/child/CPU run và đọc lại xác nhận. MATD3 cũ đã có `obsolete-pre-decpomdp-fix`. Summary CPU checkpoint step được sửa từ0 thành35.840.000 và35.139.584 dựa trên saved protocol. Không ghi lại history/return/search/delivery hay thay config/source của thí nghiệm cũ. Code mới sửa trục cho các lần log sau; không tuyên bố đã tái ghi lịch sử biểu đồ cũ.

Quyết định từ review: nâng lỗi provenance khi xuất media lịch sử thành mức cần sửa vì nhãn sai có thể làm trộn cohort nghiên cứu. Đã sửa và kiểm thử. Không có minor nào được giữ lại như một lỗi xác nhận chưa sửa. Giữ branch/worktree hiện tại và push theo AGENTS.md; không merge main hay tự khởi tạo replacement20k trong task audit này.

Giữ các giả định đang được công bố: beacon/admission metadata lý tưởng trong radio range, không tính riêng airtime control-plane; safety transition có tương tác giữa UAV; reward chung và critic privileged state. Những điều này cần nêu trong paper, không tự động được coi là actor leakage. Không chuyển toàn bộ control-plane thành packet model khác trong bản sửa này.

## Kiểm thử và giới hạn xác minh

- Baseline trước sửa:717passed,74skipped; full cuối: **739 passed, 77 skipped, 1270 subtests, exit 0**. Có8 warning phụ thuộc/deprecation và test cleanup đã tồn tại.
- Real Kaggle T4×2 diagnostic tại source c05806a: **78 passed, 0 skipped, 12 subtests, exit 0**; kiểm tra sensing locality và physics/packet/vector regression. Không gọi các test CUDA bị skip trên host là đã chạy.
- Guard checkpoint:10test RED rồi GREEN; các test restore/evaluation/full-suite cùng phiên bản vẫn qua.
- CPU reporting/publication/async-axis:36test tập trung qua; có real toy CPU checkpoint evaluation, provider network được thay ở ranh giới test. Kiểm tra SDK W&B0.30 thật ở offline mode xác nhận custom metric definitions và provenance; production tracker vẫn từ chối offline mode.
- Inventory/AST parse toàn bộ **100 module source**, **97 module test** và nạp/validate **21 YAML config** đều qua. Có SHA256 từng file trong inventory. Review độc lập đọc sâu learning/replay/restore/observation/custody/MAC/evaluation/cloud; root bổ sung PHY, motion, config, optimizer/precision và rendering.
- Không phải chứng minh không còn bug hay một lượt đọc tay từng dòng của mọi utility/probe/vendored dependency. Phạm vi đọc cụ thể nằm trong báo cáo review độc lập. Không suy ra native/Warp tương đương cho mọi sự kiện, PQL async resume bitwise deterministic, hay thuật toán nào tốt hơn từ một seed và30 scenario. Không chạy lại full cloud training→CPU E2E hoặc benchmark production memory sau bản sửa mới; không có kết quả chất lượng sau retraining code mới.

## Bằng chứng và source

- `2026-10-25-masac-base-postfix-finalization.json`, `2026-10-25-masac-pql-postfix-finalization.json`.
- `2026-10-25-final-live-status.json`, `2026-10-25-final-running-check.json`.
- `2026-10-25-local-report-sensing.json`, `2026-10-25-local-report-cuda-result.json`, `2026-10-25-local-report-cuda.log`.
- `2026-10-25-observation-version-guard.json`, `2026-10-25-cpu-reporting-fix.json`.
- `2026-10-25-independent-correctness-review.md`, `2026-10-25-final-module-config-inventory.json`.
- `2026-10-25-historical-observation-labels.json`, `2026-10-25-offline-sdk-axis-check.json`.

Nhánh: `feat/cpu-replay-20261022`. Source fix cuối: **66a4b7025a46c8249ff79760a3211b0e02075b4d**, đã push và xác minh qua GitHub. Hai source fix trước: c05806a4a1ff772d7062a0850342df4853f686b0 và2f4b5a8. Commit docs sau báo cáo này không thay code đã kiểm thử.
