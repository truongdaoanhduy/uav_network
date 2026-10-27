# MASAC crash: xác minh lại và cơ chế xử lý lỗi tổng quát

Ngày 27/10/2026, múi giờ Việt Nam. Baseline `29a8fe4`; bản sửa `820b50ec000eee8e08e0875868f457d104c4057f`. Không khởi chạy đợt training production, không sửa trạng thái/histories W&B và không đổi cấu hình khoa học.

## Account trong key.txt

Đọc và kiểm tra API/quota lúc 18:17:33: file hiện có 10 account enabled. Cả 10 xác thực và trả về quota GPU: account_01 còn 23,35h; account_02 còn 25,03h; account_03–10 mỗi account trả về 30h. Đây không phải kiểm chứng GPU allocation, phone verification, accelerator thực tế, idle kernels hoặc private datasets cho một launch mới.

GitHub Actions chỉ có allowlist/secrets account_01/02. Account khác trong file local chưa tự được đồng bộ sang GitHub. Để dùng chúng trong full flow cần preflight hợp lệ rồi đồng bộ đúng secret/allowlist/identity; không chuyển owner của một lineage cũ hoặc dùng account mới để né giới hạn dịch vụ.

## Nguyên nhân bốn MASAC cũ

Fresh W&B read lúc 18:15–18:17 xác nhận cả bốn learner và bốn controller đều `crashed`, source `77a1d0f52ed4e78bc2823c3ada03d661568cd923`. Dùng trường `heartbeatAt`, không thay bằng `_timestamp` của metric cuối:

| Run | Heartbeat cuối (27/10, giờ VN) | Completed episodes của learner |
|---|---|---:|
| MASAC BASE HYBRID 63202ad9 | 02:32:06 | 8192 |
| MASAC BASE FULLGPU 25eecfd1 | 02:31:57 | 8192 |
| MASAC PQL HYBRID 6f0bcde3 | 02:31:57 | 7045 |
| MASAC PQL FULLGPU 3b39b0c8 | 02:32:08 | 8196 |
| CTRL BASE HYBRID 8e3786a8 | 02:32:01 | — |
| CTRL BASE FULLGPU 45ff19d1 | 02:32:02 | — |
| CTRL PQL HYBRID 4f61b7cd | 02:32:07 | — |
| CTRL PQL FULLGPU 16f5a5f3 | 02:32:02 | — |

Khoảng mất heartbeat là 11 giây. Bằng chứng mạnh nhất vẫn là sự cố dùng chung ở nền tảng/phiên chạy/tài khoản/kết nối/telemetry. Mất heartbeat không tự chứng minh mọi process đã bị kill. Không có thêm traceback hay file lỗi trong danh sách file W&B hiện trả về. Bốn owner cũ (mjhres, njbiugyry, fowieofweoi, kjkjhhfghjjkn) không có credential tương ứng trong file hiện tại nên chưa đọc được log private Kaggle để phân biệt các giả thuyết. Không có bằng chứng mới đủ để kết luận bị ban.

Báo cáo trước ghi nhận metrics hữu hạn, sampled host oomKills=0 và GPU1 FULLGPU PQL từng khoảng 96,55% VRAM. Host oomKills=0 không loại trừ GPU OOM hay một peak không được lấy mẫu. Các job cũ AMP/compile off và dừng trước deadline cấu hình; cập nhật acceleration sau đó không được đưa vào code đã ghim.

Checkpoint cũ thiếu replay/optimizer/RNG. Dùng lại policy là warm-start; không thể khôi phục chính xác 7045/8192/8196 episode đã mất chỉ bằng sửa controller. Xem báo cáo gốc [crash-flow recovery](2026-10-27-crash-flow-recovery.md).

## Lỗi phần mềm được tái hiện và sửa trong task này

Các lỗi dưới đây là khoảng trống của cơ chế phục hồi mới, không được dùng làm lời giải thích thay thế cho sự cố lịch sử:

1. Lỗi list/auth/API của một account thoát khỏi main trước khi kiểm tra các account sau. Đã cô lập exception theo account, giữ báo cáo đã che thông tin nhạy cảm và tiếp tục account còn lại.
2. recovery_blocked/recovery_required/service_unavailable có thể trả exit 0. Các tình trạng chưa giải quyết này giờ trả nonzero, workflow không báo xanh giả. Trạng thái W&B lịch sử giữ nguyên.
3. Dữ liệu recovery thiếu initial_global_step, khiến gate của continuation từ chối. Bổ sung prefix đã được kiểm chứng.
4. Snapshot có world state thật nhưng metadata không công bố environment_state_saved; zero-episode recovery không qua gate. BASE/PQL nay phát marker từ snapshot thực sự được cung cấp cho serializer; tracker ghi marker trong artifact. Thiếu marker ở zero-episode recovery thì dừng, không đoán là đã lưu.
5. Submit recovery trả waiting_for_quota nhưng supervisor báo submitted. Đã giữ đúng trạng thái chờ, không nhận là có child mới.
6. Job cancelled có thể bị phục sinh. Bây giờ không tự adopt/restart trạng thái cancelled. API không đủ thông tin để phân biệt mọi nguồn cancellation, nên cần operator quyết định nếu muốn tiếp tục.
7. Thêm giới hạn ba phiên crash recovery liên tiếp. Counter được giữ trong child spec, chỉ reset sau một segment clean. Account/API/quota waits không tạo child thì không tiêu tốn một lần đã submit. Shared builder cũng kiểm tra budget.
8. Thêm --report và upload supervisor-report.json bằng always() trong Actions. Nội dung chỉ có ref/status/reason code/error class; không ghi raw exception hoặc key.

## Ma trận phương pháp xử lý

| Nhóm lỗi | Hành động | Trạng thái / giới hạn |
|---|---|---|
| Chỉ W&B mất heartbeat, Kaggle còn running | Giữ job đang chạy, tránh tạo bản sao | Đã có gate active trước khi đọc output/recovery |
| Hết thời gian một session, chưa đủ tổng episode | Lưu full state, kiểm chứng, chuyển GPU segment tiếp theo | Giữ cùng owner, config, precision và episode budget |
| Hết quota hoặc API quota tạm lỗi | Chờ quota/API hợp lệ, kiểm tra lại ở cron sau | Đã sửa báo đúng waiting_for_quota ở recovery; không đổi owner |
| Hard kill/preemption, có checkpoint hợp lệ | Resume từ snapshot immutable đã xác nhận | Tối đa 3 recovery liên tiếp; không bảo đảm giữ mọi bước sau snapshot |
| Token/quyền truy cập bị lỗi | Báo account_unavailable, vẫn kiểm tra account khác | Cần khôi phục quyền hợp lệ; không tự kết luận ban |
| NaN/OOM/exception tái diễn | Giữ diagnostic/failure state, thử exact recovery có giới hạn rồi dừng điều tra | Không tự giảm batch/replay/env hay đổi reward để làm thí nghiệm khác |
| Checkpoint thiếu/hỏng/sai run/commit/prefix | Từ chối dùng dữ liệu không được xác minh; báo reason | Loader tiếp tục kiểm tra hash/schema/state; không tái tạo replay đã mất |
| Upload/checkpoint thất bại | Giữ recovery pointer tốt trước đó; staging được dọn; báo lỗi | Đã có bounded backups; không giả định lần upload hỏng là checkpoint mới |
| GPU xong nhưng CPU eval/visualize bị ngắt | Resume từ evaluation-progress, retry ref xác định trước | Đã có retry giới hạn và publication checks; không train lại GPU chỉ vì render lỗi |
| Trạng thái cancelled | Dừng tự động cho lineage đó | Chờ operator; không phục sinh job bị huỷ |
| Hết budget recovery, trạng thái không rõ | Workflow báo cần kiểm tra, giữ checkpoint/history | Circuit breaker có thể chặn cả handoff CPU nếu lỗi xảy ra sau recovery thứ ba; đây là quyết định an toàn có chủ đích |
| Sandbox/laptop tắt | GitHub cron tiếp tục điều phối cloud | Workflow active mỗi 15 phút; lịch GitHub có thể trễ |

Mục tiêu là mỗi nhóm lỗi có hành động hữu hạn và bằng chứng để kiểm tra. Lỗi quyền truy cập, dữ liệu đã mất hoặc lỗi lặp cần can thiệp; không có phần mềm đảm bảo Kaggle không bao giờ ngắt phiên.

## Các bước tiếp theo được đề xuất

1. Đồng bộ thêm account chỉ khi thật sự cần launch trên account đó, sau auth/quota/idle/private-dataset/accelerator preflight. Hiện chưa tự bật 03–10 trên GitHub.
2. Đo upload full state thực tế khoảng 15GB lên HF và peak disk/RAM trước khi giảm khoảng cách backup. Chính sách hiện tại yêu cầu 1h nhưng cap 4 lần/10h làm khoảng hiệu lực khoảng 2,5h cộng thời gian lưu/upload. Không coi đây là giới hạn mất dữ liệu tuyệt đối.
3. Cân nhắc checkpoint đầu sớm khi đã có positive progress nhưng replay còn nhỏ. Nếu hard kill trước checkpoint đầu tiên được xác nhận, hiện chưa thể exact-resume; restart từ seed ban đầu phải có nhãn/authorization rõ ràng.
4. Trước cohort mới, chạy một smoke cloud nhỏ: train → full snapshot → ngắt có kiểm soát → exact recovery → CPU eval/visualize → kiểm chứng W&B/HF. Task này không tự launch production 20k để làm bài test đó.

## Verification / deployment

- RED: 15 failure cases, gồm 4 snapshot BASE/PQL thực tế nhỏ, trước khi sửa.
- Focused GREEN: 135 passed, 38 subtests.
- Full CPU: 868 passed, 95 skipped, 8 warnings, 1383 subtests passed in 506.22s (0:08:26)
- compileall, diff check, lightweight no-torch import: pass. Tracked secret scan so với 12 secret local: 0 hit. Không sửa configs/ hoặc algorithm update logic.
- Independent review: không có Critical/Important. Chấp nhận circuit breaker bảo thủ nói trên. Minor chưa sửa: nếu probe --check-services ở đầu manual dry-run fail, reconcile chưa chạy nên chỉ có console log, chưa có JSON artifact riêng cho probe. Scheduled account/lineage reports không bị ảnh hưởng.
- Feature hosted dry-run [40497048893](https://github.com/truongdaoanhduy/uav_network/actions/runs/40497048893) tại 820b50e: success. Artifact uav-supervisor-report ID 12730654618 đã tải và đọc: account_01=[], account_02=[]. Không submit child.
- Default-branch deployment: main đã deploy source `820b50ec000eee8e08e0875868f457d104c4057f`; [40497938044](https://github.com/truongdaoanhduy/uav_network/actions/runs/40497938044) success; artifact report ID 12731566730, 189 bytes, upload thành công. Kaggle 01/02, W&B và HF probe ok; reconcile hai danh sách rỗng, không submit child.


Fresh W&B query tại 2026-10-27T18:44:05+07:00: 0 running/pending. No production root/child/controller was manually launched, restarted, cancelled, deleted or adopted. Default workflow remains active. New GPU/CUDA execution, actual large HF transfers and a full train → crash → recover → visualize cloud cycle were not performed in this task.
