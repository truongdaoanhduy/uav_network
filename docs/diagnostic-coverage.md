# Độ bao phủ bằng chứng training

Áp dụng cho source cập nhật ngày 28/10/2026. Log phục vụ kiểm chứng nguyên nhân,
không có bảo đảm xác định mọi lỗi. Dữ liệu thiếu giữ nguyên là thiếu; không gán
nguyên nhân từ trạng thái W&B. Chưa chạy lại đợt production để đo chất lượng policy.

| Tình trạng | Bằng chứng đã có / vừa bổ sung | Bước kiểm chứng và giới hạn |
|---|---|---|
| Exception Python / dependency / file / quyền truy cập | failure.json: traceback, exception chain, error_category, phase, seed/source và metrics cuối | Tái hiện cùng config/checkpoint. Wrapper khác root cause; category là phân loại dấu hiệu. |
| RAM hoặc cấp phát Torch/CUDA | RSS/peak RSS, host MemAvailable, CUDA allocated/reserved/peak theo logical device ở journal/failure; telemetry SDK | Chỉ exception cấp phát xác nhận thao tác thất bại; GPU logical index theo CUDA_VISIBLE_DEVICES. Snapshot không bắt được mọi đỉnh; xem log hệ thống/Kaggle trước khi kết luận OOM kill. |
| Hết đĩa / upload checkpoint chậm hoặc lỗi | Disk free, recovery phase/status/elapsed/error và confirmed upload age; exception errno | Phân biệt chụp state với upload. Chỉ confirmed FULL upload mới là tham chiếu phục hồi; cần kiểm tra artifact tải được. |
| HTTP 401/403, mất mạng, timeout | Exception chain/category, traceback, recovery/evaluation events và stderr subprocess | 401/403 chứng minh bị từ chối, không chứng minh tài khoản bị ban. Kiểm tra quyền, endpoint, quota và trạng thái nền tảng hợp lệ. |
| SIGTERM / SIGKILL / fatal native error | SIGTERM thực nhận ghi số/tên; parent returncode âm ghi tên signal; PYTHONFAULTHANDLER stderr | SIGKILL không cho Python ghi failure/cleanup. Cần Kaggle/container/OS log; signal không tự giải thích ai gửi hoặc lý do. |
| Không tiến độ / chờ worker / gọi hàm lâu | thread-stacks.log sau 300s không marker mới, lặp lại; last-progress marker, journal phase/time; PQL queue/rate evidence | Đối chiếu nhiều stack và metrics; backup/eval/hàm dài hợp lệ cũng có thể tạo stack. Không tự kết luận deadlock hoặc kill. Nếu cả máy treo, không ghi được. |
| Chậm nhưng vẫn chạy | TPS/update rates, replay/sample-use, PQL queue/lag có sẵn; CPU/GPU/RAM, checkpoint/eval phase | Tìm nghẽn compute/collector/replay/I/O bằng thay đổi cùng thời điểm. Utilization thấp không tự chứng minh sai thuật toán. |
| NaN/Inf / Q, TD, gradient bất thường | Learner metrics, diagnostics, last finite/nonfinite context; failed_batch/state khi learner hỗ trợ | NaN/Inf được giữ rõ trong log an toàn. Tái hiện batch với seed/precision; magnitude/loss riêng không có ngưỡng tốt chung. |
| Delivery/success thấp | Cohort mean/n/partial/source interval; search, packet PDR, eligible silent fraction, reward/action/safety/report evidence; video cuối | Phân biệt chưa tìm mục tiêu, tìm nhưng không báo về GCS, route/PHY/retry/buffer/TTL hoặc chuyển động bị chặn. Correlation định hướng điều tra, chưa chứng minh quan hệ nhân quả. |
| Suy giảm theo thời gian | Closed cohort regression warning, return/delivery/success trajectories và source steps | So các cửa sổ đủ mẫu cùng scenario. Thay độ khó/tuổi episode và nhiễu seed có thể giả thành suy giảm. |
| Policy im lặng / exploration suy giảm / reward lệch nhiệm vụ | Action fractions, silent-given-eligible, entropy/alpha khi thuật toán có, reward components/accounting warning | Im lặng có thể hợp lý nếu không có receiver/hàng đợi; return tăng mà delivery không tăng cần kiểm tra reward, không tự đổi reward. |
| Motion/safety/energy/report loss | Blocked/shield/stall diagnostics, report counters; J/delivery | So energy khi delivery/success tương đương; không khen tiết kiệm chỉ vì UAV không gửi. Xem record/video và scenario. |
| Replay/queue không cân bằng / stale policy | Replay/PQL counters, sample-use/rates/queue/lag được ghi tại nhịp log cùng source step | BASE segment khác restored PQL counters. Metric không được producer phát sẽ không được journal bịa ra. |
| Điểm eval khó diễn giải / seed overlap / dữ liệu ít | Eval count, actual seed interval/scenario hash/checkpoint artifact/step và overlap có trong protocol; per-episode CPU records | Count là kết quả checkpoint hiện hiển thị, không cộng nhiều vòng. So cùng scenario/budget/seed và nhiều training seed; overlap null là chưa biết, không phải false. |
| Train đủ nhưng eval/render/publication lỗi hoặc pause | Stage, training budget/stop reason, CPU progress, evaluation-status events, metrics.json, visualization failures và GitHub supervisor decisions | Finished của một phiên không đồng nghĩa đủ 20k hoặc video đã đăng. Truy đúng parent/child/artifact và pha lỗi; giữ nguyên kết quả train. |
| W&B không cập nhật / journal ghi lỗi | W&B Logs + file journal/status/diagnostics độc lập; optional I/O/upload warning; run ID/source manifest | Console, đĩa và upload được cô lập best-effort. Nếu cả mạng lẫn đĩa/console mất, không còn kênh để ghi. SDK online bắt buộc lỗi vẫn có thể làm job thất bại theo hợp đồng hiện tại. |
| Mất phiên / máy tắt / heartbeat đồng loạt | Last status/event/resource (có tuổi mẫu), checkpoint xác nhận, named parent exit nếu parent sống | Không dựng lại log đã mất; cần log phía Kaggle/GitHub. Giám sát ngoài tiến trình và full checkpoint giới hạn thiệt hại, không ngăn được mọi sự cố nền tảng. |

Đọc Personal Runs theo thứ tự Name/State/Created/Runtime → Stage → training/eval
count → outcome → budget/stop/failure. Algorithm và Mode ẩn vì Name đã có.
Workspace chính có 27 plot + hướng dẫn; actor loss giữ ở profile diagnostic.
Ô trống không phải 0; mỗi plot tối đa 12 đường riêng, không gộp thuật toán.

Tình trạng có thể là **đang tiến bộ**, **kết quả thấp**, **suy giảm**, **không ổn định**,
**chậm**, **không tiến độ**, **chưa đủ bằng chứng**, **kết thúc một phần**, hoặc **lỗi
hệ thống/eval/publication**. Chúng có thể cùng tồn tại. Không gán nhãn tốt/xấu tự
động chỉ từ một loss hoặc State. Mini-eval của profile chính đang tắt nên đường
training chưa chứng minh generalization; final CPU 30 tập là bằng chứng trên seed
bank đó, chưa thay thế so sánh nhiều training seed và độ bất định.

Lưu ý vận hành: lấy resource snapshot tối đa mỗi 60s khi có event và ngay khi ghi
failure; không có sampler chạy độc lập khi tiến trình treo. Watchdog giữ file
mở đến khi cancel, không reset từ heartbeat không có tiến độ. Thread stack chỉ
chứa vị trí code, không dump locals/tensor. File phụ/upload lỗi không được thay
thế exception chính. Phần bổ sung áp dụng job dùng commit mới, không hồi tố run cũ.
